"""
SWCD Panel Data ETL Pipeline

panel_data.json 파일을 DB에 적재하는 파이프라인:
입력 → 데이터 전처리 → 비정형 데이터 임베딩 → DB 저장

사용 예시:
    python backend/scripts/etl_pipeline.py --input backend/data/panel_data.json
"""

import argparse
import asyncio
import json
import os
import sys
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 프로젝트 루트 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(Path(PROJECT_ROOT) / ".env")

# LangChain + Bedrock
try:
    from langchain_aws import ChatBedrock  # type: ignore
    import boto3  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    _HAS_BEDROCK = True
except ImportError:
    ChatBedrock = None  # type: ignore
    boto3 = None  # type: ignore
    _HAS_BEDROCK = False

# KoSimCSE 임베딩 모델
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_KOSIMCSE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _HAS_KOSIMCSE = False
    print("⚠️ Warning: sentence-transformers not installed. Embedding features will be disabled.")


# ===== DB 연결 설정 =====
raw_url = os.getenv(
    "DATABASE_URL",
    "postgresql://swcd:swcdpw@127.0.0.1:5432/swcddb"
).replace("+asyncpg", "")  # 동기 연결

# RDS 연결인 경우 SSL 설정 추가 (rds.amazonaws.com 포함 시)
# psycopg2는 sslmode 파라미터를 지원
if "rds.amazonaws.com" in raw_url and "sslmode" not in raw_url:
    separator = "&" if "?" in raw_url else "?"
    DATABASE_URL = f"{raw_url}{separator}sslmode=require"
else:
    DATABASE_URL = raw_url

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


# ===== 전역 변수 =====
_KOSIMCSE_MODEL = None
_BEDROCK_SESSION = None
_BEDROCK_REGION = None


# ===== AWS Bedrock 설정 =====
def get_bedrock_config():
    """Bedrock API 키 및 설정 로드"""
    global _BEDROCK_SESSION, _BEDROCK_REGION
    
    if _BEDROCK_SESSION and _BEDROCK_REGION:
        return _BEDROCK_SESSION, _BEDROCK_REGION
    
    bedrock_key_encoded = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_BEDROCK_API_KEY")
    if not bedrock_key_encoded:
        return None, None
    
    try:
        decoded_key = base64.b64decode(bedrock_key_encoded).decode("utf-8")
        if ":" in decoded_key:
            parts = decoded_key.split(":", 1)
            access_key = parts[0]
            secret_key = parts[1] if len(parts) > 1 else ""
        else:
            access_key = decoded_key
            secret_key = ""
    except (UnicodeDecodeError, Exception):
        try:
            decoded_bytes = base64.b64decode(bedrock_key_encoded)
            bed_key_marker = b'BedrockAPIKey'
            start_idx = decoded_bytes.find(bed_key_marker)
            if start_idx > 0:
                actual_key_bytes = decoded_bytes[start_idx:]
                decoded_key = actual_key_bytes.decode("utf-8", errors="ignore")
            elif len(decoded_bytes) > 2 and decoded_bytes[0:1] == b'\x00':
                decoded_key = decoded_bytes[2:].decode("utf-8", errors="replace")
            else:
                decoded_key = decoded_bytes.decode("latin-1", errors="ignore")
            decoded_key = decoded_key.strip("\x00").strip()
            if ":" in decoded_key:
                parts = decoded_key.split(":", 1)
                access_key = parts[0]
                secret_key = parts[1] if len(parts) > 1 else ""
            else:
                access_key = decoded_key
                secret_key = ""
        except Exception:
            if ":" in bedrock_key_encoded:
                parts = bedrock_key_encoded.split(":", 1)
                access_key = parts[0]
                secret_key = parts[1] if len(parts) > 1 else ""
            else:
                access_key = bedrock_key_encoded
                secret_key = ""
    
    region = os.getenv("AWS_REGION", "us-west-2")
    
    if access_key and secret_key:
        _BEDROCK_SESSION = boto3.Session(  # type: ignore
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    else:
        _BEDROCK_SESSION = boto3.Session(region_name=region)  # type: ignore
    
    _BEDROCK_REGION = region
    return _BEDROCK_SESSION, _BEDROCK_REGION


def get_bedrock_llm(model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
    """Bedrock Claude Haiku LLM 초기화"""
    if not _HAS_BEDROCK:
        return None
    
    session, region = get_bedrock_config()
    if not session or not region:
        return None
    
    return ChatBedrock(  # type: ignore[call-arg]
        model_id=model_id,  # type: ignore[arg-type]
        credentials_profile_name=None,
        region_name=region,  # type: ignore[arg-type]
        model_kwargs={"temperature": 0.7, "max_tokens": 1000}
    )


# ===== KoSimCSE 임베딩 모델 =====
def get_embedding_model():
    """KoSimCSE 모델 반환 (768 차원)"""
    global _KOSIMCSE_MODEL
    
    if not _HAS_KOSIMCSE:
        return None
    
    if _KOSIMCSE_MODEL is None:
        print("🔄 KoSimCSE 임베딩 모델 로딩 중...")
        _KOSIMCSE_MODEL = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')  # type: ignore
        print("✅ KoSimCSE 모델 로드 완료")
    
    return _KOSIMCSE_MODEL


# ===== 데이터 전처리 =====
def parse_array_field(value: Any) -> List[str]:
    """배열 필드 파싱 (문자열 또는 리스트 형태 모두 처리)"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        # 문자열 형태의 배열 파싱: "['TV', '냉장고']" 또는 "[TV, 냉장고]"
        try:
            # ast.literal_eval 사용
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v]
        except:
            # 파싱 실패 시 빈 리스트 반환
            pass
    return []


def parse_income(income: Any) -> Optional[int]:
    """소득 문자열을 숫자로 파싱 (예: "월 500~599만원" -> 500)"""
    if income is None:
        return None
    if isinstance(income, (int, float)):
        return int(income)
    if isinstance(income, str):
        # "월 500~599만원" 형식에서 첫 번째 숫자 추출
        import re
        match = re.search(r'(\d+)', income)
        if match:
            return int(match.group(1))
    return None


def parse_car_ownership(value: Any) -> Optional[bool]:
    """차량 보유 여부 파싱 ("있다"/"없다" -> True/False)"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value == "있다" or "있" in value:
            return True
        elif value == "없다" or "없" in value:
            return False
    return None


def preprocess_panel_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """panel_data.json 형식의 데이터를 DB 스키마에 맞게 전처리"""
    panel_id_str = str(raw_data.get("panel_id", "")).strip()
    
    # 성별 (그대로 저장, "남성"/"여성")
    gender = raw_data.get("gender") or None
    
    # 나이 (float를 int로 변환)
    age = None
    if raw_data.get("age") is not None:
        try:
            age = int(float(raw_data.get("age", 0)))
        except:
            age = None
    
    # 지역
    region_city = raw_data.get("region_city") or None
    region_gu = raw_data.get("region_district") or None  # region_district -> region_gu
    
    # 결혼 여부
    marital_status = raw_data.get("marital_status") or None
    
    # 자녀수
    children_count = None
    if raw_data.get("children_count") is not None:
        try:
            children_count = int(float(raw_data.get("children_count", 0)))
        except:
            children_count = None
    
    # 가족수 (family_members -> family_size)
    family_size = None
    if raw_data.get("family_members") is not None:
        try:
            family_size = int(float(raw_data.get("family_members", 0)))
        except:
            family_size = None
    
    # 학력 (education -> education_level), 신규 키도 허용
    education_level = raw_data.get("education") or raw_data.get("education_level") or None
    
    # 직업 (job -> occupation)
    occupation = raw_data.get("job") or None
    
    # 소득
    # 구형/신규 키 모두 허용
    monthly_personal_income = parse_income(
        raw_data.get("monthly_personal_income") or raw_data.get("income_personal_monthly")
    )
    monthly_household_income = parse_income(
        raw_data.get("monthly_household_monthly") or raw_data.get("income_household_monthly")
    )
    
    # 휴대폰
    phone_brand = raw_data.get("phone_brand") or None
    phone_model = raw_data.get("phone_model") or None
    
    # 차량
    car_ownership = parse_car_ownership(raw_data.get("car_ownership"))
    car_manufacturer = raw_data.get("car_manufacturer") or None
    car_model = raw_data.get("car_model") or None
    
    # 배열 필드 파싱
    # 구형/신규 키 모두 허용
    owned_electronics = parse_array_field(
        raw_data.get("owned_electronics") or raw_data.get("electronics_owned_multi")
    )
    smoking_experience = parse_array_field(
        raw_data.get("smoking_experience") or raw_data.get("smoking_experience_multi_label")
    )
    smoking_brand = parse_array_field(
        raw_data.get("smoking_brands") or raw_data.get("smoking_brand_multi_label")
    )
    e_cig_heated_brand = parse_array_field(
        raw_data.get("heated_tobacco_brands") or raw_data.get("smoking_brand_cigarette_heat_multi_label")
    )
    e_cig_liquid_brand = parse_array_field(
        raw_data.get("liquid_ecig_brands") or raw_data.get("smoking_brand_liquid_vape_multi_label")
    )
    drinking_experience = parse_array_field(
        raw_data.get("drinking_experience") or raw_data.get("drinking_experience_multi_label")
    )
    
    return {
        "panel_id": panel_id_str,
        "gender": gender,
        "age": age,  # 정수로 변환됨
        "region_city": region_city,
        "region_gu": region_gu,
        "marital_status": marital_status,
        "children_count": children_count,  # 정수로 변환됨
        "family_size": family_size,  # 정수로 변환됨
        "education_level": education_level,
        "occupation": occupation,
        "monthly_personal_income": monthly_personal_income,
        "monthly_household_income": monthly_household_income,
        "phone_brand": phone_brand,
        "phone_model": phone_model,
        "car_ownership": car_ownership,
        "car_manufacturer": car_manufacturer,
        "car_model": car_model,
        "owned_electronics": owned_electronics,
        "smoking_experience": smoking_experience,
        "smoking_brand": smoking_brand,
        "e_cig_heated_brand": e_cig_heated_brand,
        "e_cig_liquid_brand": e_cig_liquid_brand,
        "drinking_experience": drinking_experience,
    }


# 필드명 → 한국어 매핑 (column_metadata에 없는 필드용)
FIELD_NAME_KO_MAP = {
    "fitness_management_method": "운동 관리 방법",
    "chatbot_experience": "챗봇 사용 경험",
    "chatbot_main_purpose": "챗봇 주요 사용 목적",
    "main_chatbot_used": "주로 사용하는 챗봇",
    "preferred_chatbot": "선호하는 챗봇",
    "ai_usage_field": "AI 사용 분야",
    "main_apps_used": "주로 사용하는 앱",
    "ott_service_count": "이용하는 OTT 서비스 개수",
    "skincare_spending_monthly": "월 스킨케어 지출",
    "skincare_considerations": "스킨케어 제품 선택 시 고려사항",
    "skin_satisfaction": "피부 만족도",
    "most_effective_diet_experience": "가장 효과적이었던 다이어트 경험",
    "most_saved_photos_topic": "가장 많이 저장하는 사진 주제",
    "preferred_spending_category": "선호하는 소비 카테고리",
    "high_spending_category": "높은 지출 카테고리",
    "preferred_new_year_gift": "선호하는 설 선물",
    "preferred_water_play_area": "선호하는 물놀이 장소",
    "preferred_overseas_destination": "선호하는 해외 여행지",
    "preferred_summer_snack": "선호하는 여름 간식",
    "memorable_childhood_winter_activity": "기억에 남는 어린 시절 겨울 활동",
    "travel_style": "여행 스타일",
    "traditional_market_visit_frequency": "전통시장 방문 빈도",
    "main_quick_delivery_products": "주로 주문하는 퀵배송 상품",
    "reward_points_interest": "리워드 포인트 관심도",
    "lifestyle_values": "라이프스타일 가치관",
    "privacy_habits": "개인정보 보호 습관",
    "reducing_plastic_bags": "비닐봉지 사용 줄이기 방법",
    "rainy_day_coping_method": "비 오는 날 대처 방법",
    "late_night_snack_method": "야식 섭취 방법",
    "morning_wakeup_method": "아침 기상 방법",
    "solo_dining_frequency": "혼밥 빈도",
    "preferred_chocolate_situation": "초콜릿을 선호하는 상황",
    "moving_stress_factors": "이사 스트레스 요인",
    "conditions_for_happy_old_age": "행복한 노후를 위한 조건",
    "pets": "반려동물 경험",
    "stress_factors": "스트레스 요인",
    "stress_relief_method": "스트레스 해소 방법",
    "summer_fashion_essential": "여름 패션 필수품",
    "summer_sweat_discomfort": "여름 땀 불편함",
    "summer_worries": "여름 걱정",
    "waste_disposal_method": "쓰레기 처리 방법",
}


def load_column_metadata() -> Dict[str, Dict[str, Any]]:
    """column_metadata.json을 로드하여 필드명 → 메타데이터 매핑 반환"""
    metadata_path = Path(PROJECT_ROOT) / "backend" / "data" / "column_metadata.json"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ column_metadata.json 로드 실패: {e}")
        return {}


def field_name_to_korean(field_name: str) -> str:
    """필드명을 한국어로 변환"""
    # 매핑 테이블 확인
    if field_name.lower() in FIELD_NAME_KO_MAP:
        return FIELD_NAME_KO_MAP[field_name.lower()]
    
    # snake_case를 한글로 변환 시도
    words = field_name.lower().split("_")
    # 간단한 영어 단어 → 한글 변환 (migrate_qa_format.py와 동일)
    word_map = {
        "fitness": "운동", "management": "관리", "method": "방법",
        "chatbot": "챗봇", "experience": "경험", "main": "주요", "purpose": "목적",
        "used": "사용", "preferred": "선호", "ai": "AI", "usage": "사용", "field": "분야",
        "apps": "앱", "ott": "OTT", "service": "서비스", "count": "개수",
        "skincare": "스킨케어", "spending": "지출", "monthly": "월",
        "considerations": "고려사항", "skin": "피부", "satisfaction": "만족도",
        "most": "가장", "effective": "효과적인", "diet": "다이어트",
        "saved": "저장한", "photos": "사진", "topic": "주제",
        "spending": "지출", "category": "카테고리", "high": "높은",
        "new": "새해", "year": "년", "gift": "선물",
        "water": "물", "play": "놀이", "area": "장소",
        "overseas": "해외", "destination": "여행지",
        "summer": "여름", "snack": "간식",
        "memorable": "기억에 남는", "childhood": "어린 시절", "winter": "겨울", "activity": "활동",
        "travel": "여행", "style": "스타일",
        "traditional": "전통", "market": "시장", "visit": "방문", "frequency": "빈도",
        "quick": "퀵", "delivery": "배송", "products": "상품",
        "reward": "리워드", "points": "포인트", "interest": "관심도",
        "lifestyle": "라이프스타일", "values": "가치관",
        "privacy": "개인정보", "habits": "습관",
        "reducing": "줄이기", "plastic": "비닐", "bags": "봉지",
        "rainy": "비 오는", "day": "날", "coping": "대처",
        "late": "늦은", "night": "밤",
        "morning": "아침", "wakeup": "기상",
        "solo": "혼자", "dining": "식사",
        "chocolate": "초콜릿", "situation": "상황",
        "moving": "이사", "stress": "스트레스", "factors": "요인",
        "conditions": "조건", "for": "을 위한", "happy": "행복한", "old": "노후", "age": "나이",
        "pets": "반려동물",
    }
    
    translated = []
    for word in words:
        if word in word_map:
            translated.append(word_map[word])
        else:
            translated.append(word)
    
    return " ".join(translated)


def generate_question(field_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """필드명과 메타데이터를 기반으로 질문 생성
    
    Args:
        field_name: 필드명 (예: "fitness_management_method")
        metadata: column_metadata.json의 메타데이터 (선택사항)
    
    Returns:
        질문 텍스트 (예: "운동 관리 방법은 무엇인가요?")
    """
    if metadata and "name_ko" in metadata:
        field_name_ko = metadata["name_ko"]
    else:
        field_name_ko = field_name_to_korean(field_name)
    
    # 조사 처리
    last_char = field_name_ko[-1]
    if ord(last_char) >= 0xAC00 and ord(last_char) <= 0xD7A3:
        if (ord(last_char) - 0xAC00) % 28 == 0:
            return f"{field_name_ko}는 무엇인가요?"
        else:
            return f"{field_name_ko}은 무엇인가요?"
    else:
        return f"{field_name_ko}은 무엇인가요?"


def extract_summary_segments(panel_data: Dict[str, Any], include_question: bool = True) -> List[Dict[str, str]]:
    """비정형 데이터: drinking_experience_multi_label 이후의 모든 필드를 세그먼트로 추출
    
    panel_data.json에서 drinking_experience_multi_label 이후에 나오는 모든 필드들을
    비정형 데이터로 처리하여 각각을 개별 세그먼트로 저장합니다.
    
    Args:
        panel_data: 패널 데이터 딕셔너리
        include_question: 질문을 포함할지 여부 (기본값: True)
    
    Returns:
        세그먼트 리스트 (segment_name, summary_text 포함)
    """
    segments = []
    
    # column_metadata 로드 (질문 생성을 위해)
    column_metadata = load_column_metadata() if include_question else {}
    
    # 모든 키를 순서대로 가져오기
    all_keys = list(panel_data.keys())
    
    # drinking_experience_multi_label 필드의 인덱스 찾기
    drinking_idx = next(
        (i for i, k in enumerate(all_keys) if k == "drinking_experience_multi_label"),
        -1
    )
    
    if drinking_idx < 0:
        # drinking_experience_multi_label이 없으면 빈 리스트 반환
        return segments
    
    # drinking_experience_multi_label 이후의 모든 필드를 비정형 데이터로 처리
    unstructured_keys = all_keys[drinking_idx + 1:]
    
    for field_name in unstructured_keys:
        value = panel_data.get(field_name)
        
        # 값이 None이 아니고, 문자열이거나 숫자/불린인 경우 텍스트로 변환
        if value is not None:
            answer_text = None
            
            if isinstance(value, (str, int, float, bool)):
                text = str(value).strip()
                if text and text.lower() != "null":
                    answer_text = text
            elif isinstance(value, list):
                # 리스트인 경우 쉼표로 구분된 문자열로 변환
                text = ", ".join(str(v) for v in value if v is not None).strip()
                if text:
                    answer_text = text
            
            if answer_text:
                # 질문 제거: 답변 텍스트만 사용 (검색 노이즈 제거)
                # 질문 패턴 제거: "~은 무엇인가요?", "~는 무엇인가요?" 등
                import re
                # 질문 패턴 제거 (예: "운동 관리 방법은 무엇인가요? 달리기" → "달리기")
                question_patterns = [
                    r'^[^?]*?은\s*무엇인가요\s*\?',
                    r'^[^?]*?는\s*무엇인가요\s*\?',
                    r'^[^?]*?을\s*무엇인가요\s*\?',
                    r'^[^?]*?를\s*무엇인가요\s*\?',
                ]
                
                cleaned_text = answer_text
                for pattern in question_patterns:
                    cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
                
                # 앞뒤 공백 제거
                cleaned_text = cleaned_text.strip()
                
                # 질문만 남고 답변이 없으면 원본 사용
                if not cleaned_text:
                    cleaned_text = answer_text
                
                # 질문 생성 (디버깅/로깅용, 실제 저장에는 사용 안 함)
                if include_question:
                    metadata = column_metadata.get(field_name)
                    question = generate_question(field_name, metadata)
                    # 질문은 로깅용으로만 사용, 실제 저장은 답변만
                    summary_text = cleaned_text  # 질문 제거된 답변만 저장
                else:
                    summary_text = cleaned_text
                
                    segments.append({
                    "segment_name": field_name.upper(),  # 필드명을 대문자로 변환하여 세그먼트 이름으로 사용
                    "summary_text": summary_text,
                    })
    
    return segments


# ===== LangChain Chain: 비정형 데이터 임베딩 생성 =====
def create_embedding_chain(embedding_model):
    """비정형 데이터(카테고리별 요약)를 KoSimCSE로 임베딩하는 Chain"""
    
    def generate_embeddings(segments: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """비정형 데이터를 벡터로 변환"""
        if not embedding_model:
            return []
        
        results = []
        for segment in segments:
            try:
                embedding = embedding_model.encode(
                    segment["summary_text"],
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).tolist()
                results.append({
                    **segment,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"  ⚠️ 임베딩 생성 실패 ({segment['segment_name']}): {e}")
        
        return results
    
    return RunnableLambda(generate_embeddings)


# ===== LLM 요약 생성 (패널 1-2줄 프로필) =====
def create_profile_summary_chain(bedrock_llm):
    """패널 JSON으로부터 1-2줄 핵심 요약 프로필 생성 Chain
    
    - null 정보는 언급 금지
    - 뼈대: age, gender, job, region_city
    - 살: income/차량/AI 사용/OTT/여행성향/보유전자제품 등 1~2개 핵심
    """
    if not bedrock_llm:
        # LLM이 없으면 빈 문자열 반환
        return RunnableLambda(lambda panel: "")
    
    from json import dumps as json_dumps
    
    def pick_fields(panel: Dict[str, Any]) -> Dict[str, Any]:
        # 기본 뼈대
        core_keys = [
            "age", "gender", "job", "region_city"
        ]
        # 주요 상세
        detail_keys = [
            "income_household_monthly", "income_personal_monthly",
            "car_manufacturer", "car_model", "ai_usage_field",
            "ott_service_count", "travel_style", "lifestyle_values",
            "electronics_owned_multi"
        ]
        sel: Dict[str, Any] = {}
        for k in core_keys + detail_keys:
            if k in panel and panel.get(k) is not None:
                sel[k] = panel.get(k)
        return sel
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 64개 컬럼의 패널 JSON에서 1-2줄 요약 프로필을 작성하는 AI다. "
         "규칙: 1) null/결측은 언급 금지. 2) 가능하면 age, gender, job, region_city를 뼈대로 사용. "
         "3) 그 외에는 income/차량/AI 사용/OTT/여행성향/보유전자제품 등 성향·경제력·디지털 수용도를 드러내는 핵심 1~2개만 선택. "
         "4) 자연스러운 한국어 서술형 1~2문장, 불릿/머릿말/사족 금지. 5) 과장/추측 금지."),
        ("human",
         "패널 JSON(요약용 필드만 추출됨):\n{panel_json}\n\n"
         "출력: 한국어 1~2문장으로 간결 요약.")
    ])
    
    chain = (
        RunnableLambda(lambda raw: {"panel_json": json_dumps(pick_fields(raw), ensure_ascii=False)})
        | prompt
        | bedrock_llm
        | StrOutputParser()
    )
    return chain




# ===== LangChain Chain: 통합 ETL 파이프라인 =====
def create_etl_pipeline_chain(embedding_model, enable_summary: bool = False, bedrock_llm=None):
    """ETL 전체 파이프라인 Chain (요약 생성 제외)
    
    프로세스:
    1. 데이터 전처리
    2. 비정형 데이터 추출 (G1~G7)
    3. 비정형 데이터 임베딩 생성
    4. (옵션) 패널 1-2줄 프로필 요약 생성
    """
    
    # Step 1: 데이터 전처리 및 비정형 데이터 추출
    def preprocess_and_extract(raw_panel: Dict[str, Any]) -> Dict[str, Any]:
        """전처리 및 비정형 데이터 추출"""
        panel_data = preprocess_panel_data(raw_panel)
        segments = extract_summary_segments(raw_panel)
        
        return {
            "raw_panel": raw_panel,
            "panel_data": panel_data,
            "segments": segments,
        }
    
    # Step 2: 비정형 데이터 임베딩 생성
    embedding_chain = create_embedding_chain(embedding_model)
    # Step 3: (옵션) LLM 요약 생성
    summary_chain = create_profile_summary_chain(bedrock_llm) if enable_summary else RunnableLambda(lambda _: "")
    
    # Chain 조합
    pipeline = (
        RunnableLambda(preprocess_and_extract)
        | RunnableLambda(lambda x: {
            **x,
            "segments_with_embeddings": embedding_chain.invoke(x["segments"])
        })
        | RunnableLambda(lambda x: {
            **x,
            "panel_summary_text": summary_chain.invoke(x["raw_panel"]) if enable_summary else ""
        })
    )
    
    return pipeline


# ===== DB 저장 =====
def save_to_db(
    panel_data: Dict[str, Any],
    segments_with_embeddings: List[Dict[str, Any]],
    db_session,
    summary_text: Optional[str] = None
):
    """DB에 패널 데이터 저장 (요약 텍스트 없음)"""
    panel_id = panel_data["panel_id"]
    
    # 1. panels 테이블 저장 또는 업데이트 (요약 텍스트만 업데이트)
    with db_session.begin():
        # 기존 패널이 있으면 요약 텍스트만 업데이트, 없으면 새로 삽입
        db_session.execute(
            text("""
                INSERT INTO panels (
                    panel_id, gender, age, region_city, region_gu, marital_status, children_count, family_size,
                    education_level, occupation, monthly_personal_income, monthly_household_income,
                    phone_brand, phone_model, car_ownership, car_manufacturer, car_model,
                    owned_electronics, smoking_experience, smoking_brand,
                    e_cig_heated_brand, e_cig_liquid_brand, drinking_experience,
                    panel_summary_text, search_labels
                ) VALUES (
                    :panel_id, :gender, :age, :region_city, :region_gu, :marital_status, :children_count, :family_size,
                    :education_level, :occupation, :monthly_personal_income, :monthly_household_income,
                    :phone_brand, :phone_model, :car_ownership, :car_manufacturer, :car_model,
                    :owned_electronics, :smoking_experience, :smoking_brand,
                    :e_cig_heated_brand, :e_cig_liquid_brand, :drinking_experience,
                    :panel_summary_text, :search_labels
                )
                ON CONFLICT (panel_id) 
                DO UPDATE SET 
                    -- 기본 필드가 NULL인 경우에만 업데이트 (기존 값 보존)
                    gender = COALESCE(panels.gender, EXCLUDED.gender),
                    age = COALESCE(panels.age, EXCLUDED.age),
                    region_city = COALESCE(panels.region_city, EXCLUDED.region_city),
                    region_gu = COALESCE(panels.region_gu, EXCLUDED.region_gu),
                    -- 요약 텍스트는 항상 업데이트
                    panel_summary_text = EXCLUDED.panel_summary_text,
                    updated_at = NOW()
            """),
            {
                **panel_data,
                "panel_summary_text": (summary_text or None) if summary_text else None,
                "search_labels": [],
            }
        )
        # 로그 출력 최소화 (속도 개선)
        # print(f"  ✓ panels 테이블에 패널 '{panel_id}' {'업데이트' if summary_text else '저장'} 완료")
    
    # 2. panel_summary_segments 테이블 저장 (UPSERT)
    for segment in segments_with_embeddings:
        if "embedding" not in segment:
            continue
        
        segment_id = f"{panel_id}-{segment['segment_name']}"
        try:
            with db_session.begin():
                db_session.execute(
                    text("""
                        INSERT INTO panel_summary_segments (
                            panel_id, segment_name, summary_text, embedding, ts_vector_korean
                        ) VALUES (
                            :panel_id, :segment_name, :summary_text,
                            CAST(:embedding AS vector), to_tsvector('korean', :summary_text)
                        )
                        ON CONFLICT (panel_id, segment_name) 
                        DO UPDATE SET 
                            summary_text = EXCLUDED.summary_text,
                            embedding = EXCLUDED.embedding,
                            ts_vector_korean = EXCLUDED.ts_vector_korean
                    """),
                    {
                        "panel_id": panel_id,
                        "segment_name": segment["segment_name"],
                        "summary_text": segment["summary_text"],
                        "embedding": f"[{','.join(map(str, segment['embedding']))}]",
                    }
                )
            # 로그 출력 최소화 (속도 개선)
            # print(f"  ✓ 세그먼트 '{segment_id}' 저장 완료")
        except Exception as e:
            print(f"  ⚠️ 세그먼트 '{segment_id}' 저장 실패: {e}")


# ===== 메인 ETL 파이프라인 =====
def load_json_to_db(json_file_path: str):
    """panel_data.json 파일을 읽어 DB에 적재 (요약 생성 없음)"""
    print(f"📂 JSON 파일 읽기: {json_file_path}")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        panels_data = [data]
    elif isinstance(data, list):
        panels_data = data
    else:
        raise ValueError("JSON은 객체 또는 객체 배열이어야 합니다.")

    print(f"📊 총 {len(panels_data)}개 패널 데이터 발견")
    
    # 초기화
    embedding_model = get_embedding_model()
    
    if embedding_model is None:
        raise RuntimeError("KoSimCSE Embedding Model 초기화 실패")
    
    # LLM 초기화 (옵션)
    enable_summary_env = os.getenv("ETL_ENABLE_SUMMARY", "false").lower() in ("1", "true", "yes")
    bedrock_llm = get_bedrock_llm() if enable_summary_env else None
    
    # LangChain Chain 생성
    etl_chain = create_etl_pipeline_chain(
        embedding_model=embedding_model,
        enable_summary=bool(bedrock_llm),
        bedrock_llm=bedrock_llm
    )
    
    db = SessionLocal()
    
    try:
        # 이미 처리된 패널 ID 조회 (스킵용)
        existing_panels_result = db.execute(text("SELECT panel_id FROM panels WHERE panel_summary_text IS NOT NULL"))
        existing_panel_ids = {row[0] for row in existing_panels_result}
        print(f"📋 이미 처리된 패널: {len(existing_panel_ids)}개 (스킵)")
        
        # 처리되지 않은 패널만 필터링
        panels_to_process = [p for p in panels_data if p.get("panel_id") not in existing_panel_ids]
        print(f"📊 처리 대상 패널: {len(panels_to_process)}개 (전체 {len(panels_data)}개 중)")
        
        if not panels_to_process:
            print("✅ 모든 패널이 이미 처리되었습니다.")
            return
        
        processed_count = 0
        # 성능 최적화: 배치 크기 및 동시성 증가
        batch_size = 50  # 배치 크기 증가 (15 → 50)
        max_concurrency = 10  # 동시성 증가 (3 → 10)
        
        # 배치 단위로 처리
        for batch_start in range(0, len(panels_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(panels_to_process))
            batch_panels = panels_to_process[batch_start:batch_end]
            
            if batch_start % 50 == 0 or batch_start == 0:
                print(f"\n[{batch_start}/{len(panels_to_process)}] 📝 배치 처리 중... (진행률: {batch_start/len(panels_to_process)*100:.2f}%)")
            
            try:
                # 배치로 Chain 처리 (LLM 호출 병렬화, AWS Bedrock 제한 고려)
                # 동시성 증가: 성능 향상을 위해 동시성 증가
                batch_results = etl_chain.batch(batch_panels, config={"max_concurrency": max_concurrency})
                
                # 각 결과를 DB에 저장
                for idx, processed in enumerate(batch_results):
                    raw_panel = batch_panels[idx]
                    panel_id = raw_panel.get("panel_id", "")
                    
                    try:
                        # 타입 확인
                        panel_data = cast(Dict[str, Any], processed.get("panel_data"))
                        segments_with_embeddings = cast(List[Dict[str, Any]], processed.get("segments_with_embeddings") or [])
                        summary_text = cast(Optional[str], processed.get("panel_summary_text")) or None
                        
                        if not isinstance(panel_data, dict):
                            print(f"  ⚠️ 패널 '{panel_id}' 데이터 형식 오류: {type(panel_data)}")
                            continue
                        
                        # DB 저장 (요약 텍스트 포함)
                        save_to_db(
                            panel_data,
                            segments_with_embeddings,
                            db,
                            summary_text=summary_text
                        )
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"  ⚠️ 패널 '{panel_id}' 저장 실패: {e}")
                        continue
                
                # 배치마다 커밋하여 진행 상황 확인 가능하도록
                db.commit()
                if batch_start % 50 == 0:
                    print(f"  ✅ {processed_count}개 패널 커밋 완료 (진행률: {processed_count/len(panels_to_process)*100:.2f}%)")
                
                # 배치 간 최소 대기(Throttling 완화, 성능 최적화)
                import time
                time.sleep(0.3)  # 대기 시간 감소 (2초 → 0.3초) - 성능 향상
                
            except Exception as e:
                print(f"  ⚠️ 배치 [{batch_start}-{batch_end}] 처리 실패: {e}")
                # ThrottlingException 백오프 재시도
                err_str = str(e)
                if "ThrottlingException" in err_str or "Too many requests" in err_str:
                    import time, random
                    backoff = 10 + random.uniform(0, 5)  # 대기 시간 감소 (15-25초 → 10-15초)
                    print(f"   ⏳ Throttling 감지. {backoff:.1f}s 대기 후 1회 재시도합니다.")
                    time.sleep(backoff)
                    try:
                        batch_results = etl_chain.batch(batch_panels, config={"max_concurrency": 5})  # 재시도 시 동시성 감소 (10 → 5)
                        for idx, processed in enumerate(batch_results):
                            raw_panel = batch_panels[idx]
                            panel_id = raw_panel.get("panel_id", "")
                            try:
                                panel_data = cast(Dict[str, Any], processed.get("panel_data"))
                                segments_with_embeddings = cast(List[Dict[str, Any]], processed.get("segments_with_embeddings") or [])
                                summary_text = cast(Optional[str], processed.get("panel_summary_text")) or None
                                
                                if isinstance(panel_data, dict):
                                    save_to_db(panel_data, segments_with_embeddings, db, summary_text=summary_text)
                                    processed_count += 1
                            except Exception as e2:
                                print(f"  ⚠️ 패널 '{panel_id}' 저장 실패(재시도): {e2}")
                                continue
                        db.commit()
                        import time as _time
                        _time.sleep(0.5)  # 대기 시간 감소 (2초 → 0.5초)
                        continue  # 다음 배치로
                    except Exception as e3:
                        print(f"   ⚠️ 재시도 실패: {e3}")
                # 개별 패널로 폴백 처리
                for raw_panel in batch_panels:
                    panel_id = raw_panel.get("panel_id", "")
                    try:
                        processed = cast(Dict[str, Any], etl_chain.invoke(raw_panel))
                        panel_data = cast(Dict[str, Any], processed.get("panel_data"))
                        segments_with_embeddings = cast(List[Dict[str, Any]], processed.get("segments_with_embeddings") or [])
                        summary_text = cast(Optional[str], processed.get("panel_summary_text")) or None
                        
                        if isinstance(panel_data, dict):
                            save_to_db(panel_data, segments_with_embeddings, db, summary_text=summary_text)
                            processed_count += 1
                    except Exception as e2:
                        print(f"  ⚠️ 패널 '{panel_id}' 폴백 처리 실패: {e2}")
                        continue
                db.commit()
        
        # 마지막 남은 데이터 커밋
        db.commit()
        print(f"\n✅ 총 {processed_count}/{len(panels_to_process)}개 패널 ETL 완료")
        
    except Exception as e:
        db.rollback()
        print(f"❌ ETL 오류: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


# ===== 메타데이터 적재 함수들 =====
# label.json 키 → DB 컬럼명 매핑
LABEL_TO_DB_COLUMN = {
    "region_district": "region_gu",
    "family_members": "family_size",
    "job": "occupation",
    "income_personal_monthly": "monthly_personal_income",
    "income_household_monthly": "monthly_household_income",
    "electronics_owned_multi": "owned_electronics",
    "smoking_experience_multi_label": "smoking_experience",
    "smoking_brand_multi_label": "smoking_brand",
    "smoking_brand_cigarette_heat_multi_label": "e_cig_heated_brand",
    "smoking_brand_liquid_vape_multi_label": "e_cig_liquid_brand",
    "drinking_experience_multi_label": "drinking_experience",
}


def load_column_metadata(session, metadata_path: Path):
    """column_metadata.json을 DB에 적재"""
    print("📊 컬럼 메타데이터 적재 중...")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    for column_name, meta in metadata.items():
        # range 처리
        range_min = meta.get("range", [None, None])[0] if isinstance(meta.get("range"), list) else None
        range_max = meta.get("range", [None, None])[1] if isinstance(meta.get("range"), list) else None
        
        session.execute(
            text("""
                INSERT INTO column_metadata (
                    column_name, name_ko, name_en, type, description, unit,
                    range_min, range_max, analysis_priority, chart_types, statistics
                ) VALUES (
                    :column_name, :name_ko, :name_en, :type, :description, :unit,
                    :range_min, :range_max, :analysis_priority, :chart_types, :statistics
                )
                ON CONFLICT (column_name) DO UPDATE SET
                    name_ko = EXCLUDED.name_ko,
                    name_en = EXCLUDED.name_en,
                    type = EXCLUDED.type,
                    description = EXCLUDED.description,
                    unit = EXCLUDED.unit,
                    range_min = EXCLUDED.range_min,
                    range_max = EXCLUDED.range_max,
                    analysis_priority = EXCLUDED.analysis_priority,
                    chart_types = EXCLUDED.chart_types,
                    statistics = EXCLUDED.statistics,
                    updated_at = NOW()
            """),
            {
                "column_name": column_name,
                "name_ko": meta.get("name_ko"),
                "name_en": meta.get("name_en"),
                "type": meta.get("type"),
                "description": meta.get("description"),
                "unit": meta.get("unit"),
                "range_min": range_min,
                "range_max": range_max,
                "analysis_priority": meta.get("analysis_priority"),
                "chart_types": meta.get("chart_types", []),
                "statistics": meta.get("statistics", []),
            }
        )
    
    session.commit()
    print(f"  ✅ {len(metadata)}개 컬럼 메타데이터 적재 완료")


def load_label_values(session, label_path: Path):
    """label.json을 DB에 적재"""
    print("🏷️ 라벨 값 적재 중...")
    
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    
    total_values = 0
    
    for label_key, values in labels.items():
        # label.json 키를 DB 컬럼명으로 변환
        column_name = LABEL_TO_DB_COLUMN.get(label_key, label_key)
        
        # 먼저 컬럼이 column_metadata에 있는지 확인
        result = session.execute(
            text("SELECT column_name FROM column_metadata WHERE column_name = :col"),
            {"col": column_name}
        ).fetchone()
        
        if not result:
            print(f"  ⚠️ 컬럼 '{column_name}' (label.json 키: '{label_key}')이 column_metadata에 없습니다. 건너뜁니다.")
            continue
        
        # 기존 값 비활성화
        session.execute(
            text("UPDATE label_values SET is_active = FALSE WHERE column_name = :col"),
            {"col": column_name}
        )
        
        # 새 값 삽입
        for idx, value in enumerate(values):
            # 값 타입 판단
            if isinstance(value, (int, float)):
                value_type = "number"
                value_str = str(value)
            elif isinstance(value, bool):
                value_type = "boolean"
                value_str = str(value)
            else:
                value_type = "string"
                value_str = str(value)
            
            session.execute(
                text("""
                    INSERT INTO label_values (
                        column_name, value, value_type, display_order, is_active
                    ) VALUES (
                        :column_name, :value, :value_type, :display_order, TRUE
                    )
                    ON CONFLICT (column_name, value) DO UPDATE SET
                        value_type = EXCLUDED.value_type,
                        display_order = EXCLUDED.display_order,
                        is_active = TRUE,
                        updated_at = NOW()
                """),
                {
                    "column_name": column_name,
                    "value": value_str,
                    "value_type": value_type,
                    "display_order": idx,
                }
            )
            total_values += 1
    
    session.commit()
    print(f"  ✅ {total_values}개 라벨 값 적재 완료")


def load_category_groups(session, groups_path: Path):
    """category_groups.json을 DB에 적재"""
    print("📁 카테고리 그룹 적재 중...")
    
    with open(groups_path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    
    for group_key, group_data in groups.items():
        # 카테고리 그룹 삽입
        session.execute(
            text("""
                INSERT INTO category_groups (
                    group_key, name_ko, name_en, description, analysis_focus
                ) VALUES (
                    :group_key, :name_ko, :name_en, :description, :analysis_focus
                )
                ON CONFLICT (group_key) DO UPDATE SET
                    name_ko = EXCLUDED.name_ko,
                    name_en = EXCLUDED.name_en,
                    description = EXCLUDED.description,
                    analysis_focus = EXCLUDED.analysis_focus,
                    updated_at = NOW()
            """),
            {
                "group_key": group_key,
                "name_ko": group_data.get("name_ko"),
                "name_en": group_data.get("name_en"),
                "description": group_data.get("description"),
                "analysis_focus": group_data.get("analysis_focus", []),
            }
        )
        
        # 그룹-컬럼 매핑 삽입
        fields = group_data.get("fields", [])
        for idx, column_name in enumerate(fields):
            # 비정형 필드 목록 (세그먼트로 저장되는 필드들 - column_metadata에 없어도 매핑 가능)
            unstructured_fields = {
                "fitness_management_method", "skin_satisfaction", "skincare_spending_monthly",
                "skincare_considerations", "most_effective_diet_experience", "summer_worries",
                "summer_sweat_discomfort", "conditions_for_happy_old_age", "ai_usage_field",
                "most_saved_photos_topic", "ott_service_count", "main_apps_used",
                "chatbot_experience", "main_chatbot_used", "chatbot_main_purpose",
                "preferred_chatbot", "preferred_new_year_gift", "main_quick_delivery_products",
                "reward_points_interest", "preferred_spending_category", "high_spending_category",
                "preferred_water_play_area", "travel_style", "traditional_market_visit_frequency",
                "preferred_overseas_destination", "memorable_childhood_winter_activity",
                "preferred_summer_snack", "stress_factors", "stress_relief_method",
                "moving_stress_factors", "rainy_day_coping_method", "privacy_habits",
                "preferred_chocolate_situation", "waste_disposal_method", "morning_wakeup_method",
                "late_night_snack_method", "reducing_plastic_bags", "solo_dining_frequency",
                "summer_fashion_essential", "pets", "lifestyle_values"
            }
            
            # 비정형 필드는 column_metadata 체크 없이 매핑 가능
            if column_name not in unstructured_fields:
                # 정형 필드는 column_metadata에 존재하는지 확인
                result = session.execute(
                    text("SELECT column_name FROM column_metadata WHERE column_name = :col"),
                    {"col": column_name}
                ).fetchone()
                
                if not result:
                    print(f"  ⚠️ 컬럼 '{column_name}'이 column_metadata에 없습니다. 건너뜁니다.")
                    continue
            
            # 매핑 삽입 (비정형 필드도 포함)
            session.execute(
                text("""
                    INSERT INTO category_group_columns (
                        group_key, column_name, display_order
                    ) VALUES (
                        :group_key, :column_name, :display_order
                    )
                    ON CONFLICT (group_key, column_name) DO UPDATE SET
                        display_order = EXCLUDED.display_order
                """),
                {
                    "group_key": group_key,
                    "column_name": column_name,
                    "display_order": idx,
                }
            )
    
    session.commit()
    print(f"  ✅ {len(groups)}개 카테고리 그룹 적재 완료")


# ===== 비정형 필드 메타데이터 =====
UNSTRUCTURED_FIELDS_METADATA = {
    "fitness_management_method": {
        "name_ko": "체력 관리 방법",
        "name_en": "Fitness Management Method",
        "type": "text",
        "description": "체력 관리 및 운동 방법",
        "analysis_priority": "medium"
    },
    "skin_satisfaction": {
        "name_ko": "피부 만족도",
        "name_en": "Skin Satisfaction",
        "type": "text",
        "description": "피부 상태에 대한 만족도",
        "analysis_priority": "low"
    },
    "skincare_spending_monthly": {
        "name_ko": "월 스킨케어 지출",
        "name_en": "Monthly Skincare Spending",
        "type": "text",
        "description": "월간 스킨케어 제품 지출",
        "analysis_priority": "medium"
    },
    "skincare_considerations": {
        "name_ko": "스킨케어 고려사항",
        "name_en": "Skincare Considerations",
        "type": "text",
        "description": "스킨케어 제품 선택 시 고려사항",
        "analysis_priority": "medium"
    },
    "most_effective_diet_experience": {
        "name_ko": "가장 효과적인 다이어트 경험",
        "name_en": "Most Effective Diet Experience",
        "type": "text",
        "description": "가장 효과적이었던 다이어트 방법",
        "analysis_priority": "medium"
    },
    "summer_worries": {
        "name_ko": "여름 걱정사항",
        "name_en": "Summer Worries",
        "type": "text",
        "description": "여름철 걱정되는 사항",
        "analysis_priority": "low"
    },
    "summer_sweat_discomfort": {
        "name_ko": "여름 땀 불편감",
        "name_en": "Summer Sweat Discomfort",
        "type": "text",
        "description": "여름철 땀으로 인한 불편감",
        "analysis_priority": "low"
    },
    "conditions_for_happy_old_age": {
        "name_ko": "행복한 노후 조건",
        "name_en": "Conditions for Happy Old Age",
        "type": "text",
        "description": "행복한 노후를 위한 조건",
        "analysis_priority": "medium"
    },
    "ai_usage_field": {
        "name_ko": "AI 사용 분야",
        "name_en": "AI Usage Field",
        "type": "text",
        "description": "AI를 활용하는 분야",
        "analysis_priority": "high"
    },
    "most_saved_photos_topic": {
        "name_ko": "가장 많이 저장한 사진 주제",
        "name_en": "Most Saved Photos Topic",
        "type": "text",
        "description": "가장 많이 저장하는 사진의 주제",
        "analysis_priority": "low"
    },
    "ott_service_count": {
        "name_ko": "OTT 서비스 이용 개수",
        "name_en": "OTT Service Count",
        "type": "text",
        "description": "이용 중인 OTT 서비스 개수",
        "analysis_priority": "medium"
    },
    "main_apps_used": {
        "name_ko": "주로 사용하는 앱",
        "name_en": "Main Apps Used",
        "type": "text",
        "description": "주로 사용하는 모바일 앱",
        "analysis_priority": "medium"
    },
    "chatbot_experience": {
        "name_ko": "챗봇 경험",
        "name_en": "Chatbot Experience",
        "type": "text",
        "description": "챗봇 사용 경험",
        "analysis_priority": "medium"
    },
    "main_chatbot_used": {
        "name_ko": "주로 사용하는 챗봇",
        "name_en": "Main Chatbot Used",
        "type": "text",
        "description": "주로 사용하는 챗봇 서비스",
        "analysis_priority": "medium"
    },
    "chatbot_main_purpose": {
        "name_ko": "챗봇 주요 목적",
        "name_en": "Chatbot Main Purpose",
        "type": "text",
        "description": "챗봇 사용의 주요 목적",
        "analysis_priority": "medium"
    },
    "preferred_chatbot": {
        "name_ko": "선호하는 챗봇",
        "name_en": "Preferred Chatbot",
        "type": "text",
        "description": "선호하는 챗봇 서비스",
        "analysis_priority": "medium"
    },
    "preferred_new_year_gift": {
        "name_ko": "선호하는 설 선물",
        "name_en": "Preferred New Year Gift",
        "type": "text",
        "description": "설날에 선호하는 선물",
        "analysis_priority": "low"
    },
    "main_quick_delivery_products": {
        "name_ko": "주로 주문하는 빠른 배송 상품",
        "name_en": "Main Quick Delivery Products",
        "type": "text",
        "description": "빠른 배송 서비스를 통해 주로 주문하는 상품",
        "analysis_priority": "medium"
    },
    "reward_points_interest": {
        "name_ko": "리워드 포인트 관심도",
        "name_en": "Reward Points Interest",
        "type": "text",
        "description": "리워드 포인트에 대한 관심도",
        "analysis_priority": "medium"
    },
    "preferred_spending_category": {
        "name_ko": "선호하는 지출 카테고리",
        "name_en": "Preferred Spending Category",
        "type": "text",
        "description": "선호하는 소비 카테고리",
        "analysis_priority": "high"
    },
    "high_spending_category": {
        "name_ko": "높은 지출 카테고리",
        "name_en": "High Spending Category",
        "type": "text",
        "description": "지출이 높은 카테고리",
        "analysis_priority": "high"
    },
    "preferred_water_play_area": {
        "name_ko": "선호하는 물놀이 장소",
        "name_en": "Preferred Water Play Area",
        "type": "text",
        "description": "물놀이를 선호하는 장소",
        "analysis_priority": "low"
    },
    "travel_style": {
        "name_ko": "여행 스타일",
        "name_en": "Travel Style",
        "type": "text",
        "description": "선호하는 여행 스타일",
        "analysis_priority": "medium"
    },
    "traditional_market_visit_frequency": {
        "name_ko": "전통시장 방문 빈도",
        "name_en": "Traditional Market Visit Frequency",
        "type": "text",
        "description": "전통시장 방문 빈도",
        "analysis_priority": "low"
    },
    "preferred_overseas_destination": {
        "name_ko": "선호하는 해외 여행지",
        "name_en": "Preferred Overseas Destination",
        "type": "text",
        "description": "선호하는 해외 여행지",
        "analysis_priority": "medium"
    },
    "memorable_childhood_winter_activity": {
        "name_ko": "기억에 남는 어린 시절 겨울 활동",
        "name_en": "Memorable Childhood Winter Activity",
        "type": "text",
        "description": "어린 시절 겨울에 기억에 남는 활동",
        "analysis_priority": "low"
    },
    "preferred_summer_snack": {
        "name_ko": "선호하는 여름 간식",
        "name_en": "Preferred Summer Snack",
        "type": "text",
        "description": "여름에 선호하는 간식",
        "analysis_priority": "low"
    },
    "stress_factors": {
        "name_ko": "스트레스 요인",
        "name_en": "Stress Factors",
        "type": "text",
        "description": "주요 스트레스 요인",
        "analysis_priority": "high"
    },
    "stress_relief_method": {
        "name_ko": "스트레스 해소 방법",
        "name_en": "Stress Relief Method",
        "type": "text",
        "description": "스트레스를 해소하는 방법",
        "analysis_priority": "high"
    },
    "moving_stress_factors": {
        "name_ko": "이사 스트레스 요인",
        "name_en": "Moving Stress Factors",
        "type": "text",
        "description": "이사 시 느끼는 스트레스 요인",
        "analysis_priority": "low"
    },
    "rainy_day_coping_method": {
        "name_ko": "우울한 날 대처 방법",
        "name_en": "Rainy Day Coping Method",
        "type": "text",
        "description": "우울하거나 비 오는 날 대처 방법",
        "analysis_priority": "low"
    },
    "privacy_habits": {
        "name_ko": "프라이버시 습관",
        "name_en": "Privacy Habits",
        "type": "text",
        "description": "프라이버시 보호 습관",
        "analysis_priority": "medium"
    },
    "preferred_chocolate_situation": {
        "name_ko": "선호하는 초콜릿 상황",
        "name_en": "Preferred Chocolate Situation",
        "type": "text",
        "description": "초콜릿을 선호하는 상황",
        "analysis_priority": "low"
    },
    "waste_disposal_method": {
        "name_ko": "쓰레기 처리 방법",
        "name_en": "Waste Disposal Method",
        "type": "text",
        "description": "쓰레기 처리 방법",
        "analysis_priority": "low"
    },
    "morning_wakeup_method": {
        "name_ko": "아침 기상 방법",
        "name_en": "Morning Wakeup Method",
        "type": "text",
        "description": "아침에 일어나는 방법",
        "analysis_priority": "low"
    },
    "late_night_snack_method": {
        "name_ko": "야식 습관",
        "name_en": "Late Night Snack Method",
        "type": "text",
        "description": "야식 섭취 습관",
        "analysis_priority": "low"
    },
    "reducing_plastic_bags": {
        "name_ko": "비닐봉지 줄이기",
        "name_en": "Reducing Plastic Bags",
        "type": "text",
        "description": "비닐봉지 사용을 줄이는 방법",
        "analysis_priority": "medium"
    },
    "solo_dining_frequency": {
        "name_ko": "혼밥 빈도",
        "name_en": "Solo Dining Frequency",
        "type": "text",
        "description": "혼자 식사하는 빈도",
        "analysis_priority": "low"
    },
    "summer_fashion_essential": {
        "name_ko": "여름 패션 필수품",
        "name_en": "Summer Fashion Essential",
        "type": "text",
        "description": "여름에 필수적인 패션 아이템",
        "analysis_priority": "low"
    },
    "pets": {
        "name_ko": "반려동물",
        "name_en": "Pets",
        "type": "text",
        "description": "반려동물 보유 여부 및 종류",
        "analysis_priority": "low"
    },
    "lifestyle_values": {
        "name_ko": "라이프스타일 가치관",
        "name_en": "Lifestyle Values",
        "type": "text",
        "description": "라이프스타일 가치관 및 철학",
        "analysis_priority": "high"
    }
}


def add_unstructured_metadata(session):
    """비정형 필드들을 column_metadata에 추가"""
    print("📝 비정형 필드 메타데이터 추가 중...")
    
    for column_name, metadata in UNSTRUCTURED_FIELDS_METADATA.items():
        session.execute(
            text("""
                INSERT INTO column_metadata (
                    column_name, name_ko, name_en, type, description, 
                    analysis_priority, chart_types, statistics
                ) VALUES (
                    :column_name, :name_ko, :name_en, :type, :description,
                    :analysis_priority, :chart_types, :statistics
                )
                ON CONFLICT (column_name) DO UPDATE SET
                    name_ko = EXCLUDED.name_ko,
                    name_en = EXCLUDED.name_en,
                    type = EXCLUDED.type,
                    description = EXCLUDED.description,
                    analysis_priority = EXCLUDED.analysis_priority,
                    updated_at = NOW()
            """),
            {
                "column_name": column_name,
                "name_ko": metadata.get("name_ko"),
                "name_en": metadata.get("name_en"),
                "type": metadata.get("type", "text"),
                "description": metadata.get("description"),
                "analysis_priority": metadata.get("analysis_priority", "low"),
                "chart_types": [],
                "statistics": [],
            }
        )
    
    session.commit()
    print(f"  ✅ 총 {len(UNSTRUCTURED_FIELDS_METADATA)}개 비정형 필드 메타데이터 추가 완료")


def load_all_metadata(data_dir: Path):
    """모든 메타데이터를 DB에 적재"""
    session = SessionLocal()
    
    try:
        metadata_path = data_dir / "column_metadata.json"
        label_path = data_dir / "label.json"
        groups_path = data_dir / "category_groups.json"
        
        # 파일 존재 확인
        if not metadata_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {metadata_path}")
            return
        if not label_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {label_path}")
            return
        if not groups_path.exists():
            print(f"❌ 파일을 찾을 수 없습니다: {groups_path}")
            return
        
        # 1. 비정형 필드 메타데이터 추가 (먼저 실행)
        add_unstructured_metadata(session)
        
        # 2. 컬럼 메타데이터 적재
        load_column_metadata(session, metadata_path)
        
        # 3. 라벨 값 적재
        load_label_values(session, label_path)
        
        # 4. 카테고리 그룹 적재
        load_category_groups(session, groups_path)
        
        print("\n✅ 모든 메타데이터 적재 완료!")
        
    except Exception as e:
        session.rollback()
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


# ===== CLI 진입점 =====
def main():
    parser = argparse.ArgumentParser(
        description="통합 ETL 파이프라인: 메타데이터 적재 및 패널 데이터 적재",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 단계:
  1. metadata  - 메타데이터 적재 (비정형 필드, 컬럼 메타데이터, 라벨, 카테고리 그룹)
  2. panels    - 패널 데이터 적재 (전처리, 임베딩, 요약 생성)
  3. migration - 마이그레이션 (QA 형식 변환, 형태소 분석, 임베딩 재생성)
  4. all       - 모든 단계 자동 순차 실행 (기본값, 1→2 자동 진행, --auto-migration 옵션 시 1→2→3 자동 진행)

사용 예시:
  # 모든 단계 실행
  python backend/scripts/etl_pipeline.py --step all --input backend/data/panel_data.json
  
  # 메타데이터만 적재
  python backend/scripts/etl_pipeline.py --step metadata
  
  # 패널 데이터만 적재
  python backend/scripts/etl_pipeline.py --step panels --input backend/data/panel_data.json
        """
    )
    parser.add_argument(
        "--step", "-s",
        choices=["metadata", "panels", "migration", "all"],
        default="all",
        help="실행할 단계 선택 (기본: all, all 선택 시 자동으로 순차 진행)"
    )
    parser.add_argument(
        "--input", "-i",
        help="패널 데이터 JSON 파일 경로 (panels 단계에서 필수)"
    )
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="패널별 1-2줄 요약(LLM, Bedrock Haiku)을 생성하여 panels.panel_summary_text에 저장 (기본: 비활성화)"
    )
    parser.add_argument(
        "--auto-migration",
        action="store_true",
        help="all 단계 실행 시 마이그레이션도 자동 실행 (기본: 비활성화, QA 형식 변환, 형태소 분석, 임베딩 재생성)"
    )

    args = parser.parse_args()

    # 패널 데이터 적재 단계에서는 입력 파일 필수
    if args.step in ["panels", "all"] and not args.input:
        parser.error("--input 옵션이 필요합니다 (panels 또는 all 단계 실행 시)")

    data_dir = Path(PROJECT_ROOT) / "backend" / "data"

    try:
        # 메타데이터 적재 단계
        if args.step in ["metadata", "all"]:
            print("=" * 60)
            print("📋 단계 1: 메타데이터 적재")
            print("=" * 60)
            load_all_metadata(data_dir)
            print("✅ 단계 1 완료! 다음 단계로 진행합니다...\n")
            
            # all 모드인 경우 자동으로 다음 단계 진행
            if args.step == "all":
                args.step = "panels"  # 다음 단계로 설정

        # 패널 데이터 적재 단계
        if args.step in ["panels", "all"]:
            print("=" * 60)
            print("📊 단계 2: 패널 데이터 적재")
            print("=" * 60)
            
            # 실행 플래그를 환경 변수로도 전달(파이프라인 내부에서 참조)
            if args.generate_summaries:
                os.environ["ETL_ENABLE_SUMMARY"] = "true"
            
            json_path = Path(args.input)
            if not json_path.exists():
                print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
                sys.exit(1)

            load_json_to_db(str(json_path))
            print("✅ 단계 2 완료! 다음 단계로 진행합니다...\n")
            
            # all 모드이고 auto-migration 옵션이 있는 경우 자동으로 다음 단계 진행
            if args.step == "all" and args.auto_migration:
                args.step = "migration"  # 다음 단계로 설정
            elif args.step == "all":
                args.step = None  # 마이그레이션은 수동 실행

        # 마이그레이션 단계 (QA 형식, 형태소 분석, 임베딩 재생성)
        if args.step in ["migration", "all"] and args.auto_migration:
            print("=" * 60)
            print("🔄 단계 3: 마이그레이션 (QA 형식 변환, 형태소 분석, 임베딩 재생성)")
            print("=" * 60)
            
            # 마이그레이션 스크립트 자동 실행 (최적화된 배치 크기 사용)
            migration_scripts = [
                ("QA 형식 변환", "backend/scripts/migrate_qa_format.py", "--batch-size", "5000"),
                ("형태소 분석", "backend/scripts/migrate_tsvector_morphology.py", "--batch-size", "3000"),
                ("임베딩 재생성", "backend/scripts/regenerate_embeddings.py", "--batch-size", "2000"),
            ]
            
            for step_name, script_path, *args in migration_scripts:
                script_full_path = Path(PROJECT_ROOT) / script_path
                if not script_full_path.exists():
                    print(f"⚠️  스크립트를 찾을 수 없습니다: {script_path}")
                    continue
                
                print(f"\n📝 {step_name} 실행 중... (배치 크기: {args[1] if len(args) > 1 else '기본값'})")
                print(f"   스크립트: {script_path}")
                
                # 스크립트 실행 (비동기 함수 호출)
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, str(script_full_path)] + list(args),
                        cwd=PROJECT_ROOT,
                        capture_output=False,
                        text=True,
                        env={**os.environ, "PYTHONPATH": PROJECT_ROOT}
                    )
                    if result.returncode == 0:
                        print(f"✅ {step_name} 완료!")
                    else:
                        print(f"⚠️  {step_name} 실패 (반환 코드: {result.returncode})")
                except Exception as e:
                    print(f"⚠️  {step_name} 실행 중 오류: {e}")
            
            print("✅ 단계 3 완료!\n")

        print("=" * 60)
        print("✅ 모든 ETL 작업 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ETL 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
