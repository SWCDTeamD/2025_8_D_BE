"""
SWCD Panel Search Service

자연어 질의 기반 패널 검색 서비스
- Bedrock Claude Opus 4로 질의를 의미있는 단위로 분리 (환경 변수 SEARCH_LLM_MODEL_ID로 모델 변경 가능)
- label.json 기반 정형/비정형 데이터 분류 및 매핑
- 정형 데이터 검색 (label.json 기반 SQL) + 비정형 데이터 검색 (벡터 + FTS → RRF) → 교집합
"""

from __future__ import annotations

import json
import os
import base64
import re
import asyncio
import time as time_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel, Field

from backend.repositories.panel_repository import PanelRepository
from backend.repositories.document_repository import DocumentRepository
from backend.repositories.database import AsyncSessionLocal
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# 형태소 분석기 (kiwipiepy)
_KIWI_TAGGER = None
def get_morphology_tagger():
    """형태소 분석기 초기화 및 반환 (싱글톤)"""
    global _KIWI_TAGGER
    if _KIWI_TAGGER is None:
        try:
            from kiwipiepy import Kiwi  # type: ignore
            _KIWI_TAGGER = Kiwi()
            print("✅ Kiwi 형태소 분석기 초기화 완료")
        except ImportError:
            print("⚠️ kiwipiepy가 설치되지 않았습니다. 형태소 분할 검색을 사용하려면 설치: pip install kiwipiepy")
            _KIWI_TAGGER = False  # 설치되지 않음을 표시
        except Exception as e:
            print(f"⚠️ 형태소 분석기 초기화 실패: {e}")
            _KIWI_TAGGER = False
    return _KIWI_TAGGER if _KIWI_TAGGER is not False else None

# LangChain + Bedrock (질의 분석용 LLM)
try:
    from langchain_aws import ChatBedrock  # type: ignore
    import boto3  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    _HAS_BEDROCK = True
except ImportError:
    ChatBedrock = None  # type: ignore
    boto3 = None  # type: ignore
    _HAS_BEDROCK = False

# .env 로드
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

# 세그먼트 메타데이터 로드 (캐싱)
_SEGMENT_METADATA: Optional[Dict[str, Any]] = None

def load_segment_metadata() -> Dict[str, Any]:
    """세그먼트 메타데이터 로드 (segment_metadata.json)"""
    global _SEGMENT_METADATA
    if _SEGMENT_METADATA is None:
        metadata_path = Path(__file__).parent.parent / "data" / "segment_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            _SEGMENT_METADATA = json.load(f)
    # 타입 체커를 위한 보장: _SEGMENT_METADATA가 None이면 빈 딕셔너리 반환
    return _SEGMENT_METADATA if _SEGMENT_METADATA is not None else {}


def split_query_by_morphology(query: str, max_chunks: int = 5) -> List[str]:
    """쿼리를 의미 단위로 분할하여 검색 청크 생성
    
    Args:
        query: 검색 쿼리 텍스트
        max_chunks: 최대 분할 개수 (너무 많이 나누면 성능 저하 및 정확도 하락)
    
    Returns:
        의미 단위로 분할된 청크 리스트 (예: ["반려동물 키우는", "OTT 1개 이상"])
    """
    if not query or not query.strip():
        return []
    
    # 쿼리가 너무 짧으면 분할하지 않음 (정확도 유지)
    if len(query.strip()) <= 5:
        return [query.strip()]
    
    # 공백 기준으로 먼저 분할 (의미 단위)
    words = [w.strip() for w in query.split() if w.strip()]
    
    if len(words) <= 1:
        return [query.strip()]
    
    # 의미 단위로 묶기: 완전한 의미를 유지
    # 예: "반려동물 키우는" → ["반려동물 키우는"] (완전한 의미)
    # 예: "OTT 1개 이상" → ["OTT 1개 이상"] (숫자 포함)
    # 예: "반려동물 키우는 OTT 1개 이상" → ["반려동물 키우는", "OTT 1개 이상"]
    
    kiwi_tagger = get_morphology_tagger()
    if not kiwi_tagger:
        # 형태소 분석기가 없으면 공백 기준으로 의미 단위 분할
        # 동사/형용사가 포함된 구문을 하나의 청크로 유지
        chunks = []
        i = 0
        while i < len(words):
            # 현재 단어가 동사/형용사 형태인지 확인 (간단한 휴리스틱)
            current_word = words[i].lower()
            is_verb_or_adj = any(suffix in current_word for suffix in ['는', '은', '을', '를', '이', '가', '한다', '한다', '하는', '된', '된'])
            
            if i + 1 < len(words):
                next_word = words[i + 1].lower()
                next_is_verb_or_adj = any(suffix in next_word for suffix in ['는', '은', '을', '를', '이', '가', '한다', '한다', '하는', '된', '된'])
                
                # 숫자가 포함된 경우 함께 묶기 (예: "1개", "2개 이상")
                has_number = any(char.isdigit() for char in words[i] + words[i + 1])
                
                if has_number or (is_verb_or_adj or next_is_verb_or_adj):
                    # 2개 단어를 함께 묶기 (의미 단위 유지)
                    chunks.append(f"{words[i]} {words[i+1]}")
                    i += 2
                else:
                    # 단독으로 추가
                    chunks.append(words[i])
                    i += 1
            else:
                # 마지막 단어
                chunks.append(words[i])
                i += 1
            
            if len(chunks) >= max_chunks:
                break
        
        return chunks if chunks else [query.strip()]
    
    # 조사 제거 함수 (핵심 키워드만 추출)
    def remove_particles(word: str) -> str:
        """조사 제거하여 핵심 키워드만 반환"""
        # 조사 목록
        particles = ['를', '을', '이', '가', '는', '은', '의', '에', '에서', '로', '으로', '와', '과', '도', '만', '부터', '까지']
        for particle in particles:
            if word.endswith(particle):
                cleaned = word[:-len(particle)]
                # 대문자로 시작하는 경우 (예: "OTT") 대문자 유지, 소문자는 대문자로 변환
                if cleaned.lower() == 'ott':
                    return 'OTT'
                return cleaned
        # 대소문자 통일 (OTT는 대문자로)
        if word.lower() == 'ott':
            return 'OTT'
        return word
    
    # Kiwi 형태소 분석 없이도 원본 단어를 유지하면서 의미 단위로 분할
    # 형태소 분석은 복잡하고 원본 단어를 잃을 수 있으므로, 간단한 휴리스틱 사용
    chunks = []
    i = 0
    
    # 비교 연산자 키워드 (숫자와 함께 묶어야 함)
    comparison_keywords = ['이상', '이하', '초과', '미만', '이내', '이외']
    
    while i < len(words):
        current_word = words[i]
        # 조사 제거하여 핵심 키워드 추출
        current_word_clean = remove_particles(current_word)
        
        # 숫자가 포함된 단어는 다음 단어들과 함께 묶기 (예: "1개 이상", "2개 이하")
        has_number = any(char.isdigit() for char in current_word)
        
        # 동사/형용사 어미가 있는 단어 확인 (예: "키우는", "이용하는")
        has_verb_suffix = any(suffix in current_word for suffix in ['는', '은', '한다', '하는', '된', '된', '하는'])
        
        if i + 1 < len(words):
            next_word = words[i + 1]
            next_word_clean = remove_particles(next_word)
            next_has_number = any(char.isdigit() for char in next_word)
            next_has_verb_suffix = any(suffix in next_word for suffix in ['는', '은', '한다', '하는', '된', '된', '하는'])
            next_is_comparison = next_word in comparison_keywords
            
            # 숫자가 포함된 경우: 앞 단어(명사) + 숫자 + 단위 + 비교연산자까지 함께 묶기
            # 예: "OTT 1개 이상", "반려동물 2마리"
            if has_number or next_has_number:
                # 숫자로 시작하는 청크 생성
                chunk_parts = []
                
                # 현재 단어가 숫자가 아니면 명사로 간주하고 포함 (예: "OTT", "반려동물")
                # 단, 이미 처리된 청크가 있으면 포함하지 않음 (중복 방지)
                if not has_number and (len(chunks) == 0 or not any(current_word_clean in c for c in chunks)):
                    chunk_parts.append(current_word_clean)
                    i += 1
                
                # 숫자와 단위 포함
                if i < len(words):
                    num_word = words[i]
                    chunk_parts.append(num_word)
                    i += 1
                    
                    # 다음 단어가 단위나 비교연산자면 포함
                    if i < len(words):
                        unit_word = words[i]
                        # 단위 키워드 또는 비교 연산자 확인
                        is_unit = unit_word in ['개', '명', '마리', '대', '개월', '년', '시간', '분', '초']
                        if unit_word in comparison_keywords or is_unit:
                            chunk_parts.append(unit_word)
                            i += 1
                            
                            # 비교 연산자가 있으면 추가로 묶기 (예: "1개 이상")
                            if i < len(words) and words[i] in comparison_keywords:
                                chunk_parts.append(words[i])
                                i += 1
                
                if chunk_parts:  # 빈 청크 방지
                    chunks.append(' '.join(chunk_parts))
            # 동사/형용사가 포함된 경우 함께 묶기 (의미 단위 유지, 조사 제거)
            elif has_verb_suffix or next_has_verb_suffix:
                # 조사 제거된 버전 사용
                chunks.append(f"{current_word_clean} {next_word_clean}")
                i += 2
            # 그 외에는 조사 제거된 버전 사용
            else:
                chunks.append(current_word_clean)
                i += 1
        else:
            # 마지막 단어 (조사 제거)
            chunks.append(current_word_clean)
            i += 1
        
        if len(chunks) >= max_chunks:
            break
    
    return chunks if chunks else [query.strip()]


# ===== Pydantic 스키마 =====
class NLQueryRequest(BaseModel):
    """자연어 검색 요청"""
    query: str = Field(..., description="자연어 질의 텍스트")
    top_k: int = Field(20, ge=1, le=10000, description="최대 반환 개수 (질의에서 개수가 명시되지 않으면 모든 결과 반환)")


class SearchResultItem(BaseModel):
    """검색 결과 아이템"""
    panel_id: str
    score: float
    source: str
    # 정확도 정보 (새로 추가)
    accuracy_score: Optional[float] = Field(None, description="종합 정확도 점수 (0.0 ~ 1.0)")
    vector_score: Optional[float] = Field(None, description="벡터 검색 유사도 점수 (0.0 ~ 1.0)")
    fts_score: Optional[float] = Field(None, description="FTS 검색 점수 (0.0 ~ 1.0)")
    rrf_score: Optional[float] = Field(None, description="RRF 통합 점수")
    matched_fields: Optional[List[str]] = Field(None, description="매칭된 정형 필드 목록 (예: ['gender', 'age', 'region_city'])")
    # 패널 기본 정보 (비정형 데이터 제외)
    gender: Optional[str] = None
    age: Optional[int] = None
    region_city: Optional[str] = None
    region_gu: Optional[str] = None
    marital_status: Optional[str] = None
    children_count: Optional[int] = None
    family_size: Optional[int] = None
    education_level: Optional[str] = None
    occupation: Optional[str] = None
    monthly_personal_income: Optional[int] = None
    monthly_household_income: Optional[int] = None
    phone_brand: Optional[str] = None
    phone_model: Optional[str] = None
    car_ownership: Optional[bool] = None
    car_manufacturer: Optional[str] = None
    car_model: Optional[str] = None
    # 배열 필드들
    owned_electronics: Optional[List[str]] = None
    smoking_experience: Optional[List[str]] = None
    smoking_brand: Optional[List[str]] = None
    e_cig_heated_brand: Optional[List[str]] = None
    e_cig_liquid_brand: Optional[List[str]] = None
    drinking_experience: Optional[List[str]] = None
    panel_summary_text: Optional[str] = None


class SearchResponse(BaseModel):
    """검색 응답"""
    results: List[SearchResultItem]
    analysis: Optional[Dict[str, Any]] = Field(None, description="질의 분석 정보")


# ===== label.json 로드 =====
_LABEL_DATA: Optional[Dict[str, Any]] = None

def load_label_data() -> Dict[str, Any]:
    """label.json 파일 로드"""
    global _LABEL_DATA
    if _LABEL_DATA is None:
        label_path = Path(__file__).resolve().parents[2] / "backend" / "data" / "label.json"
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                _LABEL_DATA = json.load(f)
        except Exception as e:
            print(f"⚠️ label.json 로드 실패: {e}")
            _LABEL_DATA = {}
    return _LABEL_DATA or {}


# ===== Bedrock 설정 =====
def get_bedrock_llm(model_id: Optional[str] = None):
    """Bedrock Claude LLM 초기화 (검색 질의 분석용)
    
    Args:
        model_id: 모델 ID (None이면 환경 변수 또는 기본값 사용)
            - 환경 변수 SEARCH_LLM_MODEL_ID가 있으면 우선 사용
            - 없으면 기본값: Claude 3.5 Sonnet (anthropic.claude-3-5-sonnet-20241022-v2:0)
    
    Returns:
        ChatBedrock 인스턴스 또는 None
    """
    # 환경 변수에서 모델 ID 확인 (우선순위 1)
    if model_id is None:
        model_id = os.getenv("SEARCH_LLM_MODEL_ID")
    
    # 기본값: Claude 3.5 Sonnet (우선순위 2)
    # Sonnet 3.5는 on-demand를 지원하므로 inference profile 불필요
    if model_id is None:
        # Claude 3.5 Sonnet 사용
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    if not _HAS_BEDROCK:
        return None
    
    bedrock_key_encoded = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_BEDROCK_API_KEY")
    if not bedrock_key_encoded:
        return None
    
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
        session = boto3.Session(  # type: ignore
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    else:
        session = boto3.Session(region_name=region)  # type: ignore
    
    # 분석 작업용 LLM은 더 긴 응답과 약간의 창의성이 필요하므로
    # max_tokens를 증가시키고 temperature를 약간 높임
    # 단, 검색 질의 분석은 일관성이 중요하므로 temperature는 0 유지
    is_analysis_model = "sonnet" in model_id.lower()
    model_kwargs = {
        "temperature": 0.3 if is_analysis_model else 0,  # 분석: 0.3 (일관성과 다양성 균형), 검색: 0 (일관성 우선)
        "max_tokens": 8000 if is_analysis_model else 2000  # 분석: 8000 (긴 응답 필요), 검색: 2000 (충분)
    }
    
    return ChatBedrock(  # type: ignore[call-arg]
        model_id=model_id,  # type: ignore[arg-type]
        credentials_profile_name=None,
        region_name=region,  # type: ignore[arg-type]
        client=session.client("bedrock-runtime"),
        model_kwargs=model_kwargs
    )


def create_query_analysis_chain(llm):
    """질의를 의미있는 단위로 분리하고 label.json 기반으로 정형/비정형 분류하는 Chain"""
    if not llm:
        return None

    label_data = load_label_data()
    
    # label.json의 카테고리 목록 추출
    categories = list(label_data.keys())
    category_examples = {}
    for category in categories:
        values = label_data.get(category, [])
        if isinstance(values, list) and len(values) > 0:
            # 처음 5개만 예시로 보여줌
            category_examples[category] = values[:5]
    
    # 주의: label.json의 모든 키는 이미 DB 컬럼명과 동일합니다
    # 예: label.json의 "age" = DB의 "age" 컬럼
    #     label.json의 "region_city" = DB의 "region_city" 컬럼
    # 따라서 별도의 매핑이 필요 없습니다
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """질의를 의미 단위로 분리하고, label.json 카테고리로 매핑하세요.

카테고리: {categories}
예시: {category_examples}

=== 1. 기본 원칙 ===
- 의미 단위 청크 분리: "서울에 사는"(O) / "서울", "에", "사는"(X)
- 정형/비정형 구분: 카테고리 매칭 시 정형, 주관적 서술은 비정형
- 개수 표현 제외: "100명 뽑아줘" 같은 개수 표현은 chunks에 포함하지 않음

=== 2. 의미 기반 매핑 (핵심 원칙) ===
**매칭 우선순위 (반드시 순서대로):**
1. **정확한 일치 최우선**: 사용자 입력이 label.json 값과 **정확히 일치**하면 그 값을 사용
   - 예: "기아" 입력 → label.json에 "기아" 있음 → "기아" 사용 (다른 값 고려 X)
2. **유사어 매핑**: 정확히 일치하지 않으면 아래 유사어 매핑 규칙 참고
3. **의미적 동등성**: 동의어나 다른 표현이지만 같은 의미인 경우만 매핑
4. **매핑 불가**: label.json에 없고 유사어도 없으면 비정형으로 처리

**중요 주의사항:**
- **"기아" ≠ "KG모빌리티"**: 완전히 다른 회사! "K"가 공통이어도 절대 혼동하지 말 것
- **"현대" ≠ "제네시스"**: 제네시스는 현대의 럭셔리 브랜드지만 별도 제조사로 구분
- **정확한 값이 있으면 유사도 무시**: label.json에 정확한 값이 있으면 다른 값 고려하지 말 것

=== 3. 값 증강 및 정규화 (Value Augmentation & Normalization) ===
**핵심 원칙:** 사용자의 표현을 **의미를 이해하여** label.json 목록의 표준 값으로 매핑하되, 범위나 계층 구조를 고려하여 **포함되는 모든 값으로 확장**해야 합니다.

**지역 증강:**
- 광역 도: "충청도" → ["충북", "충남"], "경상도" → ["경북", "경남"], "전라도" → ["전북", "전남"], "강원도" → ["강원"]
- 수도권: "수도권" → ["서울", "경기"]
- 지역: label.json의 "지역(시)" 또는 "지역(구)" 목록에 있는 가장 적절한 값으로 매핑

**나이 증강:**
- 연령대: "20대" → [20,21,22,23,24,25,26,27,28,29], "30대" → [30,31,32,33,34,35,36,37,38,39]
- 특정 나이: "24세" → [24]
- 이상/이하 표현:
  * "40대 이상" → [40,41,42,...,102] (label.json의 age 목록에서 40 이상 모든 값)
  * "30대 이하" → [15,16,17,...,39] (label.json의 age 목록에서 39 이하 모든 값)
  * "20대~30대" → [20,21,...,39] (범위 내 모든 값)

**학력 증강 (매우 중요!):**
학력 계층 구조 (낮음 → 높음):
1. "고등학교 졸업 이하"
2. "대학교 재학/휴학"
3. "대학교 졸업"
4. "대학원 재학/졸업 이상"

**"이상" 표현 처리 규칙:**
- "고졸 이상", "고등학교 졸업 이상" → ["대학교 재학/휴학", "대학교 졸업", "대학원 재학/졸업 이상"]
  * 고등학교 졸업 이하를 제외한 모든 학력 (고등학교 졸업자 이상)
- "대학 이상", "대학교 이상" → ["대학교 재학/휴학", "대학교 졸업", "대학원 재학/졸업 이상"]
  * "대학교"가 들어간 모든 학력 + 그보다 높은 학력
- "대학교 졸업 이상", "대졸 이상" → ["대학교 졸업", "대학원 재학/졸업 이상"]
  * "대학교 졸업" 본인 + 그보다 높은 학력
- "대학원 이상" → ["대학원 재학/졸업 이상"]
  * "대학원" 관련 학력만
  
**"이하" 표현 처리 규칙:**
- "고졸 이하", "고등학교 졸업 이하" → ["고등학교 졸업 이하"]
- "대학 재학 이하", "대학교 재학 이하" → ["고등학교 졸업 이하", "대학교 재학/휴학"]
- "대학 졸업 이하", "대학교 졸업 이하", "대졸 이하" → ["고등학교 졸업 이하", "대학교 재학/휴학", "대학교 졸업"]

**정확한 매칭 (범위 표현 없음):**
- "고졸", "고등학교 졸업" → ["고등학교 졸업 이하"] (정확한 학력만)
- "대학생", "대학교 재학" → ["대학교 재학/휴학"] (정확한 학력만)
- "대졸", "대학교 졸업" → ["대학교 졸업"] (정확한 학력만)
- "대학원생", "대학원" → ["대학원 재학/졸업 이상"] (정확한 학력만)

**직업 증강:**
- "전문직" → label.json의 occupation 목록에서 "전문직"이 포함된 모든 값
  * 예: ["간호·조산 전문직", "간호·조산 준 전문직", "감독 행정 준 전문직", "공학 전문직", "경영 전문직", "보건 전문직", "물리·지구 과학 전문직", "생명 과학 전문직", "사회·종교 전문직", "재무 전문직", "정보 통신 기술 전문직", "판매·마케팅·홍보 전문직", "기타 교육 전문직"]
- "준 전문직" → "준 전문직"이 포함된 모든 값
- "관리자" → "관리자"가 포함된 모든 값
- "기사" → "기사"가 포함된 모든 값
- 직업 카테고리 키워드가 포함된 모든 occupation 값을 배열로 반환

**일반 명사 변환:**
- "남자" → "남성", "아이폰" → "애플", "서울시" → "서울", "기혼자" → "기혼"
- "삼성폰" → "삼성전자", "LG폰" → "LG전자"

**오타/유사 표현 매핑 (중요!):**
사용자가 오타나 유사한 표현을 입력하면, label.json의 정확한 값으로 매핑해야 합니다.

**차량 제조사 (중요!):**
- "기아" → "기아" (KIA Motors)
- "kia" → "기아"
- "KIA" → "기아"
- "현대" → "현대" (Hyundai)
- "hyundai" → "현대"
- "쌍용" → "KG모빌리티" (구 쌍용자동차)
- "쌍용자동차" → "KG모빌리티"
- "KG모빌리티" → "KG모빌리티"
- "제네시스" → "제네시스" (Genesis)
- "genesis" → "제네시스"
- "삼성" → "르노삼성"
- "르노" → "르노삼성"
- "쉐보레" → "쉐보레" (Chevrolet)
- "쉐비" → "쉐보레"
- "벤츠" → "메르세데스-벤츠"
- "메르세데스" → "메르세데스-벤츠"
- "bmw" → "BMW"
- "BMW" → "BMW"
- "테슬라" → "테슬라" (Tesla)
- "tesla" → "테슬라"

**차량 모델:**
- "소나타" → "쏘나타"
- "아반테" → "아반떼"
- "산타페" → "싼타페"
- "쏘렌토" → "쏘렌토"
- "싼타페" → "싼타페"
- "카니발" → "카니발"
- "투산" → "투싼"
- "코나" → "코나"

**담배 브랜드:**
- "말보루" → "말보로"
- "에쎄" → "에쎄"
- "레죵" → "레종"
- "디스" → "디스"
- "보헴" → "보헴"
- "던힐" → "던힐"

**휴대폰 브랜드/모델:**
- "아이폰" → "애플"
- "갤럭시" → "삼성전자"
- "LG폰" → "LG전자"

**국산/외제 구분 (매우 중요!):**
사용자가 "국산/외제"라는 표현을 사용하면, 해당 카테고리의 **모든 국산/외제 브랜드를 배열로 반환**해야 합니다.

**국산 핸드폰 브랜드:**
- "국산 핸드폰", "국산폰", "국산 스마트폰" → ["삼성전자", "LG전자"]
- 국산 브랜드: 삼성전자, LG전자

**외제 핸드폰 브랜드:**
- "외제 핸드폰", "외산 핸드폰", "수입 핸드폰" → ["애플", "샤오미"]
- 외제 브랜드: 애플(미국), 샤오미(중국)

**국산 차량 제조사:**
- "국산차", "국산 자동차", "국내 차량" → ["현대", "기아", "제네시스", "KG모빌리티", "르노삼성", "쉐보레"]
- 국산 브랜드: 현대, 기아, 제네시스, KG모빌리티(구 쌍용), 르노삼성, 쉐보레(한국GM)

**외제 차량 제조사:**
- "외제차", "수입차", "외산차" → label.json의 car_manufacturer 중 국산 6개를 제외한 모든 브랜드
- 외제 브랜드 예시: BMW, 메르세데스-벤츠, 아우디, 테슬라, 토요타, 렉서스, 혼다, 닛산, 볼보, 포르쉐, 폭스바겐, 재규어, 랜드로버, 링컨, 캐딜락, 지프, 크라이슬러, 인피니티 등

**일반 원칙:**
- 오타가 의심되면 label.json 값과 **발음이나 철자가 유사한 값**으로 자동 매핑
- 예: "ㅆ"과 "ㅅ", "ㄹ"과 "ㄷ" 같은 받침 차이는 같은 것으로 처리
- 동의어/유사 표현은 **의미를 이해하여** 정규화된 표준 값으로 변환

=== 4. 부정/긍정 표현 처리 ===
**흡연 (smoking_experience 카테고리):**
- **부정** ("담배 안 피는", "흡연 안 함", "비흡연", "비흡연자" 등):
  * mapped_values: ["담배를 피워본 적이 없다"]
  
- **긍정** ("담배 피는", "흡연하는", "흡연자", "흡연 경험이 있는" 등):
  * mapped_values: 다음 4가지 값을 **정확히 이 순서대로, 모두** 포함:
    1. "일반 담배"
    2. "궐련형 전자 담배/ 가열식 담배"  ← 슬래시(/) 앞뒤 띄어쓰기 정확히!
    3. "액상형 전자담배"  ← 띄어쓰기 없음!
    4. "기타 담배"
  
  **⚠️ 매우 중요:**
  - 위 4가지 값을 **하나도 빠뜨리지 말고 모두** 배열에 넣어야 합니다!
  - 특히 "궐련형 전자 담배/ 가열식 담배"를 빼먹지 마세요!
  - 띄어쓰기와 특수문자(/)를 정확히 지켜야 합니다!

**음주 (drinking_experience 카테고리):**
- **부정** ("술 안 마시는", "음주 안 함", "비음주", "금주" 등):
  * mapped_values: ["최근 1년 이내 술을 마시지 않음"]
  
- **긍정** ("술 마시는", "음주하는", "음주자" 등):
  * mapped_values: 다음 값들을 **모두** 포함 (부정 값 제외):
    "과일칵테일주", "기타", "막걸리/탁주", "맥주", "소주", "양주", "와인", "일본청주/사케", "저도주", "하이볼"
  
  **⚠️ 중요:** "술 마시는" 같은 일반적인 음주 표현이 나오면, "최근 1년 이내 술을 마시지 않음"을 제외한 **모든 술 종류를 배열에 포함**해야 합니다!

**차량:**
- 부정 ("차 없음" 등) → ["없다"] 또는 false (car_ownership)

**⚠️ 중요: 반려동물은 label.json에 없으므로 비정형으로 처리:**
- "반려동물을 키우는", "반려동물 키우는", "펫을 키우는" 등의 표현은 **비정형 청크**로 분류
- related_segments: ["PETS"]
- keywords: ["반려동물", "키우는", "펫"]
- search_hints의 exclude_patterns: ["키우지 않는다", "키워본 적이 없다", "없다", "없음"]

=== 5. 메타데이터 생성 ===
**비정형 청크 (필수):**
- `intent`: "positive" (구체적 객체/행동), "negative" (부정 표현), "neutral" (그 외)
- `keywords`: 핵심 키워드 배열 (3-5개 권장)
- `negation_detected`: 부정 표현 감지 여부 (boolean)
- `related_segments`: 이 청크와 관련된 설문 세그먼트 이름 배열 (아래 목록에서 선택, 최대 3개 권장)
- `search_hints`: 
  - `exclude_patterns`: 긍정 의도인 경우 부정 표현 패턴 나열 (중요!), 부정 의도인 경우 빈 배열
    * 예: ["없다", "없음", "하지 않는다", "하지 않음", "받지 않는다", "이용하지 않는다", "사용하지 않는다", "선호하지 않는다", "안 한다", "안한다"]
    * 긍정 의도 ("키우는", "이용하는", "쓰는")인 경우 반드시 부정 패턴 포함!
  - `include_patterns`: 핵심 키워드나 구체적 객체

**사용 가능한 세그먼트 목록 (related_segments에서 선택):**
{available_segments}

**related_segments 선택 규칙:**
- 쿼리 내용과 직접적으로 관련된 세그먼트만 선택 (위 목록에서 segment_name 사용)
- 각 세그먼트의 키워드를 참고하여 관련성 판단
- 최대 3개까지 선택 (가장 관련성 높은 것 우선)
- 관련 세그먼트가 없으면 빈 배열 []
- 예시:
  * "이사할 때 비용문제로 스트레스를 받는" → ["MOVING_STRESS_FACTORS", "STRESS_FACTORS"]
  * "반려동물을 키우는" → ["PETS"]
  * "스킨케어 제품에 5만원 이상 쓰는" → ["SKINCARE_SPENDING_MONTHLY"]
  * "운동을 좋아하는" → ["FITNESS_MANAGEMENT_METHOD"]
  * "AI를 자주 사용하는" → ["AI_USAGE_FIELD", "CHATBOT_EXPERIENCE"]

**정형 청크 (선택적, 권장):**
- `confidence`: "high" (정확), "medium" (유사), "low" (불확실)
- `alternative_values`: 대체 가능한 값들 (유사 매칭 시)
- `fuzzy_match`: 유사 매칭 여부 (boolean)
- `intent`: "positive", "negative", "neutral"
- `search_hints`: 검색 힌트 객체 (선택적)

=== 6. DB 컬럼명 매핑 (중요!) ===
**주의: label.json의 모든 키는 이미 DB 컬럼명과 동일합니다!**

- `category`: label.json의 키 (예: "age", "region_city", "marital_status" 등)
- `db_column`: `category`와 동일한 값 사용 (별도 변환 불필요)

**특수 케이스:**
- "job" → "occupation" (유일한 예외, "job"은 label.json에 없고 "occupation"만 있음)

**예시:**
- label.json의 "age" → `category: "age"`, `db_column: "age"`
- label.json의 "region_city" → `category: "region_city"`, `db_column: "region_city"`
- label.json의 "marital_status" → `category: "marital_status"`, `db_column: "marital_status"`

=== 7. 출력 형식 ===
{{
    "chunks": [
        {{
            "text": "서울에 사는",
            "category": "region_city",
            "db_column": "region_city",
            "mapped_values": ["서울"],
            "is_structured": true,
            "confidence": "high",
            "fuzzy_match": false,
            "intent": "neutral"
        }},
        {{
            "text": "운동을 좋아하는",
            "category": null,
            "db_column": null,
            "mapped_values": [],
            "is_structured": false,
            "intent": "positive",
            "keywords": ["운동", "좋아"],
            "negation_detected": false,
            "related_segments": ["FITNESS_MANAGEMENT_METHOD"],
            "search_hints": {{
                "exclude_patterns": ["싫어", "안 좋아", "하지 않는다", "하지 않음", "없다", "없음"],
                "include_patterns": ["운동", "좋아"]
            }}
        }},
        {{
            "text": "OTT 서비스를 1개이상 이용하는",
            "category": null,
            "db_column": null,
            "mapped_values": [],
            "is_structured": false,
            "intent": "positive",
            "keywords": ["OTT", "서비스", "이용"],
            "negation_detected": false,
            "related_segments": ["OTT_SERVICE_COUNT"],
            "search_hints": {{
                "exclude_patterns": ["이용하지 않는다", "사용하지 않는다", "없다", "없음"],
                "include_patterns": ["OTT", "이용", "1개", "2개", "3개", "4개"]
            }}
        }},
        {{
            "text": "흡연자",
            "category": "smoking_experience",
            "db_column": "smoking_experience",
            "mapped_values": ["일반 담배", "궐련형 전자 담배/ 가열식 담배", "액상형 전자담배", "기타 담배"],
            "is_structured": true,
            "confidence": "high",
            "fuzzy_match": false,
            "intent": "positive",
            "search_hints": {{
                "exclude_patterns": ["피워본 적이 없다", "비흡연", "안 피"],
                "include_patterns": ["흡연", "담배", "피는"]
            }}
        }},
        {{
            "text": "반려동물을 키우는",
            "category": null,
            "db_column": null,
            "mapped_values": [],
            "is_structured": false,
            "intent": "positive",
            "keywords": ["반려동물", "키우는", "펫"],
            "negation_detected": false,
            "related_segments": ["PETS"],
            "search_hints": {{
                "exclude_patterns": ["키우지 않는다", "키워본 적이 없다", "없다", "없음"],
                "include_patterns": ["반려동물", "키우는", "펫", "강아지", "고양이"]
            }}
        }}
    ]
}}
"""),
        ("human", "질의: {query}")
    ])
    
    # 카테고리와 예시를 간결하게 포맷팅
    categories_str = ", ".join(categories)
    
    # 카테고리별 예시 간소화
    examples_lines = []
    for cat, examples in category_examples.items():
        if isinstance(examples, list) and len(examples) > 0:
            if isinstance(examples[0], int):
                examples_str = f"{examples[0]}~{examples[-1]}"
            else:
                examples_str = ", ".join([str(e) for e in examples[:3]])  # 3개로 줄임
            examples_lines.append(f"{cat}: {examples_str}")
    
    examples_str = "\n".join(examples_lines) if examples_lines else ""
    
    # 세그먼트 메타데이터에서 카테고리별 세그먼트 목록 생성
    try:
        segment_metadata = load_segment_metadata()
        segments_by_category: Dict[str, List[str]] = {}
        
        for segment in segment_metadata.get("segments", []):
            category = segment.get("category", "기타")
            segment_name = segment.get("segment_name", "")
            description = segment.get("description", "")
            keywords = segment.get("keywords", [])
            
            if category not in segments_by_category:
                segments_by_category[category] = []
            
            # 세그먼트 이름과 설명, 키워드를 함께 출력
            segment_info = f"{segment_name} ({description})"
            if keywords:
                keyword_str = ", ".join(keywords[:3])  # 키워드 3개만
                segment_info += f" [키워드: {keyword_str}]"
            
            segments_by_category[category].append(segment_info)
        
        # 카테고리별로 포맷팅
        available_segments_lines = []
        for category, segments in segments_by_category.items():
            segments_str = "\n  - ".join(segments)
            available_segments_lines.append(f"{category}:\n  - {segments_str}")
        
        available_segments = "\n".join(available_segments_lines)
    except Exception as e:
        print(f"⚠️ 세그먼트 메타데이터 로드 실패: {e}")
        # 실패 시 기본값 (하드코딩)
        available_segments = """건강/뷰티: FITNESS_MANAGEMENT_METHOD, SKIN_SATISFACTION, SKINCARE_SPENDING_MONTHLY, SKINCARE_CONSIDERATIONS, MOST_EFFECTIVE_DIET_EXPERIENCE
기술/디지털: AI_USAGE_FIELD, MOST_SAVED_PHOTOS_TOPIC, OTT_SERVICE_COUNT, MAIN_APPS_USED, CHATBOT_EXPERIENCE, MAIN_CHATBOT_USED, CHATBOT_MAIN_PURPOSE, PREFERRED_CHATBOT
금융/소비: PREFERRED_NEW_YEAR_GIFT, MAIN_QUICK_DELIVERY_PRODUCTS, REWARD_POINTS_INTEREST, PREFERRED_SPENDING_CATEGORY, HIGH_SPENDING_CATEGORY
여행/문화: PREFERRED_WATER_PLAY_AREA, TRAVEL_STYLE, TRADITIONAL_MARKET_VISIT_FREQUENCY, PREFERRED_OVERSEAS_DESTINATION, MEMORABLE_CHILDHOOD_WINTER_ACTIVITY, PREFERRED_SUMMER_SNACK
심리/가치관: STRESS_FACTORS, STRESS_RELIEF_METHOD, MOVING_STRESS_FACTORS, RAINY_DAY_COPING_METHOD, LIFESTYLE_VALUES, PRIVACY_HABITS, PREFERRED_CHOCOLATE_SITUATION, CONDITIONS_FOR_HAPPY_OLD_AGE
일상습관: WASTE_DISPOSAL_METHOD, MORNING_WAKEUP_METHOD, LATE_NIGHT_SNACK_METHOD, REDUCING_PLASTIC_BAGS, SOLO_DINING_FREQUENCY, SUMMER_FASHION_ESSENTIAL, PETS, SUMMER_WORRIES, SUMMER_SWEAT_DISCOMFORT"""
    
    # 프롬프트에 실제 데이터 주입
    formatted_prompt = prompt.partial(
        categories=categories_str, 
        category_examples=examples_str,
        available_segments=available_segments
    )
    
    return formatted_prompt | llm | StrOutputParser()


# ===== 유틸리티 함수 =====
def extract_count_from_query(query: str) -> Optional[int]:
    """질의에서 개수 추출 (예: "5명 뽑아줘", "10개 추출해줘", "20명 추천", "100명 뽑아줘")
    
    Returns:
        추출된 개수, 없으면 None
    """
    import re
    
    # 패턴: 더 구체적인 패턴부터 먼저 매칭 (순서 중요!)
    patterns = [
        # "뽑아줘", "뽑아", "뽑아달라" 등과 함께 있는 경우 우선
        (r'(\d+)\s*명\s*(?:뽑|추출|추천|보여|보여줘)', lambda m: int(m.group(1))),
        (r'(\d+)\s*개\s*(?:뽑|추출|추천|보여|보여줘)', lambda m: int(m.group(1))),
        # "N명 뽑아줘", "N개 뽑아줘" 패턴 (띄어쓰기 없이도 매칭)
        (r'(\d+)\s*명\s*뽑', lambda m: int(m.group(1))),
        (r'(\d+)\s*개\s*뽑', lambda m: int(m.group(1))),
        # "N명", "N개" 패턴 (단독으로 있는 경우)
        (r'(\d+)\s*명(?:\s|$|,|\.|뿐)', lambda m: int(m.group(1))),
        (r'(\d+)\s*개(?:\s|$|,|\.|뿐)', lambda m: int(m.group(1))),
        # "N명 정도", "N개 정도"
        (r'(\d+)\s*명\s*정도', lambda m: int(m.group(1))),
        (r'(\d+)\s*개\s*정도', lambda m: int(m.group(1))),
        # "N명만", "N개만"
        (r'(\d+)\s*명\s*만', lambda m: int(m.group(1))),
        (r'(\d+)\s*개\s*만', lambda m: int(m.group(1))),
    ]
    
    for pattern, extractor in patterns:
        match = re.search(pattern, query)
        if match:
            count = extractor(match)
            if 1 <= count <= 100000:  # 범위 확대 (최대 10만개)
                return count
    
    return None


def _rrf_score(rank: int, k: int = 60) -> float:
    """RRF(Reciprocal Rank Fusion) 점수 계산"""
    return 1.0 / (k + rank)


def _get_segment_reliability_score(segment_name: str) -> float:
    """세그먼트별 신뢰도 점수 (segment_metadata.json에서 로드)
    
    특정 세그먼트는 더 명확하고 신뢰도가 높음 (예: MOVING_STRESS_FACTORS - 이사 스트레스 요인)
    일반적인 세그먼트는 애매할 수 있음 (예: LIFESTYLE_VALUES - 라이프스타일 가치관)
    """
    try:
        metadata = load_segment_metadata()
        for segment in metadata.get("segments", []):
            if segment.get("segment_name") == segment_name:
                reliability = segment.get("reliability", "medium")
                if reliability == "high":
                    return 1.2
                elif reliability == "low":
                    return 0.8
                else:
                    return 1.0
    except Exception as e:
        print(f"⚠️ 세그먼트 메타데이터 로드 실패: {e}")
    
    # 기본값 (메타데이터 로드 실패 시)
    return 1.0


def _fuse_scores_by_rrf(list_a: List[str], list_b: List[str], k: int = 60, weight_a: float = 0.6, weight_b: float = 0.4) -> Dict[str, float]:
    """두 랭킹을 RRF로 통합
    
    Args:
        list_a: 벡터 검색 결과 (의미 기반)
        list_b: FTS 검색 결과 (키워드 기반)
        k: RRF k 파라미터
        weight_a: 벡터 검색 가중치 (기본 0.6) - 의미 기반 검색 우선
        weight_b: FTS 검색 가중치 (기본 0.4) - 키워드 매칭 보조
    """
    fused: Dict[str, float] = {}
    
    # 벡터 검색 (의미 기반): 가중치 60% - 의미 기반 검색 우선
    for rank, pid in enumerate(list_a, start=1):
        fused[pid] = fused.get(pid, 0.0) + (_rrf_score(rank, k) * weight_a)

    # FTS 검색 (키워드 기반): 가중치 40% - 키워드 매칭 보조
    for rank, pid in enumerate(list_b, start=1):
        fused[pid] = fused.get(pid, 0.0) + (_rrf_score(rank, k) * weight_b)

    return fused


def parse_analysis_result(result: str) -> Dict[str, Any]:
    """LLM 분석 결과 파싱 및 정형/비정형 분리"""
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = result[start:end]
            parsed = json.loads(json_str)
            
            # chunks에서 정형/비정형 분리
            structured_chunks = []
            unstructured_chunks = []
            
            for chunk in parsed.get("chunks", []):
                if chunk.get("is_structured", False):
                    # 정형 청크: 메타데이터 포함하여 저장
                    structured_chunk_data = {
                        "text": chunk.get("text", ""),
                        "category": chunk.get("category"),
                        "db_column": chunk.get("db_column"),
                        "mapped_values": chunk.get("mapped_values", []),
                        "is_structured": True,
                        # 메타데이터 (선택적)
                        "confidence": chunk.get("confidence", "high"),  # 기본값: high
                        "alternative_values": chunk.get("alternative_values", []),
                        "fuzzy_match": chunk.get("fuzzy_match", False),
                        "intent": chunk.get("intent", "neutral"),
                        "search_hints": chunk.get("search_hints", {
                            "exclude_patterns": [],
                            "include_patterns": []
                        })
                    }
                    structured_chunks.append(structured_chunk_data)
                else:
                    # 비정형 청크: 텍스트와 메타데이터 모두 저장
                    unstructured_chunk_data = {
                        "text": chunk.get("text", ""),
                        "intent": chunk.get("intent", "neutral"),  # "positive", "negative", "neutral"
                        "keywords": chunk.get("keywords", []),
                        "negation_detected": chunk.get("negation_detected", False),
                        "related_segments": chunk.get("related_segments", []),  # 세그먼트 정보 추가
                        "search_hints": chunk.get("search_hints", {
                            "exclude_patterns": [],
                            "include_patterns": []
                        })
                    }
                    unstructured_chunks.append(unstructured_chunk_data)
            
            return {
                "chunks": parsed.get("chunks", []),
                "structured_chunks": structured_chunks,
                "unstructured_chunks": unstructured_chunks,
                "reason": "정형/비정형 분류 완료"
            }
    except Exception as e:
        print(f"분석 결과 파싱 실패: {e}")
    
    return {"chunks": [], "structured_chunks": [], "unstructured_chunks": [], "reason": "파싱 실패"}


def map_chunks_to_label_filters(structured_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """정형 데이터 청크를 label.json 기반 필터로 변환
    
    Args:
        structured_chunks: LLM이 추출한 정형 데이터 청크 리스트
            예: [
                {"text": "서울에 사는", "category": "region_city", "db_column": "region_city", "mapped_values": ["서울"], "is_structured": true},
                {"text": "직업이 의사인", "category": "job", "db_column": "occupation", "mapped_values": ["의사"], "is_structured": true}
            ]
    
    Returns:
        DB 컬럼명 기반 필터 리스트 (Repository에서 직접 사용)
            예: [
                {"category": "occupation", "mapped_values": ["의사"]},  # db_column을 category로 사용
                {"category": "region_city", "mapped_values": ["서울"]}
            ]
    """
    from backend.repositories.panel_repository import LABEL_TO_DB_COLUMN
    
    label_filters: List[Dict[str, Any]] = []
    label_data = load_label_data()
    
    # 한글 카테고리 → 영어 카테고리 매핑 (하위 호환성)
    HANGUL_TO_ENGLISH_CATEGORY = {
        "성별": "gender",
        "나이": "age",
        "지역": "region_city",
        "지역(시)": "region_city",
        "지역(구)": "region_district",
        "결혼 여부": "marital_status",
        "결혼유무": "marital_status",
        "자녀수": "children_count",
        "가족수": "family_size",
        "최종학력": "education_level",
        "직업": "occupation",
        "월평균개인소득": "income_personal_monthly",
        "월평균가구소득": "income_household_monthly",
        "보유 휴대폰 브랜드": "phone_brand",
        "보유 휴대폰 모델명": "phone_model",
        "차량 보유 여부": "car_ownership",
        "보유 차량 제조사": "car_manufacturer",
        "보유 차량 모델": "car_model",
        "보유 전자 제품": "electronics_owned_multi",
        "흡연경험": "smoking_experience",
        "흡연경험브랜드": "smoking_brand",
        "음주 경험": "drinking_experience",
    }
    
    for chunk in structured_chunks:
        category = chunk.get("category")
        db_column = chunk.get("db_column")  # LLM이 출력한 DB 컬럼명
        mapped_values = chunk.get("mapped_values", [])
        
        if not mapped_values:
            continue
        
        # 우선순위 1: LLM이 출력한 db_column 사용 (가장 신뢰할 수 있음)
        if db_column:
            # db_column을 category로 사용 (Repository에서 db_column을 직접 사용)
            label_filters.append({
                "category": db_column,  # DB 컬럼명을 category로 사용
                "mapped_values": mapped_values
            })
            continue
        
        # 우선순위 2: category가 있으면 변환 시도 (하위 호환성)
        if not category:
            continue
        
        # 한글 카테고리를 영어 카테고리로 변환
        # label.json의 키는 이미 DB 컬럼명과 동일
        # 우선순위: label.json 키 → 한글 카테고리 변환 → 원본 유지
        english_category = None
        if category in label_data:
            # 이미 영어 카테고리 (label.json의 키 = DB 컬럼명)
            english_category = category
        elif category in HANGUL_TO_ENGLISH_CATEGORY:
            # 한글 카테고리 → 영어 카테고리 변환
            english_category = HANGUL_TO_ENGLISH_CATEGORY[category]
        elif category in LABEL_TO_DB_COLUMN:
            # LABEL_TO_DB_COLUMN에 있는 경우 (한글 카테고리)
            english_category = LABEL_TO_DB_COLUMN[category]
        else:
            # 매칭 실패: 원본 카테고리 유지 (나중에 필터링됨)
            english_category = category
        
        # label.json에 있는 카테고리만 필터로 추가
        if english_category:
            label_filters.append({
                "category": english_category,  # DB 컬럼명으로 저장
                "mapped_values": mapped_values
            })
    
    return label_filters


# ===== 서비스 핵심 로직 =====
async def natural_language_panel_search(
    payload: NLQueryRequest,
    session: Optional[AsyncSession] = None  # [개선] 세션을 인자로 받을 수 있도록 수정
) -> SearchResponse:
    """자연어 질의 기반 패널 검색
    
    프로세스:
    1. LLM으로 질의 분석 (정형/비정형 분류)
    2. 정형 데이터가 있으면 정형 검색 수행 (label.json 기반 SQL)
    3. 비정형 데이터가 있으면 비정형 검색 수행 (벡터 + FTS → RRF)
    4. 정형 검색 결과와 비정형 검색 결과의 교집합으로 최종 패널 출력
    
    ⚠️ 중요: DB 연결 풀 고갈 방지를 위해 하나의 세션을 모든 검색 작업에서 공유합니다.
    
    Args:
        payload: 검색 요청 (자연어 질의)
        session: DB 세션 (없으면 새로 생성, 있으면 사용)
    """
    query = (payload.query or "").strip()
    if not query:
        return SearchResponse(results=[], analysis=None)
    
    # 세션 관리: 전달받은 세션이 있으면 사용, 없으면 새로 생성
    close_session = False
    if session is None:
        session = AsyncSessionLocal()
        close_session = True
    
    try:
        # 🔑 핵심 수정: 하나의 세션을 열어서 모든 검색 작업에서 공유
        # 이렇게 하면 asyncio.gather로 100개의 검색을 해도 DB 연결은 1개만 사용됩니다.
        return await _natural_language_panel_search_with_session(payload, query, session)
    finally:
        # 세션을 여기서 생성했다면 닫아야 함
        if close_session:
            await session.close()


async def _natural_language_panel_search_with_session(
    payload: NLQueryRequest, 
    query: str, 
    session: AsyncSession
) -> SearchResponse:
    """세션을 받아서 검색 수행 (내부 함수)"""
    
    # 전체 패널 조회 감지 ("모든 패널", "전체 패널 추출해줘", "모두 보여줘" 등)
    # 핵심: "모든/전체/모두/전부" + "패널/사람/응답자" 키워드가 있고, 다른 조건이 없는 경우
    query_lower = query.lower().strip()
    
    # 전체 조회를 의미하는 키워드
    all_keywords = ["모든", "전체", "모두", "전부", "all", "every", "entire"]
    target_keywords = ["패널", "사람", "응답자", "유저", "user", "panel", "people", "respondent"]
    
    # 특정 조건을 의미하는 키워드 (이것들이 있으면 전체 조회가 아님)
    condition_keywords = [
        "남성", "여성", "남자", "여자", "male", "female",
        "세", "살", "대", "age", "년생",
        "서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        "기혼", "미혼", "결혼", "싱글",
        "대학", "고등학교", "대학원", "학력",
        "직업", "회사", "학생",
        "흡연", "담배", "음주", "술",
        "차", "자동차", "차량",
        "핸드폰", "폰", "휴대폰", "아이폰", "갤럭시",
    ]
    
    # 전체 키워드와 대상 키워드가 모두 포함되어 있는지 확인
    has_all_keyword = any(kw in query_lower for kw in all_keywords)
    has_target_keyword = any(kw in query_lower for kw in target_keywords)
    has_condition = any(kw in query_lower for kw in condition_keywords)
    
    # 매우 짧은 쿼리 (5자 이하)도 전체 조회로 간주
    is_very_short = len(query_lower.replace(" ", "")) <= 5 and has_all_keyword
    
    is_all_panels_query = (has_all_keyword and has_target_keyword and not has_condition) or is_very_short
    
    if is_all_panels_query:
        print(f"🔍 전체 패널 조회 감지: {query}")
        repo_panel = PanelRepository(session=session)  # 세션 주입
        try:
            # 필터 없이 모든 패널 조회
            all_panels = await repo_panel.filter_by_structured_filters(
                filters=None,  # 필터 없음
                limit=None,    # 제한 없음
                query=query
            )
            print(f"  ✓ 전체 패널 조회 완료: {len(all_panels)}개")
            
            # 결과 포맷팅 (SearchResultItem 형태로)
            results = [
                SearchResultItem(
                    panel_id=str(p["panel_id"]),
                    score=1.0,  # 동일한 점수
                    source="all",
                    accuracy_score=1.0,
                    vector_score=None,
                    fts_score=None,
                    rrf_score=None,
                    matched_fields=[],
                    gender=p.get("gender"),
                    age=p.get("age"),
                    region_city=p.get("region_city"),
                    region_gu=p.get("region_gu"),
                    marital_status=p.get("marital_status"),
                    children_count=p.get("children_count"),
                    family_size=p.get("family_size"),
                    education_level=p.get("education_level"),
                    occupation=p.get("occupation"),
                    monthly_personal_income=p.get("monthly_personal_income"),
                    monthly_household_income=p.get("monthly_household_income"),
                    phone_brand=p.get("phone_brand"),
                    phone_model=p.get("phone_model"),
                    car_ownership=p.get("car_ownership"),
                    car_manufacturer=p.get("car_manufacturer"),
                    car_model=p.get("car_model"),
                    owned_electronics=p.get("owned_electronics"),
                    smoking_experience=p.get("smoking_experience"),
                    smoking_brand=p.get("smoking_brand"),
                    e_cig_heated_brand=p.get("e_cig_heated_brand"),
                    e_cig_liquid_brand=p.get("e_cig_liquid_brand"),
                    drinking_experience=p.get("drinking_experience"),
                    panel_summary_text=p.get("panel_summary_text")
                )
                for p in all_panels
            ]
            
            return SearchResponse(
                results=results,
                analysis={
                    "query": query,
                    "structured_chunks": [],
                    "unstructured_chunks": [],
                    "total_found": len(results),
                    "message": f"전체 패널 {len(results)}개를 조회했습니다."
                }
            )
        except Exception as e:
            print(f"  ⚠️ 전체 패널 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"전체 패널 조회 중 오류 발생: {str(e)}"
            )
    
    # 질의에서 개수 추출 (예: "5명 뽑아줘", "10개 추출해줘")
    extracted_count = extract_count_from_query(query)
    
    # 개수가 명시되면 그 개수 사용, 아니면 top_k 사용
    # top_k가 기본값(20)이거나 큰 값(10000 이상)이면 모든 결과 반환
    if extracted_count is not None:
        top_k = extracted_count
        print(f"📊 질의에서 개수 추출: {top_k}개")
    elif payload.top_k >= 10000:  # 큰 값인 경우 모든 결과 반환
        top_k = 1000000  # 충분히 큰 값 (실질적으로 제한 없음)
        print(f"📊 모든 결과 반환 모드: 제한 없음")
    elif payload.top_k == 20:  # 기본값인 경우
        # 모든 결과 반환을 위해 큰 값 설정
        top_k = 1000000  # 충분히 큰 값
        print(f"📊 질의에 개수 없음: 모든 결과 반환")
    else:
        # 명시적으로 top_k를 지정한 경우 (10000 미만)
        top_k = payload.top_k
        print(f"📊 지정된 top_k 사용: {top_k}개")
    
    # LLM 및 Repository 초기화
    llm = get_bedrock_llm()
    
    # LLM 연결 확인 (필수)
    if llm is None:
        error_msg = "❌ LLM 연결 실패: AWS Bedrock LLM이 초기화되지 않았습니다. "
        if not _HAS_BEDROCK:
            error_msg += "langchain-aws 라이브러리가 설치되지 않았습니다."
        else:
            bedrock_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK") or os.getenv("AWS_BEDROCK_API_KEY")
            if not bedrock_key:
                error_msg += "환경 변수 AWS_BEARER_TOKEN_BEDROCK 또는 AWS_BEDROCK_API_KEY가 설정되지 않았습니다."
            else:
                error_msg += "AWS Bedrock API 키 디코딩 또는 세션 생성에 실패했습니다."
        
        print(error_msg)
        raise HTTPException(
            status_code=503,
            detail=error_msg
        )
    
    # 🔑 핵심 수정: Repository에 세션 주입 (모든 검색 작업이 같은 세션 사용)
    repo_panel = PanelRepository(session=session)
    repo_doc = DocumentRepository(session=session)
    
    # Step 1: LangChain Chain으로 질의 분석 (의미있는 단위로 분리 + label.json 매핑)
    analysis_chain = create_query_analysis_chain(llm)
    
    # 분석 Chain 확인 (필수)
    if analysis_chain is None:
        error_msg = "❌ 질의 분석 Chain 생성 실패: LLM Chain이 생성되지 않았습니다."
        print(error_msg)
        raise HTTPException(
            status_code=503,
            detail=error_msg
        )
    
    analysis_info = None
    structured_chunks = []
    unstructured_chunks = []
    llm_analysis_time = 0.0
    
    # 질의 분석 수행 (필수)
    try:
        print("=" * 80)
        print("🔍 [1단계] LLM 질의 분석 시작")
        print("=" * 80)
        print(f"📝 원본 쿼리: {query}")
        print(f"⏱️  LLM 분석 시작...")
        
        llm_analysis_start_time = time_module.time()
        analysis_result = await analysis_chain.ainvoke({"query": query})
        llm_analysis_time = time_module.time() - llm_analysis_start_time
        
        analysis_info = parse_analysis_result(analysis_result)
        structured_chunks = analysis_info.get("structured_chunks", [])
        unstructured_chunks = analysis_info.get("unstructured_chunks", [])
        
        print(f"\n✅ LLM 분석 완료 (소요 시간: {llm_analysis_time:.2f}초)")
        print(f"📊 질의 분석 결과:")
        print(f"  - 의미있는 단위: {len(structured_chunks) + len(unstructured_chunks)}개")
        print(f"  - 정형 데이터 청크: {len(structured_chunks)}개")
        print(f"  - 비정형 데이터 청크: {len(unstructured_chunks)}개")
        
        # 정형 청크 상세 출력
        if structured_chunks:
            print(f"\n📋 정형 데이터 청크 상세:")
            for i, chunk in enumerate(structured_chunks, 1):
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text", "")
                    category = chunk.get("category", "")
                    db_column = chunk.get("db_column", "")
                    mapped_values = chunk.get("mapped_values", [])
                    print(f"  [{i}] 텍스트: '{chunk_text}'")
                    print(f"      카테고리: {category}")
                    print(f"      DB 컬럼: {db_column}")
                    print(f"      매핑된 값: {mapped_values[:5]}{'...' if len(mapped_values) > 5 else ''}")
                else:
                    print(f"  [{i}] {chunk}")
        
        # 비정형 청크 상세 출력
        if unstructured_chunks:
            print(f"\n📋 비정형 데이터 청크 상세:")
            for i, chunk in enumerate(unstructured_chunks, 1):
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text", "")
                    intent = chunk.get("intent", "neutral")
                    keywords = chunk.get("keywords", [])
                    related_segments = chunk.get("related_segments", [])
                    search_hints = chunk.get("search_hints", {})
                    exclude_patterns = search_hints.get("exclude_patterns", [])
                    print(f"  [{i}] 텍스트: '{chunk_text}'")
                    print(f"      의도: {intent}")
                    print(f"      키워드: {keywords[:5]}{'...' if len(keywords) > 5 else ''}")
                    print(f"      관련 세그먼트: {related_segments}")
                    if exclude_patterns:
                        print(f"      제외 패턴: {exclude_patterns[:3]}{'...' if len(exclude_patterns) > 3 else ''}")
                else:
                    print(f"  [{i}] {chunk}")
        
        # 정형 데이터 청크를 label.json 기반 필터로 매핑
        if structured_chunks:
            label_filters = map_chunks_to_label_filters(structured_chunks)
            print(f"\n📋 label.json 기반 필터 매핑 결과: {len(label_filters)}개 필터")
            for i, lf in enumerate(label_filters, 1):
                print(f"  [{i}] 카테고리: {lf['category']}")
                print(f"      매핑된 값: {lf['mapped_values'][:5]}{'...' if len(lf['mapped_values']) > 5 else ''} (총 {len(lf['mapped_values'])}개)")
    except Exception as e:
        error_msg = f"❌ 질의 분석 실패: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
    
    # Step 2 & 3: 정형 데이터 검색과 비정형 데이터 검색 준비 (병렬 실행을 위해)
    import time
    structured_panel_ids: List[str] = []
    label_filters = None  # label.json 기반 필터
    
    # 정형 검색 준비 (병렬 실행용)
    structured_search_task = None
    if structured_chunks:
        # label.json 기반 필터 생성
        label_filters = map_chunks_to_label_filters(structured_chunks)
    
    if label_filters or structured_chunks:
        print("🔍 정형 데이터 검색 준비 (label.json 기반 + SQL)...")
        # 정형 검색을 태스크로 생성 (병렬 실행용)
        async def run_structured_search():
            start_time = time_module.time()
            try:
                print("  ⏱️  정형 검색 시작...")
                # 정형 데이터는 조건을 만족하는 모든 패널 가져오기 (LIMIT 없음)
                structured_results = await repo_panel.filter_by_structured_filters(
                    filters=None,  # label_filters 사용 시 filters는 None
                    limit=None,  # 제한 없음 (조건 만족하는 모든 패널)
                    query=query,  # Fallback용
                    label_filters=label_filters  # label.json 기반 필터 전달
                )
                panel_ids = [str(p["panel_id"]) for p in structured_results]
                elapsed_time = time_module.time() - start_time
                print(f"\n✅ 정형 검색 완료: {len(panel_ids)}개 패널 (소요 시간: {elapsed_time:.2f}초)")
                if panel_ids:
                    print(f"📋 정형 검색 결과 샘플 (5개): {panel_ids[:5]}{'...' if len(panel_ids) > 5 else ''}")
                return panel_ids, elapsed_time
            except Exception as e:
                elapsed_time = time_module.time() - start_time
                print(f"  ⚠️ 정형 검색 실패 (소요 시간: {elapsed_time:.2f}초): {e}")
                import traceback
                traceback.print_exc()
                return [], elapsed_time
        
        structured_search_task = run_structured_search()
    
    # Step 3: 비정형 데이터 검색 (벡터 + FTS → RRF)
    unstructured_panel_ids: List[str] = []
    unstructured_accuracy_map: Dict[str, Dict[str, Any]] = {}  # 비정형 검색 정확도 정보
    
    # 개수 관련 비정형 청크 제외 (예: "100명", "5개" 등은 검색 조건이 아님)
    # 비정형 청크는 이제 딕셔너리 형태이므로 텍스트 추출 필요
    meaningful_unstructured_chunks = []
    print(f"  📝 비정형 청크 원본: {len(unstructured_chunks)}개")
    for chunk in unstructured_chunks:
        if isinstance(chunk, dict):
            chunk_text = chunk.get("text", "")
        else:
            # 기존 형식 호환성 (문자열)
            chunk_text = str(chunk) if chunk else ""
        
        # 개수 관련 키워드가 포함된 경우만 제외 (더 정확한 필터링)
        # "100명 뽑아줘" 같은 경우만 제외하고, "운동을 좋아하는 사람" 같은 경우는 포함
        is_count_related = any(
            kw in chunk_text for kw in ['명 뽑', '개 뽑', '명 추출', '개 추출', '명 추천', '개 추천', '명 보여', '개 보여']
        ) or (len(chunk_text.strip()) <= 5 and any(kw in chunk_text for kw in ['명', '개']))
        
        if chunk_text and chunk_text.strip() and not is_count_related:
            meaningful_unstructured_chunks.append(chunk)
            print(f"    ✓ 비정형 청크 포함: {chunk_text[:50]}")
        else:
            print(f"    ✗ 비정형 청크 제외: {chunk_text[:50]} (개수 관련 또는 빈 텍스트)")
    
    print(f"  📝 의미있는 비정형 청크: {len(meaningful_unstructured_chunks)}개")
    
    # 비정형 검색 준비 (병렬 실행용)
    unstructured_search_task = None
    if meaningful_unstructured_chunks:
        # 비정형 청크가 있을 때만 비정형 검색 수행
        # 각 비정형 청크를 개별적으로 검색하거나, 합쳐서 검색
        # 텍스트만 추출하여 쿼리 생성
        chunk_texts = [
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in meaningful_unstructured_chunks
        ]
        unstructured_query_text = " ".join(chunk_texts)  # 벡터 검색용으로는 합쳐서 사용
        print("🔍 비정형 데이터 검색 준비 (벡터 + FTS → RRF)...")
        print(f"    - 비정형 청크: {meaningful_unstructured_chunks}")
        
        # 비정형 검색을 태스크로 생성 (병렬 실행용)
        async def run_unstructured_search():
            start_time = time_module.time()
            try:
                print("  ⏱️  비정형 검색 시작...")
                # 2차 검색: 비정형 데이터는 top_k로 제한
                # 벡터 검색과 FTS는 충분히 많이 가져온 후 RRF 통합하고 top_k만 선택
                if top_k >= 1000000:
                    search_limit = 10000  # 제한 없음 모드에서는 충분히 큰 값 (100,000 → 10,000으로 축소)
                else:
                    # top_k에 비례하여 검색하되, 불필요하게 많이 가져오지 않음
                    # 벡터/FTS 검색이 각각 limit * 2로 최적화되었으므로, 여기서는 적절한 값 사용
                    search_limit = min(top_k * 10, 10000)  # 최대 10,000개로 제한 (50 → 10으로 축소)
                
                print(f"    - 검색 제한: 최대 {search_limit}개 세그먼트 검색 후 RRF 통합")
                
                # 메타데이터 수집: 모든 비정형 청크의 메타데이터 통합
                all_intents = [chunk.get("intent", "neutral") for chunk in meaningful_unstructured_chunks if isinstance(chunk, dict)]
                all_keywords = []
                all_exclude_patterns = []
                all_include_patterns = []
                all_related_segments = []  # 관련 세그먼트 수집
                
                for chunk in meaningful_unstructured_chunks:
                    if isinstance(chunk, dict):
                        all_keywords.extend(chunk.get("keywords", []))
                        search_hints = chunk.get("search_hints", {})
                        all_exclude_patterns.extend(search_hints.get("exclude_patterns", []))
                        all_include_patterns.extend(search_hints.get("include_patterns", []))
                        # related_segments 수집 (중복 제거)
                        related_segs = chunk.get("related_segments", [])
                        if related_segs:
                            all_related_segments.extend(related_segs)
                
                # 중복 제거 및 정리 (관련 세그먼트가 있으면 반드시 해당 세그먼트로 제한)
                all_related_segments = list(set(all_related_segments)) if all_related_segments else None
                
                if all_related_segments:
                    print(f"    - 관련 세그먼트: {all_related_segments} (총 {len(all_related_segments)}개)")
                else:
                    print(f"    - 전체 세그먼트 검색 (segment_filter 없음)")
                
                # 검색 쿼리 생성: 원본 텍스트 우선 사용 (의미 보존)
                # 원본 텍스트가 더 정확한 의미를 담고 있으므로 우선 사용
                search_query = unstructured_query_text.strip()
                
                if not search_query:
                    # 원본 텍스트가 비어있으면 키워드 사용
                    if all_keywords:
                        search_query = " ".join(all_keywords[:10])  # 최대 10개 키워드
                        print(f"    - 원본 텍스트 없음: 키워드 기반 검색 쿼리: {search_query}")
                    else:
                        # 키워드도 없으면 청크 텍스트에서 직접 추출
                        search_query = " ".join([chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) 
                                               for chunk in meaningful_unstructured_chunks if chunk])
                        print(f"    - 빈 쿼리 감지: 청크 텍스트에서 직접 추출: {search_query}")
                else:
                    print(f"    - 원본 텍스트 기반 검색 쿼리: {search_query}")
                    # 키워드 정보는 로그로만 출력 (참고용)
                    if all_keywords:
                        print(f"    - 참고: LLM 추출 키워드: {all_keywords[:10]}")
                    if not search_query:
                        # 텍스트가 비어있으면 청크 텍스트에서 직접 추출
                        search_query = " ".join([chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) 
                                               for chunk in meaningful_unstructured_chunks if chunk])
                        print(f"    - 빈 쿼리 감지: 청크 텍스트에서 직접 추출: {search_query}")
                    
                    if len(search_query.split()) > 10:
                        # 긴 쿼리인 경우 핵심 키워드만 추출
                        keywords = [w for w in search_query.split() if len(w) > 1]
                        search_query = " ".join(keywords[:10])  # 최대 10개 키워드만 사용
                        print(f"    - 긴 쿼리 감지: 핵심 키워드만 추출하여 검색: {search_query}")
                
                # 검색 쿼리 검증 (빈 쿼리면 원본 텍스트 사용)
                if not search_query or not search_query.strip():
                    print(f"    ⚠️ 검색 쿼리가 비어있습니다. 원본 텍스트 사용: {unstructured_query_text}")
                    search_query = unstructured_query_text.strip()
                    if not search_query:
                        # 여전히 비어있으면 청크 텍스트에서 직접 추출
                        search_query = " ".join([chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) 
                                               for chunk in meaningful_unstructured_chunks if chunk])
                        print(f"    - 청크 텍스트에서 직접 추출: {search_query}")
                
                # 최종 검색 쿼리 검증 (여전히 비어있으면 검색 불가)
                if not search_query or not search_query.strip():
                    print(f"    ❌ 검색 쿼리가 비어있어 비정형 검색을 건너뜁니다.")
                    print(f"    - unstructured_query_text: {unstructured_query_text}")
                    print(f"    - all_keywords: {all_keywords}")
                    print(f"    - meaningful_unstructured_chunks: {meaningful_unstructured_chunks}")
                else:
                    print(f"    ✓ 최종 검색 쿼리: {search_query}")
                
                # 의도 기반 필터링 결정
                # 긍정 의도가 하나라도 있으면 부정 표현 제외
                has_positive_intent = any(intent == "positive" for intent in all_intents)
                has_negative_intent = any(intent == "negative" for intent in all_intents)
                exclude_negative = has_positive_intent and not has_negative_intent
                
                if exclude_negative:
                    print(f"    - 긍정 의도 감지: 부정 표현 자동 제외")
                
                # 검색 쿼리가 비어있으면 검색하지 않음
                if not search_query or not search_query.strip():
                    print(f"    ⚠️ 검색 쿼리가 비어있어 벡터 검색과 FTS 검색을 건너뜁니다.")
                    vector_results = []
                    vector_ids = []
                    vector_scores_map = {}
                    fts_scores = {}
                    fts_ids = []
                    fts_scores_normalized = {}
                    fused_scores = {}
                    fts_chunk_queries = []  # 빈 리스트로 초기화
                else:
                    # 벡터 검색과 FTS 검색을 병렬로 실행하기 위해 FTS 검색 쿼리 먼저 준비
                    # FTS 검색 쿼리 준비 (나중에 순차 실행)
                    # 주의: fts_search_tasks는 더 이상 사용하지 않음 (순차 실행으로 변경)
                    fts_chunk_queries = []  # FTS 검색에 사용할 쿼리 정보 저장
                    for chunk in meaningful_unstructured_chunks:
                        # 딕셔너리 형태에서 텍스트 추출
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get("text", "")
                            # 메타데이터에서 exclude_patterns 추출
                            search_hints = chunk.get("search_hints", {})
                            chunk_exclude_patterns = search_hints.get("exclude_patterns", [])
                            chunk_intent = chunk.get("intent", "neutral")
                            chunk_exclude_negative = chunk_intent == "positive"
                            # keywords가 있으면 키워드 기반 검색
                            chunk_keywords = chunk.get("keywords", [])
                            if chunk_keywords:
                                # 키워드 정제: 동의어/중복 제거 및 불필요 키워드 제거
                                if not isinstance(chunk_keywords, list):
                                    chunk_keywords = [str(chunk_keywords)]
                                kws = [str(kw).strip() for kw in chunk_keywords if str(kw).strip()]
                                # 중복 제거(순서 유지)
                                seen = set()
                                kws = [k for k in kws if not (k in seen or seen.add(k))]
                                # '반려동물'이 있으면 '펫'은 제거 (과도한 AND 제약 완화)
                                if "반려동물" in kws and "펫" in kws:
                                    kws = [k for k in kws if k != "펫"]
                                # 키워드 수 제한 (최대 3개)
                                kws = kws[:3]
                                chunk_query = " ".join(kws)
                            else:
                                chunk_query = chunk_text
                        else:
                            # 기존 형식 호환성 (문자열)
                            chunk_text = str(chunk) if chunk else ""
                            chunk_query = chunk_text
                            chunk_exclude_patterns = []
                            chunk_exclude_negative = exclude_negative
                        
                        if chunk_query and chunk_query.strip():
                            # FTS 검색 정보 저장 (나중에 순차 실행)
                            fts_chunk_queries.append({
                                "query": chunk_query.strip(),
                                "exclude_negative": chunk_exclude_negative,
                                "exclude_patterns": chunk_exclude_patterns,
                                "segment_filter": all_related_segments
                            })
                    
                    print("\n" + "=" * 80)
                    print("🔍 [2단계] 형태소 분석 및 벡터 검색 준비")
                    print("=" * 80)
                    print(f"📝 검색 쿼리: {search_query}")
                    
                    # 형태소 단위 분할 벡터 검색 (개선된 방식)
                    # 쿼리를 형태소 단위로 분할하여 각각 검색하고 점수 합산
                    # 단, 너무 많이 나누면 성능 저하 및 정확도 하락이므로 최대 3개까지만 분할
                    print(f"\n🔬 형태소 분석 시작...")
                    morph_chunks = split_query_by_morphology(search_query, max_chunks=3)
                
                if len(morph_chunks) > 1:
                    print(f"✅ 형태소 분석 완료: {len(morph_chunks)}개 청크로 분할")
                    print(f"📋 분할된 형태소 청크:")
                    for i, morph_chunk in enumerate(morph_chunks, 1):
                        print(f"  [{i}] '{morph_chunk}'")
                    print(f"\n🔍 각 형태소 청크별 벡터 검색 시작 (병렬 처리)...")
                    
                    # 각 형태소 청크로 벡터 검색 수행 (병렬 처리로 속도 향상)
                    all_morph_results: Dict[str, List[Dict[str, Any]]] = {}  # 패널별 결과 리스트
                    
                    # 병렬 검색: 모든 형태소 청크를 동시에 검색
                    search_tasks = [
                        repo_doc.semantic_search(
                            query=morph_chunk,
                            limit=search_limit,  # 각 청크마다 충분히 많이 검색
                            exclude_negative=exclude_negative,
                            exclude_patterns=all_exclude_patterns if all_exclude_patterns else None,
                            segment_filter=all_related_segments,
                            min_similarity=0.50  # 형태소 단위 검색이므로 임계값을 0.45 → 0.50으로 상향 (정확도 향상)
                        )
                        for morph_chunk in morph_chunks
                    ]
                    
                    # 벡터 검색과 FTS 검색을 순차적으로 실행 (세션 동시성 문제 해결)
                    # 각 검색 작업이 독립적인 DocumentRepository 인스턴스를 사용하도록 수정
                    print(f"  ⏱️  벡터 검색 태스크: {len(morph_chunks)}개, FTS 검색 태스크: {len(fts_chunk_queries)}개")
                    print(f"  ⚠️  세션 동시성 문제 방지를 위해 순차 실행 (독립 세션 사용)")
                    
                    # 벡터 검색 먼저 실행 (독립 세션 사용)
                    vector_results_list = []
                    for morph_chunk in morph_chunks:
                        vector_repo = DocumentRepository(session=None)  # 독립 세션 사용
                        result = await vector_repo.semantic_search(
                            query=morph_chunk,
                            limit=search_limit,
                            exclude_negative=exclude_negative,
                            exclude_patterns=all_exclude_patterns if all_exclude_patterns else None,
                            segment_filter=all_related_segments,
                            min_similarity=0.50
                        )
                        vector_results_list.append(result)
                    
                    # FTS 검색 실행 (독립 세션 사용)
                    fts_results_list = []
                    for fts_info in fts_chunk_queries:
                        fts_repo = DocumentRepository(session=None)  # 독립 세션 사용
                        result = await fts_repo.fulltext_search(
                            query=fts_info["query"], 
                            limit=search_limit,
                            use_or=False,
                            exclude_negative=fts_info["exclude_negative"],
                            exclude_patterns=fts_info["exclude_patterns"],
                            segment_filter=fts_info["segment_filter"]
                        )
                        fts_results_list.append(result)
                    
                    all_results = vector_results_list + fts_results_list
                    
                    # 결과 분리: 벡터 검색 결과와 FTS 검색 결과
                    all_chunk_results = all_results[:len(search_tasks)]  # 벡터 검색 결과
                    all_fts_results = all_results[len(search_tasks):]  # FTS 검색 결과
                    
                    print(f"\n✅ 벡터 검색 완료 (형태소별 결과):")
                    # 각 형태소별 검색 결과 상세 출력
                    for i, (morph_chunk, chunk_results) in enumerate(zip(morph_chunks, all_chunk_results), 1):
                        print(f"  [{i}] 형태소: '{morph_chunk}'")
                        print(f"      검색 결과: {len(chunk_results)}개 세그먼트")
                        if chunk_results:
                            top_5 = sorted(chunk_results, key=lambda x: x.get("score", 0), reverse=True)[:5]
                            print(f"      상위 5개 결과:")
                            for j, result in enumerate(top_5, 1):
                                panel_id = result.get("panel_id", "")
                                score = result.get("score", 0.0)
                                segment_name = result.get("segment_name", "")
                                print(f"        [{j}] 패널: {panel_id}, 세그먼트: {segment_name}, 유사도: {score:.4f}")
                    
                    # 패널별로 결과 수집 (부정 표현 필터링)
                    for chunk_results in all_chunk_results:
                        for result in chunk_results:
                            panel_id = result["panel_id"]
                            if panel_id:
                                panel_id_str = str(panel_id)
                                
                                # 부정 표현 필터링 (긍정 의도 쿼리인 경우)
                                if exclude_negative and all_exclude_patterns:
                                    # result에 summary_text가 없으면 나중에 필터링
                                    # 일단 수집하고 나중에 필터링
                                    pass
                                
                                if panel_id_str not in all_morph_results:
                                    all_morph_results[panel_id_str] = []
                                all_morph_results[panel_id_str].append(result)
                    
                    # 패널별 점수 합산 (평균 또는 가중합)
                    print(f"\n📊 패널별 점수 합산 시작...")
                    vector_scores_map: Dict[str, float] = {}
                    sample_panels = []  # 샘플 출력용
                    for panel_id_str, results in all_morph_results.items():
                        scores = []
                        segment_names = []
                        for result in results:
                            score = result.get("score", 0.0)
                            segment_name = result.get("segment_name")
                            segment_names.append(segment_name)
                            
                            # 세그먼트 신뢰도 점수 적용
                            reliability_multiplier = _get_segment_reliability_score(segment_name) if segment_name else 1.0
                            adjusted_score = float(score) * reliability_multiplier
                            scores.append(adjusted_score)
                        
                        # 점수 합산 방식: 평균 + 최고값 가중 (더 정확한 매칭에 보너스)
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            max_score = max(scores)
                            # 평균 70% + 최고값 30% (가중합)
                            final_score = avg_score * 0.7 + max_score * 0.3
                            vector_scores_map[panel_id_str] = final_score
                            
                            # 상위 5개 패널 샘플 저장
                            if len(sample_panels) < 5:
                                sample_panels.append({
                                    "panel_id": panel_id_str,
                                    "scores": scores,
                                    "avg_score": avg_score,
                                    "max_score": max_score,
                                    "final_score": final_score,
                                    "segments": list(set(segment_names))
                                })
                    
                    print(f"✅ 점수 합산 완료: {len(vector_scores_map)}개 패널")
                    if sample_panels:
                        print(f"\n📋 상위 패널 점수 합산 샘플 (5개):")
                        sorted_samples = sorted(sample_panels, key=lambda x: x["final_score"], reverse=True)
                        for i, sample in enumerate(sorted_samples, 1):
                            print(f"  [{i}] 패널: {sample['panel_id']}")
                            print(f"      개별 점수: {[f'{s:.4f}' for s in sample['scores'][:3]]}{'...' if len(sample['scores']) > 3 else ''} (총 {len(sample['scores'])}개)")
                            print(f"      평균: {sample['avg_score']:.4f}, 최고값: {sample['max_score']:.4f}")
                            print(f"      최종 점수: {sample['final_score']:.4f} (평균 70% + 최고값 30%)")
                            print(f"      매칭된 세그먼트: {sample['segments'][:3]}{'...' if len(sample['segments']) > 3 else ''}")
                    
                    # 부정 표현 필터링 (긍정 의도 쿼리인 경우)
                    if exclude_negative and all_exclude_patterns and vector_scores_map:
                        print(f"\n🚫 벡터 검색 결과 부정 표현 필터링 시작...")
                        # 상위 후보 패널들의 summary_text 조회
                        candidate_panel_ids = list(vector_scores_map.keys())[:min(1000, len(vector_scores_map))]
                        
                        if candidate_panel_ids:
                            # 🔑 세션 공유: 주입받은 세션 사용 (새 세션 생성 안 함)
                            placeholders = [f":pid_{i}" for i in range(len(candidate_panel_ids))]
                            params = {f"pid_{i}": pid for i, pid in enumerate(candidate_panel_ids)}
                            
                            # 패널별 summary_text 조회
                            from sqlalchemy import text as sql_text
                            text_query = sql_text(f"""
                                SELECT DISTINCT panel_id, summary_text
                                FROM panel_summary_segments
                                WHERE panel_id IN ({','.join(placeholders)})
                                  AND summary_text IS NOT NULL
                            """)
                            text_result = await session.execute(text_query, params)
                            text_rows = text_result.fetchall()
                            
                            # 부정 표현이 있는 패널 제외
                            filtered_count = 0
                            for pid, summary_text in text_rows:
                                if pid and summary_text:
                                    panel_id_str = str(pid)
                                    # 부정 패턴 확인
                                    has_negative = any(pattern in summary_text for pattern in all_exclude_patterns)
                                    if has_negative and panel_id_str in vector_scores_map:
                                        del vector_scores_map[panel_id_str]
                                        filtered_count += 1
                            
                            if filtered_count > 0:
                                print(f"  ✅ 부정 표현 필터링 완료: {filtered_count}개 패널 제외")
                    
                    vector_ids = list(vector_scores_map.keys())
                    print(f"\n✅ 형태소 분할 벡터 검색 완료: {len(vector_ids)}개 패널")
                    
                else:
                    # 형태소 분할이 불가능하거나 1개만 있는 경우 기존 방식 사용
                    print(f"✅ 형태소 분석 완료: 분할 불가 (단일 쿼리 사용)")
                    print(f"📝 검색 쿼리: '{search_query}'")
                    print(f"\n🔍 단일 쿼리 벡터 검색 시작...")
                    
                    # 벡터 검색과 FTS 검색을 순차적으로 실행 (세션 동시성 문제 해결)
                    # 각 검색 작업이 독립적인 DocumentRepository 인스턴스를 사용하도록 수정
                    vector_repo = DocumentRepository(session=None)  # 독립 세션 사용
                    vector_results = await vector_repo.semantic_search(
                        query=search_query, 
                        limit=search_limit,
                        exclude_negative=exclude_negative,
                        exclude_patterns=all_exclude_patterns if all_exclude_patterns else None,
                        segment_filter=all_related_segments,
                        min_similarity=0.55  # 기본 임계값 상향 (0.50 → 0.55, 정확도 향상)
                    )
                    
                    all_fts_results = []
                    # FTS 검색도 독립 세션 사용 (fts_chunk_queries 사용)
                    for fts_info in fts_chunk_queries:
                        fts_repo = DocumentRepository(session=None)  # 독립 세션 사용
                        result = await fts_repo.fulltext_search(
                            query=fts_info["query"], 
                            limit=search_limit,
                            use_or=False,
                            exclude_negative=fts_info["exclude_negative"],
                            exclude_patterns=fts_info["exclude_patterns"],
                            segment_filter=fts_info["segment_filter"]
                        )
                        all_fts_results.append(result)
                    
                    # 벡터 검색 결과에서 패널별 점수 저장 (세그먼트 신뢰도 적용)
                    print(f"\n📊 단일 쿼리 벡터 검색 결과 처리...")
                    vector_scores_map_single: Dict[str, float] = {}
                    sample_results = []  # 샘플 출력용
                    for result in vector_results:
                        panel_id = result["panel_id"]
                        score = result.get("score", 0.0)
                        segment_name = result.get("segment_name")
                        
                        # 세그먼트 신뢰도 점수 적용
                        reliability_multiplier = _get_segment_reliability_score(segment_name) if segment_name else 1.0
                        adjusted_score = float(score) * reliability_multiplier
                        
                        if panel_id:
                            # 같은 패널이 여러 세그먼트에서 나오면 최고 점수 유지
                            panel_id_str = str(panel_id)
                            old_score = vector_scores_map_single.get(panel_id_str, 0.0)
                            if adjusted_score > old_score:
                                vector_scores_map_single[panel_id_str] = adjusted_score
                            
                            # 상위 5개 샘플 저장
                            if len(sample_results) < 5:
                                sample_results.append({
                                    "panel_id": panel_id_str,
                                    "score": score,
                                    "adjusted_score": adjusted_score,
                                    "segment_name": segment_name,
                                    "reliability": reliability_multiplier
                                })
                    
                    if sample_results:
                        print(f"  📋 단일 쿼리 벡터 검색 상위 결과 샘플 (5개):")
                        sorted_samples = sorted(sample_results, key=lambda x: x["adjusted_score"], reverse=True)
                        for i, sample in enumerate(sorted_samples, 1):
                            print(f"    [{i}] 패널: {sample['panel_id']}, 세그먼트: {sample['segment_name']}")
                            print(f"        원본 점수: {sample['score']:.4f}, 신뢰도 배수: {sample['reliability']:.2f}")
                            print(f"        조정 점수: {sample['adjusted_score']:.4f}")
                    
                    # 형태소 분할 검색과 동일한 변수명으로 통일
                    vector_scores_map = vector_scores_map_single
                    vector_ids = [r["panel_id"] for r in vector_results]
                    
                    # 부정 표현 필터링 (긍정 의도 쿼리인 경우)
                    if exclude_negative and all_exclude_patterns and vector_scores_map:
                        print(f"\n🚫 벡터 검색 결과 부정 표현 필터링 시작...")
                        # 상위 후보 패널들의 summary_text 조회
                        candidate_panel_ids = list(vector_scores_map.keys())[:min(1000, len(vector_scores_map))]
                        
                        if candidate_panel_ids:
                            # 🔑 세션 공유: 주입받은 세션 사용 (새 세션 생성 안 함)
                            placeholders = [f":pid_{i}" for i in range(len(candidate_panel_ids))]
                            params = {f"pid_{i}": pid for i, pid in enumerate(candidate_panel_ids)}
                            
                            # 패널별 summary_text 조회
                            from sqlalchemy import text as sql_text
                            text_query = sql_text(f"""
                                SELECT DISTINCT panel_id, summary_text
                                FROM panel_summary_segments
                                WHERE panel_id IN ({','.join(placeholders)})
                                  AND summary_text IS NOT NULL
                            """)
                            text_result = await session.execute(text_query, params)
                            text_rows = text_result.fetchall()
                            
                            # 부정 표현이 있는 패널 제외
                            filtered_count = 0
                            for pid, summary_text in text_rows:
                                if pid and summary_text:
                                    panel_id_str = str(pid)
                                    # 부정 패턴 확인
                                    has_negative = any(pattern in summary_text for pattern in all_exclude_patterns)
                                    if has_negative and panel_id_str in vector_scores_map:
                                        del vector_scores_map[panel_id_str]
                                        filtered_count += 1
                            
                            if filtered_count > 0:
                                print(f"  ✅ 부정 표현 필터링 완료: {filtered_count}개 패널 제외")
                        
                        # 필터링 후 vector_ids 업데이트
                        vector_ids = [pid for pid in vector_ids if pid in vector_scores_map]
                    
                    print(f"\n✅ 단일 쿼리 벡터 검색 완료: {len(vector_ids)}개 패널")
                
                # Full-Text Search 결과 처리 (벡터 검색과 병렬로 실행됨)
                print("\n" + "=" * 80)
                print("🔍 [3단계] FTS 검색 결과 처리")
                print("=" * 80)
                # FTS는 키워드 기반이므로 각 청크를 개별 검색하는 것이 더 정확함
                # AND 연산 사용: 모든 키워드가 포함되어야 매칭 (정확도 우선)
                # 과도하게 제한되어도 벡터 검색(의미 기반)과 RRF로 통합하여 보완
                # 부정어는 FTS 쿼리에 직접 포함 (LIKE 사후 필터링보다 효율적)
                fts_scores: Dict[str, float] = {}  # 패널별 점수 누적
                
                # FTS 검색 결과 처리 (이미 병렬로 실행됨)
                if all_fts_results:
                    print(f"📋 FTS 검색 결과 처리: {len(all_fts_results)}개 청크 검색 결과")
                    
                    # 각 청크 검색 결과의 점수를 합산 (여러 청크에서 매칭되면 점수 누적)
                    for i, chunk_fts_results in enumerate(all_fts_results, 1):
                        print(f"  [{i}] 청크 검색 결과: {len(chunk_fts_results)}개 세그먼트")
                    for result in chunk_fts_results:
                        panel_id = result["panel_id"]
                        score = result.get("score", 0.0)
                        if panel_id not in fts_scores:
                            fts_scores[panel_id] = 0.0
                        fts_scores[panel_id] += score  # 점수 누적
                    
                    print(f"✅ FTS 점수 누적 완료: {len(fts_scores)}개 패널")
                else:
                    print(f"⚠️ FTS 검색 결과 없음")
                
                # FTS 점수 정규화 (최대값으로 나누어 0~1 범위로)
                max_fts_score = max(fts_scores.values()) if fts_scores else 1.0
                if max_fts_score > 0:
                    fts_scores_normalized = {pid: score / max_fts_score for pid, score in fts_scores.items()}
                    print(f"📊 FTS 점수 정규화 완료: 최대값 {max_fts_score:.4f}")
                else:
                    fts_scores_normalized = fts_scores
                    print(f"⚠️ FTS 점수 정규화 불가 (최대값 0)")
                
                # 점수 기준으로 정렬하여 패널 ID 리스트 생성
                sorted_fts = sorted(fts_scores.items(), key=lambda x: x[1], reverse=True)
                fts_ids = [pid for pid, _ in sorted_fts]
                
                # 상위 5개 결과 샘플 출력
                if sorted_fts[:5]:
                    print(f"\n📋 FTS 검색 상위 결과 샘플 (5개):")
                    for i, (pid, score) in enumerate(sorted_fts[:5], 1):
                        normalized = fts_scores_normalized.get(pid, 0.0)
                        print(f"  [{i}] 패널: {pid}, 원본 점수: {score:.4f}, 정규화 점수: {normalized:.4f}")
                
                print(f"\n✅ FTS 검색 완료: {len(fts_ids)}개 패널 (AND 연산, 부정어 쿼리 포함)")
                
                # RRF 통합 (벡터:FTS = 6:4 가중치)
                print("\n" + "=" * 80)
                print("🔍 [4단계] RRF 통합 (벡터 + FTS)")
                print("=" * 80)
                print(f"📊 벡터 검색 결과: {len(vector_ids)}개 패널")
                print(f"📊 FTS 검색 결과: {len(fts_ids)}개 패널")
                print(f"⚙️  RRF 파라미터: k=100, 벡터 가중치=0.6, FTS 가중치=0.4")
                
                fused_scores = _fuse_scores_by_rrf(
                    vector_ids, fts_ids, 
                    k=100,  # k=60 → k=100으로 증가
                    weight_a=0.6,  # 벡터 검색 60% (의미 기반 검색 우선)
                    weight_b=0.4   # FTS 검색 40% (키워드 매칭 보조)
                )
                
                print(f"✅ RRF 통합 완료: {len(fused_scores)}개 패널")
                if fused_scores:
                    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"\n📋 RRF 통합 상위 결과 샘플 (5개):")
                    for i, (pid, score) in enumerate(sorted_fused[:5], 1):
                        print(f"  [{i}] 패널: {pid}, RRF 점수: {score:.4f}")
                
                # 동적 의도 일치도 계산 및 점수 조정 (모든 상황에 적용)
                # 1. 쿼리에서 긍정/부정 의도 감지 (일반적인 패턴)
                def detect_intent(text: str) -> str:
                    """텍스트의 긍정/부정 의도를 감지 (동적)"""
                    if not text:
                        return "neutral"
                    
                    # 부정 표현 패턴 (일반적)
                    neg_patterns_default = [
                        "없다", "없음", "하지 않", "하지 않는다", "하지 않음",
                        "키워본 적이 없다", "키운 적이 없다", "사용하지 않", "이용하지 않", "보지 않",
                        "선호하지 않", "안 한다", "안한다", "못 한다", "못한다"
                    ]
                    # 긍정 표현 패턴 (일반적)
                    positive_patterns = [
                        "있다", "있음", "한다", "한다", "선호", "좋아", "원함"
                    ]
                    
                    text_lower = text.lower()
                    has_negative = any(pattern in text_lower for pattern in neg_patterns_default)
                    has_positive = any(pattern in text_lower for pattern in positive_patterns)
                    
                    # 구체적인 명사나 객체가 있으면 긍정으로 간주 (예: "백화점 상품권", "아이폰")
                    # 부정 표현이 명시적으로 없으면 긍정
                    if has_negative and not has_positive:
                        return "negative"
                    elif has_positive or (not has_negative and len(text.strip()) > 0):
                        return "positive"
                    else:
                        return "neutral"
                
                query_intent = detect_intent(unstructured_query_text)
                
                # 2. 결과 텍스트에서 의도 감지 및 점수 조정
                if query_intent in ["positive", "negative"] and (vector_ids or fts_ids):
                    # 상위 후보 패널들의 summary_text 조회
                    candidate_union = list(set((vector_ids or []) + (fts_ids or [])))
                    candidate_panel_ids = candidate_union[:min(1000, len(candidate_union))]
                    
                    if candidate_panel_ids:
                        # 🔑 세션 공유: 주입받은 세션 사용 (새 세션 생성 안 함)
                        placeholders = [f":pid_{i}" for i in range(len(candidate_panel_ids))]
                        params = {f"pid_{i}": pid for i, pid in enumerate(candidate_panel_ids)}
                        
                        # 패널별 summary_text 조회
                        from sqlalchemy import text as sql_text
                        text_query = sql_text(f"""
                            SELECT DISTINCT panel_id, summary_text
                            FROM panel_summary_segments
                            WHERE panel_id IN ({','.join(placeholders)})
                              AND summary_text IS NOT NULL
                        """)
                        text_result = await session.execute(text_query, params)
                        text_rows = text_result.fetchall()
                        
                        # 각 결과의 의도를 감지하고 쿼리 의도와 비교
                        intent_mismatch_count = 0  # 의도 불일치 개수 카운트
                        for pid, summary_text in text_rows:
                            if pid and summary_text:
                                panel_id_str = str(pid)
                                result_intent = detect_intent(summary_text)
                                
                                # 의도가 일치하지 않으면 점수에 페널티 적용
                                if query_intent != result_intent and result_intent != "neutral":
                                    if panel_id_str in fused_scores:
                                        # 의도 불일치 시 점수 50% 감소
                                        fused_scores[panel_id_str] *= 0.5
                                        intent_mismatch_count += 1
                        
                        # 의도 불일치 요약 통계만 출력 (개별 로그 대신)
                        if intent_mismatch_count > 0:
                            print(f"    ⚠️ 의도 불일치 감지: 총 {intent_mismatch_count}개 패널 (쿼리 의도: {query_intent}, 점수 50% 감소 적용)")
                
                sorted_panels = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 비정형 검색 정확도 정보 저장 (나중에 사용)
                for pid in vector_ids + fts_ids:
                    if pid not in unstructured_accuracy_map:
                        unstructured_accuracy_map[pid] = {
                            "vector_score": vector_scores_map.get(pid, 0.0),
                            "fts_score": fts_scores_normalized.get(pid, 0.0),
                            "rrf_score": fused_scores.get(pid, 0.0),
                        }
                
                # top_k보다 충분히 많이 가져와서 후처리에서 선택
                # 교집합을 위해 정형 검색 결과와 겹칠 가능성을 고려하여 충분히 많이 가져옴
                result_panel_ids = []
                if top_k >= 1000000:
                    result_panel_ids = [pid for pid, _ in sorted_panels]
                else:
                    # 교집합을 위해 더 많이 가져오기 (정형 검색 결과와 겹칠 가능성 고려)
                    # 정형 검색 결과가 많을수록 교집합 확률이 높으므로 더 많이 가져옴
                    # 하지만 불필요하게 많이 가져오지 않음 (20 → 10으로 축소)
                    unstructured_limit = min(top_k * 10, 5000)  # 최대 5,000개로 제한
                    result_panel_ids = [pid for pid, _ in sorted_panels[:min(unstructured_limit, len(sorted_panels))]]
                
                print(f"  ✓ 2차 비정형 검색 결과: {len(result_panel_ids)}개 패널 (top-{top_k} 요청, 교집합을 위해 {min(unstructured_limit if top_k < 1000000 else len(sorted_panels), len(sorted_panels))}개 검색)")
                elapsed_time = time_module.time() - start_time
                print(f"✅ 비정형 검색 완료 (소요 시간: {elapsed_time:.2f}초)")
                return result_panel_ids, elapsed_time
            except Exception as e:
                elapsed_time = time_module.time() - start_time
                print(f"  ⚠️ 비정형 검색 실패 (소요 시간: {elapsed_time:.2f}초): {e}")
                import traceback
                traceback.print_exc()
                return [], elapsed_time
        
        unstructured_search_task = run_unstructured_search()
    
    # 정형 검색과 비정형 검색을 병렬로 실행 (속도 향상)
    print("\n🚀 정형 검색과 비정형 검색 병렬 실행 중...")
    structured_time = 0.0
    unstructured_time = 0.0
    
    if structured_search_task and unstructured_search_task:
        # 둘 다 있는 경우: 병렬 실행
        (structured_panel_ids, structured_time), (unstructured_panel_ids, unstructured_time) = await asyncio.gather(
            structured_search_task,
            unstructured_search_task
        )
        print(f"✅ 병렬 실행 완료 - 정형: {structured_time:.2f}초, 비정형: {unstructured_time:.2f}초")
    elif structured_search_task:
        # 정형 검색만 있는 경우
        structured_panel_ids, structured_time = await structured_search_task
        unstructured_panel_ids = []
        print(f"✅ 정형 검색만 실행 완료: {structured_time:.2f}초")
    elif unstructured_search_task:
        # 비정형 검색만 있는 경우
        unstructured_panel_ids, unstructured_time = await unstructured_search_task
        structured_panel_ids = []
        print(f"✅ 비정형 검색만 실행 완료: {unstructured_time:.2f}초")
    else:
        # 둘 다 없는 경우
        structured_panel_ids = []
        unstructured_panel_ids = []
        print("⚠️ 정형/비정형 검색 모두 없음")
    
    # Step 4: 교집합 계산 (최종 패널 출력)
    print("\n" + "=" * 80)
    print("🔍 [5단계] 최종 결과 통합 (교집합)")
    print("=" * 80)
    print(f"📊 정형 검색 결과: {len(structured_panel_ids)}개 패널 (소요 시간: {structured_time:.2f}초)")
    print(f"📊 비정형 검색 결과: {len(unstructured_panel_ids)}개 패널 (소요 시간: {unstructured_time:.2f}초)")
    
    # 교집합 계산 시간 측정
    intersection_start_time = time_module.time()
    intersection_time = 0.0
    
    final_panel_ids: List[str] = []
    
    # 디버깅: 각 단계별 결과 확인
    print(f"\n📋 검색 단계별 결과:")
    print(f"  - structured_panel_ids: {len(structured_panel_ids)}개")
    print(f"  - unstructured_panel_ids: {len(unstructured_panel_ids)}개")
    if structured_panel_ids:
        print(f"  - structured_panel_ids 샘플: {structured_panel_ids[:3]}")
    if unstructured_panel_ids:
        print(f"  - unstructured_panel_ids 샘플: {unstructured_panel_ids[:3]}")
    
    if structured_panel_ids and unstructured_panel_ids:
        # 둘 다 있는 경우: 교집합에서 top_k 선택
        # 비정형 검색 결과(unstructured_panel_ids)의 순서를 유지하면서 교집합을 계산
        print(f"\n🔍 교집합 계산 중...")
        structured_set = set(structured_panel_ids)
        intersection = [
            pid for pid in unstructured_panel_ids if pid in structured_set
        ]
        intersection_time = time_module.time() - intersection_start_time
        print(f"✅ 교집합 완료: {len(intersection)}개 패널 (소요 시간: {intersection_time:.2f}초)")
        
        # 하이브리드 검색 총 시간 계산
        hybrid_total_time = structured_time + unstructured_time + intersection_time
        print(f"\n📊 하이브리드 검색 성능 요약:")
        print(f"  - 정형 검색: {structured_time:.2f}초 ({len(structured_panel_ids)}개 패널)")
        print(f"  - 비정형 검색: {unstructured_time:.2f}초 ({len(unstructured_panel_ids)}개 패널)")
        print(f"  - 교집합 계산: {intersection_time:.2f}초 ({len(intersection)}개 패널)")
        print(f"  - 하이브리드 총 시간: {hybrid_total_time:.2f}초")
        if structured_time > 0 and unstructured_time > 0:
            print(f"  - 병렬 처리 효과: 순차 실행 시 {structured_time + unstructured_time:.2f}초 → 병렬 실행 시 {max(structured_time, unstructured_time):.2f}초 (절약: {min(structured_time, unstructured_time):.2f}초)")
        
        # 교집합이 부족한 경우: 정형 검색 결과를 재확인 (이제는 이미 모든 패널을 가져왔으므로 실행될 일이 거의 없음)
        if len(intersection) < top_k and top_k < 1000000:
            print(f"  ⚠️ 교집합이 부족합니다 ({len(intersection)}개 < {top_k}개). 정형 검색 결과를 재확인합니다...")
            # 정형 검색 결과 재확인 (이미 모든 패널을 가져왔으므로 동일한 결과가 나올 것임)
            try:
                expanded_structured_results = await repo_panel.filter_by_structured_filters(
                    filters=None,
                    limit=None,  # 제한 없음 (모든 패널)
                    query=query,
                    label_filters=label_filters
                )
                expanded_structured_panel_ids = [str(p["panel_id"]) for p in expanded_structured_results]
                expanded_structured_set = set(expanded_structured_panel_ids)
                
                # 확장된 정형 검색 결과와 비정형 검색 결과의 교집합 재계산
                expanded_intersection = [
                    pid for pid in unstructured_panel_ids if pid in expanded_structured_set
                ]
                
                if len(expanded_intersection) > len(intersection):
                    intersection = expanded_intersection
                    print(f"  ✓ 확장된 교집합: {len(intersection)}개 패널 (정형 {len(expanded_structured_panel_ids)}개 중)")
            except Exception as e:
                print(f"  ⚠️ 확장된 정형 검색 실패: {e}")
        
        # 교집합 결과에서 top_k만 선택
        if top_k >= 1000000:
            final_panel_ids = intersection
        else:
            final_panel_ids = intersection[:top_k]
        print(f"✅ 최종 결과: {len(final_panel_ids)}개 패널 (요청: {top_k}개, 교집합 {len(intersection)}개 중)")
        if final_panel_ids:
            print(f"📋 최종 결과 샘플 (5개): {final_panel_ids[:5]}{'...' if len(final_panel_ids) > 5 else ''}")
    elif structured_panel_ids:
        # 정형만 있는 경우: top_k만큼 선택 (top_k가 매우 큰 값이면 모든 결과 반환)
        intersection_time = time_module.time() - intersection_start_time
        print(f"\n🔍 정형 검색 결과만 사용 (비정형 검색 결과 없음)")
        if top_k >= 1000000:
            final_panel_ids = structured_panel_ids
        else:
            # 정형 검색 결과가 top_k보다 적으면 있는 만큼만, 많으면 top_k만큼
            final_panel_ids = structured_panel_ids[:top_k]
        print(f"✅ 최종 결과: {len(final_panel_ids)}개 패널 (요청: {top_k}개, 정형 {len(structured_panel_ids)}개 중)")
        print(f"\n📊 정형 검색 성능 요약:")
        print(f"  - 정형 검색: {structured_time:.2f}초 ({len(structured_panel_ids)}개 패널)")
        print(f"  - 총 시간: {structured_time:.2f}초")
        if final_panel_ids:
            print(f"📋 최종 결과 샘플 (5개): {final_panel_ids[:5]}{'...' if len(final_panel_ids) > 5 else ''}")
    elif unstructured_panel_ids:
        # 비정형만 있는 경우: top_k 제한 (top_k가 매우 큰 값이면 모든 결과 반환)
        intersection_time = time_module.time() - intersection_start_time
        print(f"\n🔍 비정형 검색 결과만 사용 (정형 검색 결과 없음)")
        if top_k >= 1000000:
            final_panel_ids = unstructured_panel_ids
        else:
            final_panel_ids = unstructured_panel_ids[:top_k]
        print(f"✅ 최종 결과: {len(final_panel_ids)}개 패널 (요청: {top_k}개, 비정형 {len(unstructured_panel_ids)}개 중)")
        print(f"\n📊 비정형 검색 성능 요약:")
        print(f"  - 비정형 검색: {unstructured_time:.2f}초 ({len(unstructured_panel_ids)}개 패널)")
        print(f"  - 총 시간: {unstructured_time:.2f}초")
        if final_panel_ids:
            print(f"📋 최종 결과 샘플 (5개): {final_panel_ids[:5]}{'...' if len(final_panel_ids) > 5 else ''}")
    else:
        print(f"\n⚠️ 검색 결과 없음 (정형/비정형 모두 결과 없음)")
    
    # Step 5: 최종 결과 구성 (패널 기본 정보 포함)
    # final_panel_ids는 이미 top_k로 제한되었으므로 추가 제한 불필요
    # 다만 혹시 모를 경우를 위해 한 번 더 확인
    if not final_panel_ids:
        print(f"⚠️ 경고: final_panel_ids가 비어있습니다!")
        print(f"  - structured_panel_ids: {len(structured_panel_ids)}개")
        print(f"  - unstructured_panel_ids: {len(unstructured_panel_ids)}개")
        print(f"  - top_k: {top_k}")
    
    if top_k >= 1000000:
        limited_panel_ids = final_panel_ids  # 제한 없음
    else:
        # 이미 top_k로 제한되었지만, 혹시 더 많으면 다시 제한
        limited_panel_ids = final_panel_ids[:top_k] if final_panel_ids else []
        if len(limited_panel_ids) < top_k and len(final_panel_ids) > len(limited_panel_ids):
            print(f"  ⚠️ 경고: 요청한 {top_k}개보다 적은 {len(limited_panel_ids)}개만 반환됩니다.")
    
    print(f"\n📊 최종 패널 ID 리스트: {len(limited_panel_ids)}개")
    if limited_panel_ids:
        print(f"  - 샘플: {limited_panel_ids[:5]}")
    else:
        print(f"  ⚠️ 경고: limited_panel_ids가 비어있습니다!")
    
    # 패널 기본 정보 조회 (비정형 데이터 제외)
    panel_data_map: Dict[str, Dict[str, Any]] = {}
    panel_data_fetch_time = 0.0
    if limited_panel_ids:
        try:
            panel_data_start_time = time_module.time()
            # 🔑 세션 공유: 주입받은 세션 사용 (새 세션 생성 안 함)
            # asyncpg 배열 바인딩: PanelRepository와 동일한 방식 사용
            placeholders = [f":pid_{i}" for i in range(len(limited_panel_ids))]
            params = {f"pid_{i}": pid for i, pid in enumerate(limited_panel_ids)}
            
            sql_query = text(f"""
                SELECT panel_id, gender, age, region_city, region_gu, marital_status,
                       children_count, family_size, education_level, occupation,
                       monthly_personal_income, monthly_household_income,
                       phone_brand, phone_model, car_ownership, car_manufacturer, car_model,
                       owned_electronics, smoking_experience, smoking_brand,
                       e_cig_heated_brand, e_cig_liquid_brand, drinking_experience,
                       panel_summary_text
                FROM panels
                WHERE panel_id IN ({','.join(placeholders)})
            """)
            
            result = await session.execute(sql_query, params)
            rows = result.fetchall()
            panel_data_fetch_time = time_module.time() - panel_data_start_time
            print(f"📊 패널 데이터 조회 완료: {len(rows)}개 패널 (소요 시간: {panel_data_fetch_time:.2f}초)")
            
            columns = [
                "panel_id", "gender", "age", "region_city", "region_gu", "marital_status",
                "children_count", "family_size", "education_level", "occupation",
                "monthly_personal_income", "monthly_household_income",
                "phone_brand", "phone_model", "car_ownership", "car_manufacturer", "car_model",
                "owned_electronics", "smoking_experience", "smoking_brand",
                "e_cig_heated_brand", "e_cig_liquid_brand", "drinking_experience",
                "panel_summary_text"
            ]
            
            for row in rows:
                panel_data_map[str(row[0])] = dict(zip(columns, row))
        except Exception as e:
            print(f"⚠️ 패널 데이터 조회 실패: {e}")
    
    # 정형 필터 매칭 정보 추적 (어떤 필드가 매칭되었는지)
    matched_fields_map: Dict[str, List[str]] = {}
    if label_filters:
        # label_filters에서 매칭된 필드 추출
        for filter_item in label_filters:
            category = filter_item.get("category")
            if category:
                # 모든 정형 검색 결과 패널에 해당 필드 추가
                for pid in structured_panel_ids:
                    if pid not in matched_fields_map:
                        matched_fields_map[pid] = []
                    if category not in matched_fields_map[pid]:
                        matched_fields_map[pid].append(category)
    
    # 비정형 검색 RRF 점수 최대값 계산 (정규화용)
    max_rrf_score = 0.0
    if unstructured_accuracy_map:
        max_rrf_score = max([info.get("rrf_score", 0.0) for info in unstructured_accuracy_map.values()], default=0.0)
    
    # 검색 결과 구성 (패널 기본 정보 + 정확도 정보 포함)
    results = []
    for idx, pid in enumerate(limited_panel_ids):
        panel_data = panel_data_map.get(pid, {})
        
        # 정확도 정보 계산
        accuracy_info = unstructured_accuracy_map.get(pid, {})
        vector_score = accuracy_info.get("vector_score", 0.0)
        fts_score = accuracy_info.get("fts_score", 0.0)
        rrf_score = accuracy_info.get("rrf_score", 0.0)
        
        # 정형 검색 매칭 여부
        is_structured_match = pid in structured_panel_ids
        matched_fields = matched_fields_map.get(pid, [])
        
        # 종합 정확도 점수 계산
        # 정형 검색: 필터 매칭이므로 질의의 모든 정형 조건을 만족하면 100%
        # 비정형 검색: RRF 점수를 0~1 범위로 정규화
        # 하이브리드: 정형 매칭 여부와 비정형 점수를 조합
        
        accuracy_score = 0.0
        
        # 정형 검색 점수 계산
        # 정형 검색 결과에 포함된 패널은 모두 SQL WHERE 절을 만족하므로 항상 1.0
        if is_structured_match:
            structured_score = 1.0
        else:
            structured_score = 0.0
        
        # 비정형 검색 점수 계산 (RRF 점수 정규화)
        if rrf_score > 0 and max_rrf_score > 0:
            # RRF 점수를 0~1 범위로 정규화 (최대값 기준)
            unstructured_score = min(rrf_score / max_rrf_score, 1.0)
        elif rrf_score > 0:
            # 최대값이 0이면 (이론적으로 발생하지 않아야 함) 현재 점수 사용
            unstructured_score = min(rrf_score * 10, 1.0)  # 경험적 스케일링
        else:
            unstructured_score = 0.0
        
        # 종합 정확도 점수 계산
        if is_structured_match and rrf_score > 0:
            # 정형 + 비정형 검색 모두 있는 경우
            # 정형은 항상 1.0이므로: 1.0 * 0.6 + unstructured_score * 0.4
            accuracy_score = 0.6 + (unstructured_score * 0.4)
        elif is_structured_match:
            # 정형 검색만: 정형 조건을 모두 만족하므로 1.0
            accuracy_score = 1.0
        elif rrf_score > 0:
            # 비정형 검색만: RRF 점수 기반
            accuracy_score = unstructured_score
        else:
            # 둘 다 없으면 (이론적으로 발생하지 않아야 함)
            accuracy_score = 0.0
        
        # 비정형 청크가 있었는데(사용자 의도가 있었는데) 해당 패널이 비정형 결과에 포함되지 않은 경우 보수적 패널티 적용
        try:
            had_unstructured_chunks = len(meaningful_unstructured_chunks) > 0  # 상위 스코프 변수 사용
        except Exception:
            had_unstructured_chunks = False

        if had_unstructured_chunks and pid not in unstructured_panel_ids:
            # 사용자가 비정형 조건을 명시했지만 패널이 해당 텍스트 근거에 매칭되지 않음
            # 과도한 "매우 높음" 라벨을 방지하기 위해 상한/가중치 감소
            # 상한 0.6, 구조적 매칭만인 경우 추가 감소
            accuracy_score = min(accuracy_score, 0.6)
            if is_structured_match and not (rrf_score > 0):
                accuracy_score *= 0.7
        
        # 정확도 점수를 0~1 범위로 제한
        accuracy_score = min(max(accuracy_score, 0.0), 1.0)
        
        # 검색 소스 결정
        if is_structured_match and pid in unstructured_panel_ids:
            source = "hybrid"
        elif is_structured_match:
            source = "structured"
        elif pid in unstructured_panel_ids:
            source = "unstructured"
        else:
            source = "unknown"
        
        results.append(
            SearchResultItem(
                panel_id=pid,
                score=1.0 - (idx * 0.01),  # 기존 순위 기반 점수 유지
                source=source,
                # 정확도 정보
                accuracy_score=round(accuracy_score, 3),
                vector_score=round(vector_score, 3) if vector_score > 0 else None,
                fts_score=round(fts_score, 3) if fts_score > 0 else None,
                rrf_score=round(rrf_score, 6) if rrf_score > 0 else None,
                matched_fields=matched_fields if matched_fields else None,
                # 패널 기본 정보
                gender=panel_data.get("gender"),
                age=panel_data.get("age"),
                region_city=panel_data.get("region_city"),
                region_gu=panel_data.get("region_gu"),
                marital_status=panel_data.get("marital_status"),
                children_count=panel_data.get("children_count"),
                family_size=panel_data.get("family_size"),
                education_level=panel_data.get("education_level"),
                occupation=panel_data.get("occupation"),
                monthly_personal_income=panel_data.get("monthly_personal_income"),
                monthly_household_income=panel_data.get("monthly_household_income"),
                phone_brand=panel_data.get("phone_brand"),
                phone_model=panel_data.get("phone_model"),
                car_ownership=panel_data.get("car_ownership"),
                car_manufacturer=panel_data.get("car_manufacturer"),
                car_model=panel_data.get("car_model"),
                owned_electronics=panel_data.get("owned_electronics"),
                smoking_experience=panel_data.get("smoking_experience"),
                smoking_brand=panel_data.get("smoking_brand"),
                e_cig_heated_brand=panel_data.get("e_cig_heated_brand"),
                e_cig_liquid_brand=panel_data.get("e_cig_liquid_brand"),
                drinking_experience=panel_data.get("drinking_experience"),
                panel_summary_text=panel_data.get("panel_summary_text"),
            )
        )
    
    print(f"✅ 최종 검색 결과: {len(results)}개 패널 반환 (요청: {top_k}개)")
    
    # 전체 검색 시간 요약 출력
    total_search_time = llm_analysis_time + structured_time + unstructured_time + intersection_time + panel_data_fetch_time
    print(f"\n" + "=" * 80)
    print(f"📊 전체 검색 시간 요약")
    print(f"=" * 80)
    print(f"  - LLM 질의 분석: {llm_analysis_time:.2f}초")
    if structured_time > 0:
        print(f"  - 정형 검색: {structured_time:.2f}초 ({len(structured_panel_ids)}개 패널)")
    if unstructured_time > 0:
        print(f"  - 비정형 검색: {unstructured_time:.2f}초 ({len(unstructured_panel_ids)}개 패널)")
    if intersection_time > 0:
        print(f"  - 교집합 계산: {intersection_time:.2f}초")
    if panel_data_fetch_time > 0:
        print(f"  - 패널 데이터 조회: {panel_data_fetch_time:.2f}초 ({len(limited_panel_ids)}개 패널)")
    print(f"  - 총 검색 시간: {total_search_time:.2f}초")
    
    return SearchResponse(results=results, analysis=analysis_info)
