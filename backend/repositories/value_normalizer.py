"""
값 정규화 및 유사도 매칭 유틸리티
- 오탈자 보정
- 동의어 처리
- 유사도 기반 매칭
"""

from typing import List, Optional, Tuple
import difflib


# ===== 동의어 사전 =====
SYNONYMS = {
    "성별": {
        "남성": ["남성", "남자", "남", "남녀", "male", "Male"],
        "여성": ["여성", "여자", "여", "female", "Female"],
    },
    "결혼 여부": {
        "기혼": ["기혼", "결혼", "결혼함", "기혼자", "결혼자", "배우자 있음", "married"],
        "미혼": ["미혼", "싱글", "미혼자", "무배우자", "single", "unmarried"],
    },
    "차량 보유 여부": {
        "있다": ["있다", "있음", "보유", "보유함", "소유", "있어요", "yes", "true"],
        "없다": ["없다", "없음", "미보유", "없어요", "no", "false"],
    },
    "보유 휴대폰 브랜드": {
        "애플": ["애플", "아이폰", "iPhone", "iphone", "apple", "Apple", "아이폰 사용", "아이폰 쓰는"],
        "삼성전자": ["삼성전자", "삼성", "갤럭시", "Galaxy", "galaxy", "samsung", "Samsung", "삼성폰", "갤럭시폰"],
        "LG전자": ["LG전자", "LG", "엘지", "lg", "LG폰"],
    },
    "보유 전자 제품": {
        "노트북": ["노트북", "랩탑", "laptop", "Laptop", "노트북PC", "랩탑PC", "PC"],
        "태블릿PC": ["태블릿PC", "태블릿", "tablet", "Tablet", "패드", "아이패드", "iPad"],
        "스마트 워치": ["스마트 워치", "스마트워치", "smartwatch", "시계", "워치"],
        "TV": ["TV", "티비", "텔레비전", "텔레비", "티브이", "tv", "television"],
    },
    "흡연경험": {
        "담배를 피워본 적이 없다": ["담배를 피워본 적이 없다", "흡연 안", "담배 안", "안 피움", "비흡연", "흡연 경험 없음"],
        "일반 담배": ["일반 담배", "일반담배", "담배", "궐련", "연초"],
        "전자담배": ["전자담배", "전담", "e-cigarette", "e-cig"],
    },
    "보유 차량 제조사": {
        "기아": ["기아", "kia", "KIA", "Kia", "기아자동차"],
        "현대": ["현대", "hyundai", "Hyundai", "HYUNDAI", "현대자동차"],
        "제네시스": ["제네시스", "genesis", "Genesis", "GENESIS"],
        "KG모빌리티": ["KG모빌리티", "쌍용", "쌍용자동차", "KG", "kg모빌리티"],
        "르노삼성": ["르노삼성", "삼성", "르노", "Renault Samsung", "renault"],
        "쉐보레": ["쉐보레", "쉐비", "chevrolet", "Chevrolet", "CHEVROLET"],
        "메르세데스-벤츠": ["메르세데스-벤츠", "벤츠", "메르세데스", "Mercedes-Benz", "Benz", "benz"],
        "BMW": ["BMW", "bmw", "비엠더블유"],
        "테슬라": ["테슬라", "tesla", "Tesla", "TESLA"],
        "토요타": ["토요타", "toyota", "Toyota", "TOYOTA"],
        "렉서스": ["렉서스", "lexus", "Lexus", "LEXUS"],
        "혼다": ["혼다", "honda", "Honda", "HONDA"],
        "닛산": ["닛산", "nissan", "Nissan", "NISSAN"],
        "폭스바겐": ["폭스바겐", "volkswagen", "Volkswagen", "VW", "vw"],
        "아우디": ["아우디", "audi", "Audi", "AUDI"],
        "포르쉐": ["포르쉐", "포르셰", "porsche", "Porsche", "PORSCHE"],
    },
    "보유 차량 모델": {
        "쏘나타": ["쏘나타", "소나타", "Sonata", "sonata"],
        "아반떼": ["아반떼", "아반테", "Avante", "avante", "엘란트라", "Elantra"],
        "싼타페": ["싼타페", "산타페", "Santa Fe", "santafe", "싼타페"],
        "투싼": ["투싼", "투산", "Tucson", "tucson"],
        "그랜저": ["그랜저", "Grandeur", "grandeur"],
        "쏘렌토": ["쏘렌토", "소렌토", "Sorento", "sorento"],
        "카니발": ["카니발", "Carnival", "carnival"],
        "코나": ["코나", "Kona", "kona"],
        "니로": ["니로", "Niro", "niro"],
        "스팅어": ["스팅어", "Stinger", "stinger"],
        "셀토스": ["셀토스", "Seltos", "seltos"],
        "스포티지": ["스포티지", "Sportage", "sportage"],
    },
    "흡연경험브랜드": {
        "말보로": ["말보로", "말보루", "Marlboro", "marlboro"],
        "에쎄": ["에쎄", "esse", "Esse", "ESSE"],
        "레종": ["레종", "레죵", "Raison", "raison"],
        "던힐": ["던힐", "Dunhill", "dunhill"],
        "디스": ["디스", "This", "this"],
        "보헴": ["보헴", "Bohem", "bohem"],
    },
}


def normalize_value_with_synonyms(value: str, category: str) -> Optional[str]:
    """동의어 사전을 사용하여 값을 정규화
    
    Args:
        value: 입력 값
        category: 카테고리명
        
    Returns:
        정규화된 값 (매칭되는 표준 값) 또는 None
    """
    if category not in SYNONYMS:
        return None
    
    value_lower = str(value).lower().strip()
    
    for standard_value, synonyms in SYNONYMS[category].items():
        for synonym in synonyms:
            if synonym.lower() == value_lower or synonym.lower() in value_lower or value_lower in synonym.lower():
                return standard_value
    
    return None


def find_similar_value(input_value: str, candidate_values: List[str], threshold: float = 0.6) -> Optional[Tuple[str, float]]:
    """입력 값과 가장 유사한 후보 값 찾기 (Levenshtein 거리 기반)
    
    Args:
        input_value: 입력 값
        candidate_values: 후보 값 리스트
        threshold: 최소 유사도 (0.0 ~ 1.0)
        
    Returns:
        (가장 유사한 값, 유사도) 또는 None
    """
    if not input_value or not candidate_values:
        return None
    
    input_lower = str(input_value).lower().strip()
    
    best_match = None
    best_ratio = 0.0
    
    for candidate in candidate_values:
        candidate_str = str(candidate).lower().strip()
        
        # 정확히 일치
        if input_lower == candidate_str:
            return (str(candidate), 1.0)
        
        # 부분 일치
        if input_lower in candidate_str or candidate_str in input_lower:
            ratio = max(len(input_lower) / len(candidate_str), len(candidate_str) / len(input_lower))
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = str(candidate)
        
        # SequenceMatcher로 유사도 계산
        ratio = difflib.SequenceMatcher(None, input_lower, candidate_str).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = str(candidate)
    
    if best_ratio >= threshold and best_match:
        return (best_match, best_ratio)
    
    return None


def normalize_mapped_values(mapped_values: List[str], category: str, label_data: dict) -> List[str]:
    """mapped_values를 정규화하고 유사도 기반으로 보정
    
    Args:
        mapped_values: LLM이 추출한 값 리스트
        category: 카테고리명 (영어 또는 한글)
        label_data: label.json 데이터
        
    Returns:
        정규화 및 보정된 값 리스트
    """
    normalized = []
    
    # 카테고리명 변환 (영어 -> 동의어 사전 키)
    category_for_synonyms = category
    if category == "gender":
        category_for_synonyms = "성별"
    elif category == "marital_status":
        category_for_synonyms = "결혼 여부"
    elif category == "car_ownership":
        category_for_synonyms = "차량 보유 여부"
    elif category == "phone_brand":
        category_for_synonyms = "보유 휴대폰 브랜드"
    elif category == "owned_electronics":
        category_for_synonyms = "보유 전자 제품"
    elif category == "smoking_experience":
        category_for_synonyms = "흡연경험"
    elif category == "car_manufacturer":
        category_for_synonyms = "보유 차량 제조사"
    elif category == "car_model":
        category_for_synonyms = "보유 차량 모델"
    elif category == "smoking_brand":
        category_for_synonyms = "흡연경험브랜드"
    
    category_values = label_data.get(category, [])
    
    for val in mapped_values:
        val_str = str(val).strip()
        
        # 1. 동의어 사전으로 정규화 시도 (한글 카테고리명 사용)
        synonym_normalized = normalize_value_with_synonyms(val_str, category_for_synonyms)
        if synonym_normalized:
            normalized.append(synonym_normalized)
            continue
        
        # 2. label.json 값 중 정확히 일치하는지 확인
        if val_str in category_values:
            normalized.append(val_str)
            continue
        
        # 3. 부분 일치 확인 (대소문자 무시)
        for cat_val in category_values:
            cat_val_str = str(cat_val)
            if val_str.lower() in cat_val_str.lower() or cat_val_str.lower() in val_str.lower():
                normalized.append(cat_val_str)
                break
        else:
            # 4. 유사도 기반 매칭
            similar = find_similar_value(val_str, category_values, threshold=0.7)
            if similar:
                normalized.append(similar[0])
            else:
                # 매칭 실패해도 원본 값 유지 (부분 검색에서 사용)
                normalized.append(val_str)
    
    # 중복 제거
    return list(set(normalized))


def fuzzy_match_in_list(value: str, candidate_list: List[str]) -> Optional[str]:
    """리스트에서 퍼지 매칭으로 가장 유사한 값 찾기"""
    if not value or not candidate_list:
        return None
    
    result = find_similar_value(value, candidate_list, threshold=0.7)
    return result[0] if result else None

