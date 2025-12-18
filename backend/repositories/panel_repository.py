"""
Panel Repository (v2)

정형 데이터 검색을 위한 Repository
- label.json 기반 필터 매핑
- SQL 쿼리로 PostgreSQL 검색
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.database import AsyncSessionLocal
from backend.repositories.value_normalizer import (
    normalize_value_with_synonyms,
    normalize_mapped_values,
    find_similar_value
)


# ===== 주의: label.json의 모든 키는 DB 컬럼명과 동일합니다 =====
# label.json 키 = DB 컬럼명 (추가 매핑 불필요)
# 예: label.json의 "age" → DB의 "age" 컬럼
#     label.json의 "region_city" → DB의 "region_city" 컬럼

# ===== label.json 카테고리 → DB 컬럼 매핑 (한글 카테고리 → DB 컬럼명) =====
LABEL_TO_DB_COLUMN = {
    "성별": "gender",
    "나이": "age",
    "지역": "region_city",
    "지역(시)": "region_city",
    "지역(구)": "region_gu",
    "결혼 여부": "marital_status",
    "결혼유무": "marital_status",
    "자녀수": "children_count",
    "가족수": "family_size",
    "최종학력": "education_level",
    "직업": "occupation",
    "월평균개인소득": "monthly_personal_income",
    "월평균가구소득": "monthly_household_income",
    "보유 휴대폰 브랜드": "phone_brand",
    "보유 휴대폰 모델명": "phone_model",
    "차량 보유 여부": "car_ownership",
    "보유 차량 제조사": "car_manufacturer",
    "보유 차량 모델": "car_model",
    "보유 전자 제품": "owned_electronics",
    "흡연경험": "smoking_experience",
    "흡연경험브랜드": "smoking_brand",
    "궐련 / 가열형 전자담배 흡연 경험 브랜드": "e_cig_heated_brand",
    "액상형 전자담배 흡연경험 브랜드": "e_cig_liquid_brand",
    "음주 경험": "drinking_experience",
}


def load_label_data() -> Dict[str, Any]:
    """label.json 파일 로드"""
    label_path = Path(__file__).resolve().parents[2] / "backend" / "data" / "label.json"
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ label.json 로드 실패: {e}")
        return {}


def normalize_gender_value(value: str) -> str:
    """성별 값을 DB 형식으로 정규화"""
    if "남" in str(value):
        return "male"
    elif "여" in str(value):
        return "female"
    return str(value)


def normalize_boolean_value(value: str, true_keywords: List[str], false_keywords: List[str]) -> Optional[bool]:
    """불린 값 정규화"""
    value_str = str(value).lower()
    if any(kw in value_str for kw in true_keywords):
        return True
    elif any(kw in value_str for kw in false_keywords):
        return False
    return None


class PanelRepository:
    """패널 정형 데이터 Repository"""
    
    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        self.session = session
        self._label_data = None
    
    async def _get_session(self) -> AsyncSession:
        if self.session:
            return self.session
        return AsyncSessionLocal()
    
    def _get_label_data(self) -> Dict[str, Any]:
        """label.json 데이터 캐싱"""
        if self._label_data is None:
            self._label_data = load_label_data()
        return self._label_data
    
    
    def map_label_filters_to_db_filters(self, label_filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """label.json 기반 필터를 DB 쿼리 필터로 변환
        
        Args:
            label_filters: [
                {"category": "region_city", "mapped_values": ["서울", "강남구"], "metadata": {...}},  # 영어 카테고리
                {"category": "age", "mapped_values": [20, 21, ..., 29], "metadata": {...}}  # 또는 한글 카테고리
            ] 형태의 딕셔너리 리스트
            metadata (선택적): {
                "confidence": "high" | "medium" | "low",
                "fuzzy_match": boolean,
                "intent": "positive" | "negative" | "neutral",
                "search_hints": {"exclude_patterns": [], "include_patterns": []}
            }
        
        Returns:
            DB 쿼리에 사용할 필터 딕셔너리
        """
        db_filters: Dict[str, Any] = {}
        label_data = self._get_label_data()
        
        # label_filters는 리스트 형태로만 받음
        for filter_item in label_filters:
            if not isinstance(filter_item, dict):
                continue
                
            category = filter_item.get("category")
            mapped_values = filter_item.get("mapped_values", [])
            metadata = filter_item.get("metadata", {})  # 메타데이터 추출
            
            if not category or not mapped_values:
                continue
            
            # 메타데이터에서 정보 추출
            confidence = metadata.get("confidence", "high")
            fuzzy_match = metadata.get("fuzzy_match", False)
            intent = metadata.get("intent", "neutral")
            search_hints = metadata.get("search_hints", {})
            
            # category는 label.json의 키 또는 한글 카테고리명
            # label.json의 키는 이미 DB 컬럼명과 동일하므로 직접 사용
            
            # DB 컬럼명 목록 (panels 테이블의 실제 컬럼명)
            db_columns = [
                "gender", "age", "region_city", "region_gu", "marital_status",
                "children_count", "family_size", "education_level", "occupation",
                "monthly_personal_income", "monthly_household_income",
                "phone_brand", "phone_model", "car_ownership", "car_manufacturer", "car_model",
                "owned_electronics", "smoking_experience", "smoking_brand",
                "e_cig_heated_brand", "e_cig_liquid_brand", "drinking_experience"
            ]
            
            # 우선순위:
            # 1. category가 이미 DB 컬럼명인 경우 (LLM이 label.json 키를 출력한 경우)
            # 2. 특수 케이스 매핑 (job → occupation)
            # 3. 한글 카테고리 → DB 컬럼명 변환 (하위 호환성)
            
            if category in db_columns:
                # label.json 키 = DB 컬럼명 (대부분의 경우)
                db_column = category
            elif category == "job":
                # 특수 케이스: "job"은 "occupation"으로 매핑
                db_column = "occupation"
            else:
                # 한글 카테고리 → DB 컬럼명 변환 (하위 호환성)
                db_column = LABEL_TO_DB_COLUMN.get(category)
                if not db_column:
                    # 알 수 없는 카테고리는 건너뜀
                    continue
            
            # mapped_values 정규화 및 오탈자 보정
            # 나이는 숫자 값이므로 정규화를 거치지 않고 그대로 사용
            if category == "age" or category == "나이":
                # 나이 값은 숫자로 유지 (정규화 없이)
                self._add_filter_for_category(db_filters, db_column, category, mapped_values, label_data)
            else:
                # 다른 카테고리는 정규화 적용 (영어 카테고리 사용)
                normalized_values = normalize_mapped_values(mapped_values, category, label_data)
                if normalized_values:
                    self._add_filter_for_category(db_filters, db_column, category, normalized_values, label_data)
        
        return db_filters
    
    def _get_column_type(self, db_column: str) -> str:
        """DB 컬럼의 타입 반환 (자동 필터링을 위해)
        
        Returns:
            "varchar" | "integer" | "boolean" | "array" | "unknown"
        """
        # 배열 타입 컬럼
        array_columns = [
            "owned_electronics", "smoking_experience", "smoking_brand",
            "e_cig_heated_brand", "e_cig_liquid_brand", "drinking_experience"
        ]
        if db_column in array_columns:
            return "array"
        
        # BOOLEAN 타입 컬럼
        boolean_columns = ["car_ownership"]
        if db_column in boolean_columns:
            return "boolean"
        
        # INTEGER 타입 컬럼
        integer_columns = [
            "age", "children_count", "family_size",
            "monthly_personal_income", "monthly_household_income"
        ]
        if db_column in integer_columns:
            return "integer"
        
        # VARCHAR 타입 컬럼 (기본값)
        return "varchar"
    
    def _add_filter_for_category(
        self, 
        db_filters: Dict[str, Any], 
        db_column: str, 
        category: str, 
        mapped_values: List[Any],
        label_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """카테고리별로 DB 필터 추가 (자동화된 로직)
        
        컬럼 타입에 따라 자동으로 적절한 필터 생성:
        - VARCHAR: 정확 매칭 (단일 값) 또는 IN (여러 값)
        - INTEGER: 정확 매칭, 범위, 또는 IN
        - BOOLEAN: 정확 매칭
        - ARRAY: 배열 교집합 또는 특정 값 포함
        """
        if not mapped_values:
            return
        
        column_type = self._get_column_type(db_column)
        
        # 메타데이터 활용 (의도 기반 필터링)
        if metadata:
            intent = metadata.get("intent", "neutral")
            # 부정 의도인 경우 특별 처리 (예: "흡연 안 하는")
            if intent == "negative" and column_type == "array":
                if db_column == "smoking_experience":
                    db_filters["smoking_experience_has"] = False
                    return
                elif db_column == "drinking_experience":
                    db_filters["drinking_experience_has"] = False
                    return
        
        # ===== 자동화된 필터 생성 로직 =====
        
        # 1. 배열 타입 컬럼 처리
        if column_type == "array":
            # 특수 케이스: "담배를 피워본 적이 없다" 같은 부정 표현 처리
            if db_column == "smoking_experience":
                non_smoker_keywords = ["피워본 적이 없다", "피운 적 없", "흡연 안", "담배를 피워본 적이 없다"]
                if any(kw in str(v) for v in mapped_values for kw in non_smoker_keywords):
                    db_filters["smoking_experience_has"] = False
                    return
            elif db_column == "drinking_experience":
                non_drinker_keywords = ["마시지 않음", "안 마심", "음주 안", "최근 1년 이내 술을 마시지 않음"]
                if any(kw in str(v) for v in mapped_values for kw in non_drinker_keywords):
                    db_filters["drinking_experience_has"] = False
                    return
            
            # 일반 배열 필터: 배열 교집합
            filter_key = f"{db_column}_in"
            db_filters[filter_key] = [str(v) for v in mapped_values]
            return
        
        # 2. BOOLEAN 타입 컬럼 처리
        if column_type == "boolean":
            # 값 정규화
            val_str = str(mapped_values[0]).strip().lower()
            if val_str in ["true", "1", "yes", "있", "있다", "있음", "보유", "소유"]:
                db_filters[db_column] = True
            elif val_str in ["false", "0", "no", "없", "없다", "없음", "미보유"]:
                db_filters[db_column] = False
            return
        
        # 3. INTEGER 타입 컬럼 처리
        if column_type == "integer":
            # 숫자 값 추출
            numeric_values = []
            for v in mapped_values:
                if isinstance(v, (int, float)):
                    numeric_values.append(int(v))
                elif isinstance(v, str):
                    # 문자열에서 숫자 추출
                    import re
                    num_match = re.search(r'\d+', v)
                    if num_match:
                        numeric_values.append(int(num_match.group()))
            
            if numeric_values:
                if db_column == "age":
                    # 나이: 범위가 5 이상이면 범위로, 아니면 IN
                    age_min = min(numeric_values)
                    age_max = max(numeric_values)
                    if age_max - age_min >= 5:
                        db_filters["age_min"] = age_min
                        db_filters["age_max"] = age_max
                    elif len(numeric_values) == 1:
                        db_filters["age_in"] = numeric_values
                    else:
                        db_filters["age_in"] = numeric_values
                elif db_column in ["children_count", "family_size"]:
                    # 자녀수/가족수: 단일 값이면 정확 매칭, 여러 값이면 최소값
                    if len(numeric_values) == 1:
                        db_filters[db_column] = numeric_values[0]
                    else:
                        db_filters[f"{db_column}_min"] = min(numeric_values)
                else:
                    # 소득 등: 최소값 기준
                    db_filters[f"{db_column}_min"] = min(numeric_values)
            return
        
        # 4. VARCHAR 타입 컬럼 처리 (기본)
        # 특수 케이스: 성별, 결혼 여부 등은 정규화 필요
        if db_column == "gender":
            for val in mapped_values:
                val_str = str(val).strip()
                normalized = normalize_value_with_synonyms(val_str, "성별")
                if normalized:
                    db_filters["gender"] = normalized
                    return
                elif "남" in val_str or val_str == "남성":
                    db_filters["gender"] = "남성"
                    return
                elif "여" in val_str or val_str == "여성":
                    db_filters["gender"] = "여성"
                    return
            return
        
        if db_column == "marital_status":
            for val in mapped_values:
                val_str = str(val).strip()
                normalized = normalize_value_with_synonyms(val_str, "결혼 여부")
                if normalized:
                    db_filters["marital_status"] = normalized
                    return
                elif any(kw in val_str for kw in ["기혼", "결혼", "배우자"]):
                    db_filters["marital_status"] = "기혼"
                    return
                elif any(kw in val_str for kw in ["미혼", "싱글", "무배우자"]):
                    db_filters["marital_status"] = "미혼"
                    return
                elif any(kw in val_str for kw in ["기타"]):
                    db_filters["marital_status"] = "기타"
                    return
            return
        
        if db_column == "occupation":
            # 직업: label.json의 job 리스트와 정확히 매칭
            job_list = label_data.get("job", [])
            normalized_occupations = []
            for val in mapped_values:
                val_str = str(val).strip()
                if val_str in job_list:
                    normalized_occupations.append(val_str)
                else:
                    similar = find_similar_value(val_str, job_list, threshold=0.9)
                    if similar:
                        normalized_occupations.append(similar[0])
                    else:
                        normalized_occupations.append(val_str)
            
            if normalized_occupations:
                if len(normalized_occupations) == 1:
                    db_filters["occupation_in"] = normalized_occupations
                else:
                    db_filters["occupation_in"] = normalized_occupations
            return
        
        if db_column == "education_level":
            # 학력: label.json의 education_level 리스트와 정확히 매칭
            education_list = label_data.get("education_level", [])
            val_str = str(mapped_values[0]).strip()
            if val_str in education_list:
                db_filters["education_level"] = val_str
            else:
                similar = find_similar_value(val_str, education_list, threshold=0.8)
                if similar:
                    db_filters["education_level"] = similar[0]
                else:
                    db_filters["education_level"] = val_str
            return
        
        # 지역 처리 (특수 케이스)
        if db_column in ["region_city", "region_gu"]:
            region_values = [str(v).strip() for v in mapped_values]
            if region_values:
                filter_key = f"{db_column}_in" if len(region_values) > 1 else db_column
                if filter_key.endswith("_in"):
                    if filter_key not in db_filters:
                        db_filters[filter_key] = []
                    db_filters[filter_key].extend(region_values)
                    db_filters[filter_key] = list(set(db_filters[filter_key]))
                else:
                    db_filters[filter_key] = region_values[0]
            return
        
        # 일반 VARCHAR 컬럼: 단일 값이면 정확 매칭, 여러 값이면 IN
        # (phone_brand, phone_model, car_model, car_manufacturer 등)
        if len(mapped_values) == 1:
            # 단일 값: 정규화 시도 (휴대폰 브랜드 등)
            val_str = str(mapped_values[0]).strip()
            if db_column == "phone_brand":
                normalized = normalize_value_with_synonyms(val_str, "보유 휴대폰 브랜드")
                if normalized:
                    db_filters[db_column] = normalized
                else:
                    db_filters[db_column] = val_str
            else:
                db_filters[db_column] = val_str
        else:
            # 여러 값: IN 절 사용
            normalized = []
            for val in mapped_values:
                val_str = str(val).strip()
                if db_column == "phone_brand":
                    synonym = normalize_value_with_synonyms(val_str, "보유 휴대폰 브랜드")
                    if synonym:
                        normalized.append(synonym)
                    else:
                        normalized.append(val_str)
                else:
                    normalized.append(val_str)
            db_filters[f"{db_column}_in"] = normalized

    async def get_panel_ids_by_filters(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        label_filters: Optional[List[Dict[str, Any]]] = None,
        session: Optional[AsyncSession] = None
    ) -> List[str]:
        """정형 필터로 패널 ID만 조회 (성능 최적화)
        
        Args:
            filters: 직접 전달된 DB 필터
            limit: 결과 개수 제한
            label_filters: label.json 기반 필터 리스트
            session: DB 세션 (없으면 새로 생성)
        
        Returns:
            패널 ID 리스트만 반환 (객체 변환 비용 없음)
        """
        db_session = session if session else await self._get_session()
        close_session = session is None
        
        try:
            # label_filters가 있으면 우선 사용
            db_filters: Dict[str, Any] = {}
            if label_filters:
                db_filters = self.map_label_filters_to_db_filters(label_filters)
            
            # 기존 filters가 있으면 병합
            if filters:
                db_filters.update(filters)
            
            # db_filters가 비어 있으면 빈 리스트 반환
            if not db_filters:
                if filters is None and label_filters is None:
                    # 전체 조회는 ID만 조회해도 빠름
                    pass
                else:
                    return []
            
            # SQL WHERE 조건 생성 (기존 로직 재사용)
            where_conditions = []
            params: Dict[str, Any] = {}
            
            # 나이 범위
            if db_filters.get("age_min") is not None:
                where_conditions.append("age >= :age_min")
                params["age_min"] = db_filters["age_min"]
            
            if db_filters.get("age_max") is not None:
                where_conditions.append("age <= :age_max")
                params["age_max"] = db_filters["age_max"]
            
            if db_filters.get("age_in"):
                placeholders = [f":age_val_{i}" for i in range(len(db_filters["age_in"]))]
                where_conditions.append(f"age IN ({','.join(placeholders)})")
                for i, age_val in enumerate(db_filters["age_in"]):
                    params[f"age_val_{i}"] = age_val
            
            # 성별
            if db_filters.get("gender"):
                where_conditions.append("gender = :gender")
                params["gender"] = db_filters["gender"]
            
            # 지역
            if db_filters.get("region_city"):
                where_conditions.append("region_city = :region_city")
                params["region_city"] = db_filters["region_city"]
            
            if db_filters.get("region_city_in"):
                placeholders = [f":region_city_{i}" for i in range(len(db_filters["region_city_in"]))]
                where_conditions.append(f"region_city IN ({','.join(placeholders)})")
                for i, city in enumerate(db_filters["region_city_in"]):
                    params[f"region_city_{i}"] = city
            
            if db_filters.get("region_gu"):
                where_conditions.append("region_gu = :region_gu")
                params["region_gu"] = db_filters["region_gu"]
            
            if db_filters.get("region_gu_in"):
                placeholders = [f":region_gu_{i}" for i in range(len(db_filters["region_gu_in"]))]
                where_conditions.append(f"region_gu IN ({','.join(placeholders)})")
                for i, gu in enumerate(db_filters["region_gu_in"]):
                    params[f"region_gu_{i}"] = gu
            
            # 결혼 여부
            if db_filters.get("marital_status"):
                where_conditions.append("marital_status = :marital_status")
                params["marital_status"] = db_filters["marital_status"]
            
            # 자녀수
            if db_filters.get("children_count_min") is not None:
                where_conditions.append("children_count >= :children_count_min")
                params["children_count_min"] = db_filters["children_count_min"]
            
            if db_filters.get("children_count") is not None:
                where_conditions.append("children_count = :children_count")
                params["children_count"] = db_filters["children_count"]
            
            # 가족수
            if db_filters.get("family_size_min") is not None:
                where_conditions.append("family_size >= :family_size_min")
                params["family_size_min"] = db_filters["family_size_min"]
            
            # 차량 보유 여부
            if db_filters.get("car_ownership") is not None:
                where_conditions.append("car_ownership = :car_ownership")
                params["car_ownership"] = db_filters["car_ownership"]
            
            # 전자 제품 (배열 교집합)
            if db_filters.get("owned_electronics_in"):
                where_conditions.append("owned_electronics && :owned_electronics_array")
                params["owned_electronics_array"] = db_filters["owned_electronics_in"]
            
            # 흡연경험 (배열 교집합)
            if db_filters.get("smoking_experience_in"):
                where_conditions.append(
                    "smoking_experience IS NOT NULL "
                    "AND array_length(smoking_experience, 1) > 0 "
                    "AND NOT ('담배를 피워본 적이 없다' = ANY(smoking_experience)) "
                    "AND smoking_experience && :smoking_experience_array"
                )
                params["smoking_experience_array"] = db_filters["smoking_experience_in"]
            
            if db_filters.get("smoking_experience_has") is False:
                where_conditions.append("'담배를 피워본 적이 없다' = ANY(smoking_experience)")
            
            # 음주 경험
            if db_filters.get("drinking_experience_in"):
                where_conditions.append("drinking_experience && :drinking_experience_array")
                params["drinking_experience_array"] = db_filters["drinking_experience_in"]
            
            if db_filters.get("drinking_experience_has") is False:
                where_conditions.append("'최근 1년 이내 술을 마시지 않음' = ANY(drinking_experience)")
            
            # 학력
            if db_filters.get("education_level"):
                where_conditions.append("education_level = :education_level")
                params["education_level"] = db_filters["education_level"]
            elif db_filters.get("education_level_in"):
                placeholders = [f":education_{i}" for i in range(len(db_filters["education_level_in"]))]
                where_conditions.append(f"education_level IN ({','.join(placeholders)})")
                for i, edu in enumerate(db_filters["education_level_in"]):
                    params[f"education_{i}"] = edu
            
            # 직업
            if db_filters.get("occupation"):
                where_conditions.append("occupation = :occupation")
                params["occupation"] = db_filters["occupation"]
            elif db_filters.get("occupation_in"):
                if len(db_filters["occupation_in"]) == 1:
                    where_conditions.append("occupation = :occupation")
                    params["occupation"] = db_filters["occupation_in"][0]
                else:
                    placeholders = [f":occupation_{i}" for i in range(len(db_filters["occupation_in"]))]
                    where_conditions.append(f"occupation IN ({','.join(placeholders)})")
                    for i, occ in enumerate(db_filters["occupation_in"]):
                        params[f"occupation_{i}"] = occ
            
            # 소득
            if db_filters.get("monthly_personal_income_min") is not None:
                where_conditions.append("monthly_personal_income >= :monthly_personal_income_min")
                params["monthly_personal_income_min"] = db_filters["monthly_personal_income_min"]
            
            if db_filters.get("monthly_household_income_min") is not None:
                where_conditions.append("monthly_household_income >= :monthly_household_income_min")
                params["monthly_household_income_min"] = db_filters["monthly_household_income_min"]
            
            # 휴대폰 브랜드
            if db_filters.get("phone_brand"):
                where_conditions.append("phone_brand = :phone_brand")
                params["phone_brand"] = db_filters["phone_brand"]
            elif db_filters.get("phone_brand_in"):
                if len(db_filters["phone_brand_in"]) == 1:
                    where_conditions.append("phone_brand LIKE :phone_brand_like")
                    params["phone_brand_like"] = f"%{db_filters['phone_brand_in'][0]}%"
                else:
                    placeholders = [f":phone_brand_{i}" for i in range(len(db_filters["phone_brand_in"]))]
                    where_conditions.append(f"phone_brand IN ({','.join(placeholders)})")
                    for i, brand in enumerate(db_filters["phone_brand_in"]):
                        params[f"phone_brand_{i}"] = brand
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            limit_clause = "LIMIT :limit" if limit is not None else ""
            
            # ID만 조회 (훨씬 빠름)
            sql_query = text(f"""
                SELECT panel_id
                FROM panels
                WHERE {where_clause}
                {limit_clause}
            """)
            
            if limit is not None:
                params["limit"] = limit
            
            result = await db_session.execute(sql_query, params)
            rows = result.fetchall()
            
            # ID 리스트만 반환
            return [row[0] for row in rows]
        finally:
            if close_session:
                await db_session.close()
    
    async def filter_by_structured_filters(
        self, 
        filters: Optional[Dict[str, Any]] = None, 
        limit: Optional[int] = None, 
        query: Optional[str] = None,
        label_filters: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """정형 필터로 패널 검색 (label.json 기반 SQL)
        
        Args:
            filters: 직접 전달된 DB 필터 (기존 방식)
            limit: 결과 개수 제한 (None이면 제한 없음)
            query: 자연어 질의 (참고용, 현재 미사용)
            label_filters: label.json 기반 필터 리스트 (새로운 방식)
                예: [{"category": "지역", "mapped_values": ["서울"]}, ...]
        """
        session = await self._get_session()
        
        # label_filters가 있으면 우선 사용
        db_filters: Dict[str, Any] = {}
        if label_filters:
            db_filters = self.map_label_filters_to_db_filters(label_filters)
            print(f"📊 label.json 기반 필터 변환: {db_filters}")
        
        # 기존 filters가 있으면 병합
        if filters:
            db_filters.update(filters)
        
        # db_filters가 비어 있으면 전체 패널 조회 (조건 없음)
        # 단, 명시적으로 filters=None이고 label_filters=None인 경우만 허용
        if not db_filters:
            if filters is None and label_filters is None:
                print("ℹ️ 조건 없이 전체 패널 조회 모드")
            else:
                print("⚠️ label.json 기반 필터가 없습니다. LLM 분석 결과가 없으므로 정형 검색을 중단합니다.")
                return []
        
        # SQL WHERE 조건 생성
        where_conditions = []
        params: Dict[str, Any] = {}
        
        # 나이 범위
        if db_filters.get("age_min") is not None:
            where_conditions.append("age >= :age_min")
            params["age_min"] = db_filters["age_min"]
        
        if db_filters.get("age_max") is not None:
            where_conditions.append("age <= :age_max")
            params["age_max"] = db_filters["age_max"]
        
        # 나이 특정 값들
        if db_filters.get("age_in"):
            placeholders = [f":age_val_{i}" for i in range(len(db_filters["age_in"]))]
            where_conditions.append(f"age IN ({','.join(placeholders)})")
            for i, age_val in enumerate(db_filters["age_in"]):
                params[f"age_val_{i}"] = age_val
        
        # 성별
        if db_filters.get("gender"):
            where_conditions.append("gender = :gender")
            params["gender"] = db_filters["gender"]
        
        # 지역 (정확히 일치)
        if db_filters.get("region_city"):
            where_conditions.append("region_city = :region_city")
            params["region_city"] = db_filters["region_city"]
        
        # 지역 (여러 도시 중 하나 - OR 조건)
        if db_filters.get("region_city_in"):
            placeholders = [f":region_city_{i}" for i in range(len(db_filters["region_city_in"]))]
            where_conditions.append(f"region_city IN ({','.join(placeholders)})")
            for i, city in enumerate(db_filters["region_city_in"]):
                params[f"region_city_{i}"] = city
        
        # 지역 구/시 (구체적인 지역명)
        if db_filters.get("region_gu"):
            where_conditions.append("region_gu = :region_gu")
            params["region_gu"] = db_filters["region_gu"]

        if db_filters.get("region_gu_in"):
            placeholders = [f":region_gu_{i}" for i in range(len(db_filters["region_gu_in"]))]
            where_conditions.append(f"region_gu IN ({','.join(placeholders)})")
            for i, gu in enumerate(db_filters["region_gu_in"]):
                params[f"region_gu_{i}"] = gu
        
        # 결혼 여부
        if db_filters.get("marital_status"):
            where_conditions.append("marital_status = :marital_status")
            params["marital_status"] = db_filters["marital_status"]
        
        # 자녀수
        if db_filters.get("children_count_min") is not None:
            where_conditions.append("children_count >= :children_count_min")
            params["children_count_min"] = db_filters["children_count_min"]
        
        if db_filters.get("children_count") is not None:
            where_conditions.append("children_count = :children_count")
            params["children_count"] = db_filters["children_count"]
        
        # 가족수
        if db_filters.get("family_size_min") is not None:
            where_conditions.append("family_size >= :family_size_min")
            params["family_size_min"] = db_filters["family_size_min"]
        
        # 차량 보유 여부
        if db_filters.get("car_ownership") is not None:
            where_conditions.append("car_ownership = :car_ownership")
            params["car_ownership"] = db_filters["car_ownership"]
        
        # 차량 제조사 (VARCHAR이므로 IN 사용)
        if db_filters.get("car_manufacturer_in"):
            placeholders = [f":car_mfg_{i}" for i in range(len(db_filters["car_manufacturer_in"]))]
            where_conditions.append(f"car_manufacturer IN ({','.join(placeholders)})")
            for i, mfg in enumerate(db_filters["car_manufacturer_in"]):
                params[f"car_mfg_{i}"] = mfg
        
        # 전자 제품 (배열 교집합 매칭 - asyncpg 배열 바인딩)
        if db_filters.get("owned_electronics_in"):
            # PostgreSQL 배열 교집합 연산자 사용 (하나 이상 일치)
            # asyncpg는 배열을 리스트로 직접 바인딩 가능
            where_conditions.append("owned_electronics && :owned_electronics_array")
            params["owned_electronics_array"] = db_filters["owned_electronics_in"]
        
        # 흡연경험 (배열 교집합 매칭)
        if db_filters.get("smoking_experience_in"):
            # Null 값과 빈 배열 제외, 그리고 "담배를 피워본 적이 없다" 제외
            where_conditions.append(
                "smoking_experience IS NOT NULL "
                "AND array_length(smoking_experience, 1) > 0 "
                "AND NOT ('담배를 피워본 적이 없다' = ANY(smoking_experience)) "
                "AND smoking_experience && :smoking_experience_array"
            )
            params["smoking_experience_array"] = db_filters["smoking_experience_in"]
        
        if db_filters.get("smoking_experience_has") is False:
            # 비흡연자: "담배를 피워본 적이 없다"가 포함된 경우만
            where_conditions.append("'담배를 피워본 적이 없다' = ANY(smoking_experience)")
        
        # 음주 경험 (배열 교집합 매칭)
        if db_filters.get("drinking_experience_in"):
            where_conditions.append("drinking_experience && :drinking_experience_array")
            params["drinking_experience_array"] = db_filters["drinking_experience_in"]
        
        if db_filters.get("drinking_experience_has") is False:
            where_conditions.append("'최근 1년 이내 술을 마시지 않음' = ANY(drinking_experience)")
        
        # 학력 (단일 값 또는 배열 모두 지원)
        if db_filters.get("education_level"):
            # 단일 값
            where_conditions.append("education_level = :education_level")
            params["education_level"] = db_filters["education_level"]
        elif db_filters.get("education_level_in"):
            # 여러 값
            placeholders = [f":education_{i}" for i in range(len(db_filters["education_level_in"]))]
            where_conditions.append(f"education_level IN ({','.join(placeholders)})")
            for i, edu in enumerate(db_filters["education_level_in"]):
                params[f"education_{i}"] = edu
        
        # 직업 (단일 값 또는 배열 모두 지원)
        if db_filters.get("occupation"):
            # 단일 값: 정확 매칭
            where_conditions.append("occupation = :occupation")
            params["occupation"] = db_filters["occupation"]
        elif db_filters.get("occupation_in"):
            # 여러 값: IN 절 사용
            if len(db_filters["occupation_in"]) == 1:
                where_conditions.append("occupation = :occupation")
                params["occupation"] = db_filters["occupation_in"][0]
            else:
                placeholders = [f":occupation_{i}" for i in range(len(db_filters["occupation_in"]))]
                where_conditions.append(f"occupation IN ({','.join(placeholders)})")
                for i, occ in enumerate(db_filters["occupation_in"]):
                    params[f"occupation_{i}"] = occ
        elif db_filters.get("occupation_like"):
            # 부분 매칭 (폴백 - 정확한 매칭이 없을 때만 사용)
            occupation_keywords = db_filters["occupation_like"].split() if isinstance(db_filters["occupation_like"], str) else [str(db_filters["occupation_like"])]
            if len(occupation_keywords) == 1:
                where_conditions.append("occupation LIKE :occupation_like")
                params["occupation_like"] = f"%{occupation_keywords[0]}%"
            else:
                # 여러 키워드 중 하나라도 일치
                or_conditions = []
                for i, keyword in enumerate(occupation_keywords):
                    or_conditions.append(f"occupation LIKE :occ_{i}")
                    params[f"occ_{i}"] = f"%{keyword}%"
                where_conditions.append(f"({' OR '.join(or_conditions)})")
        
        # 소득
        if db_filters.get("monthly_personal_income_min") is not None:
            where_conditions.append("monthly_personal_income >= :monthly_personal_income_min")
            params["monthly_personal_income_min"] = db_filters["monthly_personal_income_min"]
        
        if db_filters.get("monthly_household_income_min") is not None:
            where_conditions.append("monthly_household_income >= :monthly_household_income_min")
            params["monthly_household_income_min"] = db_filters["monthly_household_income_min"]
        
        # 휴대폰 브랜드 (VARCHAR이므로 정확 매칭 또는 IN 사용)
        if db_filters.get("phone_brand"):
            # 단일 값
            where_conditions.append("phone_brand = :phone_brand")
            params["phone_brand"] = db_filters["phone_brand"]
        elif db_filters.get("phone_brand_in"):
            # 여러 값이 있으면 IN, 하나면 LIKE로 부분 일치
            if len(db_filters["phone_brand_in"]) == 1:
                where_conditions.append("phone_brand LIKE :phone_brand_like")
                params["phone_brand_like"] = f"%{db_filters['phone_brand_in'][0]}%"
            else:
                placeholders = [f":phone_brand_{i}" for i in range(len(db_filters["phone_brand_in"]))]
                where_conditions.append(f"phone_brand IN ({','.join(placeholders)})")
                for i, brand in enumerate(db_filters["phone_brand_in"]):
                    params[f"phone_brand_{i}"] = brand
        
        # 휴대폰 모델
        if db_filters.get("phone_model"):
            # 단일 값
            where_conditions.append("phone_model = :phone_model")
            params["phone_model"] = db_filters["phone_model"]
        elif db_filters.get("phone_model_in"):
            # 여러 값
            placeholders = [f":phone_model_{i}" for i in range(len(db_filters["phone_model_in"]))]
            where_conditions.append(f"phone_model IN ({','.join(placeholders)})")
            for i, model in enumerate(db_filters["phone_model_in"]):
                params[f"phone_model_{i}"] = model
        
        # 차량 모델 (VARCHAR이므로 정확 매칭 또는 IN 사용)
        if db_filters.get("car_model"):
            # 단일 값
            where_conditions.append("car_model = :car_model")
            params["car_model"] = db_filters["car_model"]
        elif db_filters.get("car_model_in"):
            # 여러 값
            placeholders = [f":car_model_{i}" for i in range(len(db_filters["car_model_in"]))]
            where_conditions.append(f"car_model IN ({','.join(placeholders)})")
            for i, model in enumerate(db_filters["car_model_in"]):
                params[f"car_model_{i}"] = model
        
        # 차량 제조사도 동일하게 처리
        if db_filters.get("car_manufacturer"):
            # 단일 값
            where_conditions.append("car_manufacturer = :car_manufacturer")
            params["car_manufacturer"] = db_filters["car_manufacturer"]
        
        # 흡연 브랜드 등 (배열 필드)
        if db_filters.get("smoking_brand_in"):
            where_conditions.append("smoking_brand && :smoking_brand_array")
            params["smoking_brand_array"] = db_filters["smoking_brand_in"]
        
        if db_filters.get("e_cig_heated_brand_in"):
            where_conditions.append("e_cig_heated_brand && :e_cig_heated_brand_array")
            params["e_cig_heated_brand_array"] = db_filters["e_cig_heated_brand_in"]
        
        if db_filters.get("e_cig_liquid_brand_in"):
            where_conditions.append("e_cig_liquid_brand && :e_cig_liquid_brand_array")
            params["e_cig_liquid_brand_array"] = db_filters["e_cig_liquid_brand_in"]
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # LIMIT 절 추가 (limit이 None이 아닐 때만)
        limit_clause = "LIMIT :limit" if limit is not None else ""
        
        # [최적화] SELECT 할 때 'panel_summary_text' 제외!
        # 목록 조회용이므로 무거운 텍스트 데이터는 뺍니다.
        # 필요하다면 상세 조회(get_panels_by_ids)에서 가져오면 됩니다.
        sql_query = text(f"""
            SELECT panel_id, gender, age, region_city, region_gu, marital_status,
                   children_count, family_size, education_level, occupation,
                   monthly_personal_income, monthly_household_income,
                   phone_brand, phone_model, car_ownership, car_manufacturer, car_model,
                   owned_electronics, smoking_experience, smoking_brand,
                   e_cig_heated_brand, e_cig_liquid_brand, drinking_experience
                   -- panel_summary_text 제거함 (속도 향상)
            FROM panels
            WHERE {where_clause}
            {limit_clause}
        """)
        
        if limit is not None:
            params["limit"] = limit
        
        result = await session.execute(sql_query, params)
        rows = result.fetchall()
        
        # 컬럼 목록에서도 panel_summary_text 제거
        columns = [
            "panel_id", "gender", "age", "region_city", "region_gu", "marital_status",
            "children_count", "family_size", "education_level", "occupation",
            "monthly_personal_income", "monthly_household_income",
            "phone_brand", "phone_model", "car_ownership", "car_manufacturer", "car_model",
            "owned_electronics", "smoking_experience", "smoking_brand",
            "e_cig_heated_brand", "e_cig_liquid_brand", "drinking_experience"
        ]
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def get_panels_by_ids(
        self,
        panel_ids: List[str],
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """패널 ID 리스트로 패널 데이터 조회 (청크 처리 및 안정성 강화)
        
        Args:
            panel_ids: 조회할 패널 ID 리스트
            session: DB 세션 (없으면 새로 생성)
        
        Returns:
            패널 데이터 리스트
        """
        if not panel_ids:
            return []
        
        db_session = session if session else await self._get_session()
        close_session = session is None
        
        try:
            # [최적화] ID가 많을 경우 Chunking (1,000개씩 끊어서 조회)
            # 파라미터 개수 제한 에러 방지 및 메모리 효율화
            CHUNK_SIZE = 1000
            all_rows = []
            
            columns = [
                "panel_id", "gender", "age", "region_city", "region_gu", "marital_status",
                "children_count", "family_size", "education_level", "occupation",
                "monthly_personal_income", "monthly_household_income",
                "phone_brand", "phone_model", "car_ownership", "car_manufacturer", "car_model",
                "owned_electronics", "smoking_experience", "smoking_brand",
                "e_cig_heated_brand", "e_cig_liquid_brand", "drinking_experience",
                "panel_summary_text"
            ]

            # 청크 단위로 루프 실행
            for i in range(0, len(panel_ids), CHUNK_SIZE):
                chunk_ids = panel_ids[i : i + CHUNK_SIZE]
                
                # [핵심 수정] 파라미터 바인딩 방식 변경 (가장 안전한 방식)
                # 딕셔너리로 파라미터를 넘기고, 쿼리문에는 :pid_0, :pid_1 형태로 직접 삽입
                
                placeholders = []
                params = {}
                for idx, pid in enumerate(chunk_ids):
                    param_key = f"pid_{i}_{idx}"  # 유니크한 파라미터 이름 생성 (청크 인덱스 포함)
                    placeholders.append(f":{param_key}")
                    params[param_key] = pid
                
                # 쿼리 문자열 조립 (f-string 사용, text() 내부에서 처리)
                sql_str = f"""
                    SELECT 
                        panel_id, gender, age, region_city, region_gu, marital_status,
                        children_count, family_size, education_level, occupation,
                        monthly_personal_income, monthly_household_income,
                        phone_brand, phone_model, car_ownership, car_manufacturer, car_model,
                        owned_electronics, smoking_experience, smoking_brand,
                        e_cig_heated_brand, e_cig_liquid_brand, drinking_experience,
                        panel_summary_text
                    FROM panels
                    WHERE panel_id IN ({','.join(placeholders)})
                """
                
                result = await db_session.execute(text(sql_str), params)
                rows = result.fetchall()
                all_rows.extend(rows)
            
            # 결과 변환
            return [dict(zip(columns, row)) for row in all_rows]
            
        except Exception as e:
            print(f"❌ 패널 상세 조회 실패: {e}")
            # 에러 발생 시 빈 리스트 반환보다는 에러를 던져서 상위에서 알게 하는 게 낫지만,
            # 현재 구조상 빈 리스트 반환이 안전할 수 있음. 로그 확인 필수!
            import traceback
            traceback.print_exc()
            return []
            
        finally:
            if close_session:
                await db_session.close()
