"""
비교군 검색 모듈

검색된 패널과 유사하거나 대조되는 패널 그룹을 찾는 모듈
"""

from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.database import AsyncSessionLocal


class ComparisonGroupFinder:
    """비교군 검색 클래스"""
    
    async def find_comparison_groups(
        self,
        panels_data: List[Dict[str, Any]],
        session: Optional[AsyncSession] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """비교군 검색
        
        Args:
            panels_data: 검색된 패널 데이터
            session: DB 세션
            limit: 각 비교군 타입별 최대 개수
        
        Returns:
            [
                {
                    "type": "similar|contrast|complement",
                    "panel_ids": [...],
                    "similarity_score": 0.85,
                    "reason": "..."
                }
            ]
        """
        if not panels_data:
            return []
        
        close_session = False
        if session is None:
            session = AsyncSessionLocal()
            close_session = True
        
        try:
            # 검색된 패널의 특성 요약
            target_features = self._extract_features(panels_data)
            
            # 유사 그룹 검색
            similar_groups = await self._find_similar_groups(
                target_features, panels_data, session, limit
            )
            
            # 대조 그룹 검색
            contrast_groups = await self._find_contrast_groups(
                target_features, panels_data, session, limit
            )
            
            # 보완 그룹 검색 (선택적)
            complement_groups = await self._find_complement_groups(
                target_features, panels_data, session, limit
            )
            
            return similar_groups + contrast_groups + complement_groups
        finally:
            if close_session:
                await session.close()
    
    def _extract_features(self, panels_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """패널 그룹의 특성 추출"""
        if not panels_data:
            return {}
        
        # 주요 특성 추출
        features = {
            "avg_age": None,
            "gender_dist": {},
            "region_city_dist": {},
            "marital_status_dist": {},
            "avg_household_income": None,
            "car_ownership_rate": None,
        }
        
        # 나이 평균
        ages: List[int] = []
        for p in panels_data:
            age = p.get("age")
            if age is not None and isinstance(age, (int, float)):
                ages.append(int(age))
        if ages:
            features["avg_age"] = sum(ages) / len(ages)
        
        # 성별 분포
        for p in panels_data:
            gender = p.get("gender")
            if gender:
                features["gender_dist"][gender] = features["gender_dist"].get(gender, 0) + 1
        
        # 지역 분포
        for p in panels_data:
            region = p.get("region_city")
            if region:
                features["region_city_dist"][region] = features["region_city_dist"].get(region, 0) + 1
        
        # 결혼 여부 분포
        for p in panels_data:
            marital = p.get("marital_status")
            if marital:
                features["marital_status_dist"][marital] = features["marital_status_dist"].get(marital, 0) + 1
        
        # 가구 소득 평균
        incomes: List[int] = []
        for p in panels_data:
            income = p.get("monthly_household_income")
            if income is not None and isinstance(income, (int, float)):
                incomes.append(int(income))
        if incomes:
            features["avg_household_income"] = sum(incomes) / len(incomes)
        
        # 차량 소유율
        car_owners = sum(1 for p in panels_data if p.get("car_ownership") is True)
        features["car_ownership_rate"] = car_owners / len(panels_data) if panels_data else 0
        
        return features
    
    async def _find_similar_groups(
        self,
        target_features: Dict[str, Any],
        panels_data: List[Dict[str, Any]],
        session: AsyncSession,
        limit: int
    ) -> List[Dict[str, Any]]:
        """유사 그룹 검색 (비슷한 특성을 가진 패널)"""
        # 검색된 패널 ID 제외
        exclude_ids = [p.get("panel_id") for p in panels_data if p.get("panel_id")]
        
        where_conditions = []
        params: Dict[str, Any] = {}
        
        # 나이 범위 (평균 ± 5세)
        if target_features.get("avg_age"):
            where_conditions.append("age BETWEEN :age_min AND :age_max")
            params["age_min"] = int(target_features["avg_age"] - 5)
            params["age_max"] = int(target_features["avg_age"] + 5)
        
        # 성별 분포가 비슷한 경우 (주요 성별 포함)
        if target_features.get("gender_dist"):
            main_gender = max(target_features["gender_dist"].items(), key=lambda x: x[1])[0]
            where_conditions.append("gender = :main_gender")
            params["main_gender"] = main_gender
        
        # 지역이 비슷한 경우 (주요 지역 포함)
        if target_features.get("region_city_dist"):
            main_regions = sorted(
                target_features["region_city_dist"].items(),
                key=lambda x: -x[1]
            )[:3]
            region_list = [r[0] for r in main_regions]
            placeholders = [f":region_{i}" for i in range(len(region_list))]
            where_conditions.append(f"region_city IN ({','.join(placeholders)})")
            for i, region in enumerate(region_list):
                params[f"region_{i}"] = region
        
        # 소득 범위 (평균 ± 20%)
        if target_features.get("avg_household_income"):
            where_conditions.append("monthly_household_income BETWEEN :income_min AND :income_max")
            params["income_min"] = int(target_features["avg_household_income"] * 0.8)
            params["income_max"] = int(target_features["avg_household_income"] * 1.2)
        
        if not where_conditions:
            return []
        
        # 검색된 패널 제외
        if exclude_ids:
            placeholders = [f":exclude_{i}" for i in range(len(exclude_ids))]
            where_conditions.append(f"panel_id NOT IN ({','.join(placeholders)})")
            for i, pid in enumerate(exclude_ids):
                params[f"exclude_{i}"] = pid
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
            SELECT panel_id
            FROM panels
            WHERE {where_clause}
            LIMIT :limit
        """)
        params["limit"] = limit * 10  # 더 많이 가져와서 필터링
        
        result = await session.execute(query, params)
        panel_ids = [row[0] for row in result]
        
        if not panel_ids:
            return []
        
        return [{
            "type": "similar",
            "panel_ids": panel_ids[:limit],
            "similarity_score": 0.8,  # 간단한 점수
            "reason": "유사한 인구통계 및 경제력 특성을 가진 패널 그룹"
        }]
    
    async def _find_contrast_groups(
        self,
        target_features: Dict[str, Any],
        panels_data: List[Dict[str, Any]],
        session: AsyncSession,
        limit: int
    ) -> List[Dict[str, Any]]:
        """대조 그룹 검색 (반대 특성을 가진 패널)"""
        exclude_ids = [p.get("panel_id") for p in panels_data if p.get("panel_id")]
        
        where_conditions = []
        params: Dict[str, Any] = {}
        
        # 성별이 다른 경우
        if target_features.get("gender_dist"):
            main_gender = max(target_features["gender_dist"].items(), key=lambda x: x[1])[0]
            opposite_gender = "여성" if main_gender == "남성" else "남성"
            where_conditions.append("gender = :opposite_gender")
            params["opposite_gender"] = opposite_gender
        
        # 나이대가 다른 경우 (20대 vs 40대+)
        if target_features.get("avg_age"):
            if target_features["avg_age"] < 35:
                where_conditions.append("age >= 40")
            else:
                where_conditions.append("age < 30")
        
        if not where_conditions:
            return []
        
        if exclude_ids:
            placeholders = [f":exclude_{i}" for i in range(len(exclude_ids))]
            where_conditions.append(f"panel_id NOT IN ({','.join(placeholders)})")
            for i, pid in enumerate(exclude_ids):
                params[f"exclude_{i}"] = pid
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
            SELECT panel_id
            FROM panels
            WHERE {where_clause}
            LIMIT :limit
        """)
        params["limit"] = limit * 10
        
        result = await session.execute(query, params)
        panel_ids = [row[0] for row in result]
        
        if not panel_ids:
            return []
        
        return [{
            "type": "contrast",
            "panel_ids": panel_ids[:limit],
            "similarity_score": 0.3,
            "reason": "대조되는 특성을 가진 패널 그룹 (비교 분석용)"
        }]
    
    async def _find_complement_groups(
        self,
        target_features: Dict[str, Any],
        panels_data: List[Dict[str, Any]],
        session: AsyncSession,
        limit: int
    ) -> List[Dict[str, Any]]:
        """보완 그룹 검색 (추가 분석을 위한 보완 패널)"""
        # 간단히 구현: 랜덤 샘플링
        exclude_ids = [p.get("panel_id") for p in panels_data if p.get("panel_id")]
        
        where_conditions = ["1=1"]
        params: Dict[str, Any] = {}
        
        if exclude_ids:
            placeholders = [f":exclude_{i}" for i in range(len(exclude_ids))]
            where_conditions.append(f"panel_id NOT IN ({','.join(placeholders)})")
            for i, pid in enumerate(exclude_ids):
                params[f"exclude_{i}"] = pid
        
        where_clause = " AND ".join(where_conditions)
        
        query = text(f"""
            SELECT panel_id
            FROM panels
            WHERE {where_clause}
            ORDER BY random()
            LIMIT :limit
        """)
        params["limit"] = limit
        
        result = await session.execute(query, params)
        panel_ids = [row[0] for row in result]
        
        if not panel_ids:
            return []
        
        return [{
            "type": "complement",
            "panel_ids": panel_ids,
            "similarity_score": 0.5,
            "reason": "추가 분석을 위한 보완 패널 그룹"
        }]

