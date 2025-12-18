"""
통계 계산기

패널 데이터에 대한 통계를 계산하는 모듈
- 카테고리별 통계 (인구통계, 경제력, 디지털, 라이프스타일)
- 기본 통계 (평균, 중앙값, 분포 등)
"""

from typing import Any, Dict, List, Optional
import statistics


class StatisticsCalculator:
    """통계 계산 클래스"""
    
    def calculate(
        self,
        panels_data: List[Dict[str, Any]],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """패널 데이터에 대한 통계 계산
        
        Args:
            panels_data: 패널 데이터 리스트
            focus_areas: 분석 대상 카테고리 그룹 (예: ["demographics", "economic"])
        
        Returns:
            {
                "demographics": {...},
                "economic": {...},
                "digital": {...},
                "lifestyle": {...}
            }
        """
        if not panels_data:
            return {}
        
        stats = {}
        
        # 카테고리별 통계 계산
        category_mapping = {
            "demographics": self._calculate_demographics,
            "economic": self._calculate_economic,
            "digital": self._calculate_digital,
            "lifestyle": self._calculate_lifestyle,
        }
        
        if focus_areas:
            for area in focus_areas:
                if area in category_mapping:
                    stats[area] = category_mapping[area](panels_data)
        else:
            # 모든 카테고리 계산
            for area, calc_func in category_mapping.items():
                stats[area] = calc_func(panels_data)
        
        return stats
    
    def _calculate_demographics(self, panels_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """인구통계 통계 계산"""
        stats = {}
        
        # 성별 분포
        gender_counts = {}
        for p in panels_data:
            gender = p.get("gender")
            if gender:
                gender_counts[gender] = gender_counts.get(gender, 0) + 1
        stats["gender"] = gender_counts
        
        # 나이 통계
        ages: List[int] = []
        for p in panels_data:
            age = p.get("age")
            if age is not None and isinstance(age, (int, float)):
                ages.append(int(age))
        if ages:
            stats["age"] = {
                "mean": round(statistics.mean(ages), 1),
                "median": statistics.median(ages),
                "min": min(ages),
                "max": max(ages),
                "std": round(statistics.stdev(ages), 1) if len(ages) > 1 else 0,
            }
            # 연령대 분포
            age_groups = {"10대": 0, "20대": 0, "30대": 0, "40대": 0, "50대": 0, "60대+": 0}
            for age in ages:
                if age < 20:
                    age_groups["10대"] += 1
                elif age < 30:
                    age_groups["20대"] += 1
                elif age < 40:
                    age_groups["30대"] += 1
                elif age < 50:
                    age_groups["40대"] += 1
                elif age < 60:
                    age_groups["50대"] += 1
                else:
                    age_groups["60대+"] += 1
            stats["age_groups"] = age_groups
        
        # 지역 분포
        region_counts = {}
        for p in panels_data:
            region = p.get("region_city")
            if region:
                region_counts[region] = region_counts.get(region, 0) + 1
        # 상위 5개만
        stats["region_city"] = dict(sorted(region_counts.items(), key=lambda x: -x[1])[:5])
        
        # 결혼 여부 분포
        marital_counts = {}
        for p in panels_data:
            marital = p.get("marital_status")
            if marital:
                marital_counts[marital] = marital_counts.get(marital, 0) + 1
        stats["marital_status"] = marital_counts
        
        # 자녀수 통계
        children: List[int] = []
        for p in panels_data:
            count = p.get("children_count")
            if count is not None and isinstance(count, (int, float)):
                children.append(int(count))
        if children:
            stats["children_count"] = {
                "mean": round(statistics.mean(children), 1),
                "median": statistics.median(children),
                "max": max(children),
            }
        
        # 가족수 통계
        family_sizes: List[int] = []
        for p in panels_data:
            size = p.get("family_size")
            if size is not None and isinstance(size, (int, float)):
                family_sizes.append(int(size))
        if family_sizes:
            stats["family_size"] = {
                "mean": round(statistics.mean(family_sizes), 1),
                "median": statistics.median(family_sizes),
            }
        
        # 학력 분포
        education_counts = {}
        for p in panels_data:
            edu = p.get("education_level")
            if edu:
                education_counts[edu] = education_counts.get(edu, 0) + 1
        stats["education_level"] = dict(sorted(education_counts.items(), key=lambda x: -x[1])[:5])
        
        # 직업 분포
        occupation_counts = {}
        for p in panels_data:
            occ = p.get("occupation")
            if occ:
                occupation_counts[occ] = occupation_counts.get(occ, 0) + 1
        stats["occupation"] = dict(sorted(occupation_counts.items(), key=lambda x: -x[1])[:5])
        
        return stats
    
    def _calculate_economic(self, panels_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """경제력 통계 계산"""
        stats = {}
        
        # 개인 소득 통계
        personal_incomes: List[int] = []
        for p in panels_data:
            income = p.get("monthly_personal_income")
            if income is not None and isinstance(income, (int, float)):
                personal_incomes.append(int(income))
        if personal_incomes:
            stats["monthly_personal_income"] = {
                "mean": round(statistics.mean(personal_incomes), 0),
                "median": statistics.median(personal_incomes),
                "min": min(personal_incomes),
                "max": max(personal_incomes),
            }
        
        # 가구 소득 통계
        household_incomes: List[int] = []
        for p in panels_data:
            income = p.get("monthly_household_income")
            if income is not None and isinstance(income, (int, float)):
                household_incomes.append(int(income))
        if household_incomes:
            stats["monthly_household_income"] = {
                "mean": round(statistics.mean(household_incomes), 0),
                "median": statistics.median(household_incomes),
                "min": min(household_incomes),
                "max": max(household_incomes),
            }
        
        # 차량 소유율
        car_owners = sum(1 for p in panels_data if p.get("car_ownership") is True)
        stats["car_ownership"] = {
            "count": car_owners,
            "rate": round(car_owners / len(panels_data) * 100, 1) if panels_data else 0,
        }
        
        # 차량 제조사 분포
        car_manufacturer_counts = {}
        for p in panels_data:
            mfg = p.get("car_manufacturer")
            if mfg:
                car_manufacturer_counts[mfg] = car_manufacturer_counts.get(mfg, 0) + 1
        stats["car_manufacturer"] = dict(sorted(car_manufacturer_counts.items(), key=lambda x: -x[1])[:5])
        
        return stats
    
    def _calculate_digital(self, panels_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """디지털 통계 계산"""
        stats = {}
        
        # 휴대폰 브랜드 분포
        phone_brand_counts = {}
        for p in panels_data:
            brand = p.get("phone_brand")
            if brand:
                phone_brand_counts[brand] = phone_brand_counts.get(brand, 0) + 1
        stats["phone_brand"] = dict(sorted(phone_brand_counts.items(), key=lambda x: -x[1])[:5])
        
        # 보유 전자제품 통계
        electronics_counts = {}
        for p in panels_data:
            electronics = p.get("owned_electronics")
            if electronics:
                if isinstance(electronics, list):
                    for item in electronics:
                        electronics_counts[item] = electronics_counts.get(item, 0) + 1
        stats["owned_electronics"] = dict(sorted(electronics_counts.items(), key=lambda x: -x[1])[:10])
        
        return stats
    
    def _calculate_lifestyle(self, panels_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """라이프스타일 통계 계산"""
        stats = {}
        
        # 흡연 경험 통계
        smoking_counts = {}
        for p in panels_data:
            smoking = p.get("smoking_experience")
            if smoking:
                if isinstance(smoking, list):
                    for item in smoking:
                        smoking_counts[item] = smoking_counts.get(item, 0) + 1
        stats["smoking_experience"] = dict(sorted(smoking_counts.items(), key=lambda x: -x[1])[:5])
        
        # 흡연 브랜드 분포
        smoking_brand_counts = {}
        for p in panels_data:
            brands = p.get("smoking_brand")
            if brands:
                if isinstance(brands, list):
                    for brand in brands:
                        smoking_brand_counts[brand] = smoking_brand_counts.get(brand, 0) + 1
        stats["smoking_brand"] = dict(sorted(smoking_brand_counts.items(), key=lambda x: -x[1])[:5])
        
        # 음주 경험 통계
        drinking_counts = {}
        for p in panels_data:
            drinking = p.get("drinking_experience")
            if drinking:
                if isinstance(drinking, list):
                    for item in drinking:
                        drinking_counts[item] = drinking_counts.get(item, 0) + 1
        stats["drinking_experience"] = dict(sorted(drinking_counts.items(), key=lambda x: -x[1])[:5])
        
        return stats

