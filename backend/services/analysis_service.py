"""
RAG 기반 패널 데이터 분석 서비스

검색된 패널 데이터를 분석하여 인사이트, 차트 추천, 비교군 추천을 제공
"""

from typing import Any, Dict, List, Optional
import json
import asyncio
import random
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from botocore.exceptions import ClientError  # type: ignore

from backend.repositories.panel_repository import PanelRepository
from backend.repositories.database import AsyncSessionLocal
from backend.services.statistics_calculator import StatisticsCalculator
from backend.services.comparison_group_finder import ComparisonGroupFinder
from backend.services.metadata_loader import MetadataLoader
from backend.services.search_service import get_bedrock_llm


class AnalysisService:
    """RAG 기반 패널 분석 서비스"""
    
    def __init__(self):
        self.panel_repo = PanelRepository()
        self.stats_calculator = StatisticsCalculator()
        self.comparison_finder = ComparisonGroupFinder()
        self.metadata_loader = MetadataLoader()
        self.llm = get_bedrock_llm(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    
    async def analyze_panels(
        self,
        panel_ids: List[str],
        analysis_type: str = "comprehensive",
        focus_areas: Optional[List[str]] = None,
        query: Optional[str] = None,
        requested_count: Optional[int] = None,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """RAG 기반 패널 분석
        
        Args:
            panel_ids: 분석할 패널 ID 리스트
            analysis_type: 분석 타입 ("basic" | "comprehensive" | "custom")
            focus_areas: 분석 대상 카테고리 그룹 (예: ["demographics", "economic"])
            session: DB 세션 (없으면 새로 생성)
        
        Returns:
            분석 결과 딕셔너리
        """
        close_session = False
        if session is None:
            session = AsyncSessionLocal()
            close_session = True
        
        try:
            total_panel_count = len(panel_ids)
            
            # [최적화 1] 샘플링 개수를 300명으로 축소 (시연 속도 최적화)
            # 500명 -> 300명으로 줄여서 분석 속도 대폭 향상 (시연용)
            SAMPLE_LIMIT = 300
            if total_panel_count > SAMPLE_LIMIT:
                print(f"⚡ 분석 최적화: {total_panel_count}명 -> {SAMPLE_LIMIT}명 샘플링 (시연 속도 최적화)")
                random.seed(42)
                target_panel_ids = random.sample(panel_ids, SAMPLE_LIMIT)
            else:
                target_panel_ids = panel_ids
            
            # 1. Retrieval: 샘플링된 데이터만 DB에서 조회
            raw_panels = await self._retrieve_panel_data(target_panel_ids, session)
            
            # [중요] ORM 객체 -> Dict 변환 (Lazy Loading 방지)
            # Repository에서 가져온 데이터가 ORM 객체인 경우 dict로 변환하여
            # 이후 루프에서 DB 쿼리가 추가로 발생하는 것을 방지합니다.
            panels_data = []
            for p in raw_panels:
                if hasattr(p, "__dict__") and not isinstance(p, dict):
                    # ORM 객체인 경우: SQLAlchemy state 속성 제외하고 dict로 변환
                    p_dict = getattr(p, "__dict__", {})
                    d = {k: v for k, v in p_dict.items() if not k.startswith('_')}
                    panels_data.append(d)
                else:
                    # 이미 dict인 경우 그대로 사용
                    panels_data.append(p)
            
            if not panels_data:
                return {
                    "summary": {"total_panels": 0, "key_insights": [], "notable_findings": []},
                    "statistics": {},
                    "insights": [],
                    "chart_recommendations": [],
                    "comparison_groups": [],
                }
            
            # 2. Augmentation: 컨텍스트 구성
            # context에는 'total_count'를 전체 개수로 넘겨주어 LLM이 전체 규모를 인지하게 함
            context = await self._build_context(panels_data, focus_areas, session)
            context["total_count"] = total_panel_count  # 실제 전체 개수로 덮어쓰기
            
            # [최적화 2] LLM 입력 텍스트 길이 강제 제한 (시연 속도 최적화)
            # 6,000자 -> 3,000자로 축소하여 LLM 처리 시간 대폭 단축 (시연용)
            if len(context.get("panels_text_summary", "")) > 3000:
                context["panels_text_summary"] = context["panels_text_summary"][:3000] + "\n...(요약됨)"
            
            # 3. Generation: LLM 분석
            analysis_result = await self._generate_analysis(panels_data, context, query=query, requested_count=requested_count)
            
            # 샘플링 정보 추가
            if total_panel_count > SAMPLE_LIMIT:
                if "summary" in analysis_result:
                    analysis_result["summary"]["note"] = f"전체 {total_panel_count}명 중 {SAMPLE_LIMIT}명을 표본 분석함"
            
            return analysis_result
        except Exception as e:
            print(f"❌ 분석 중 치명적 오류: {e}")
            import traceback
            traceback.print_exc()
            # 최소한의 통계라도 반환
            total_count = len(panel_ids) if 'panel_ids' in locals() else 0
            return {
                "error": str(e),
                "summary": {"total_panels": total_count},
                "statistics": {},
                "insights": [],
                "chart_recommendations": [],
                "comparison_groups": [],
            }
        finally:
            if close_session:
                await session.close()
    
    async def _retrieve_panel_data(
        self,
        panel_ids: List[str],
        session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """패널 데이터 수집"""
        return await self.panel_repo.get_panels_by_ids(panel_ids, session)
    
    async def _build_context(
        self,
        panels_data: List[Dict[str, Any]],
        focus_areas: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """RAG 컨텍스트 구성"""
        # focus_areas가 None이면 모든 카테고리 사용
        if focus_areas is None:
            # 모든 카테고리 그룹 조회
            category_groups = await self.metadata_loader.load_category_groups(None, session)
            focus_areas = list(category_groups.keys())
        
        # 1. 정형 데이터 분포 요약
        panels_data_summary = self._format_panels_data_summary(panels_data)
        
        # 2. 비정형 데이터 요약 (panel_summary_text)
        panels_text_summary = self._format_panels_text_summary(panels_data)
        
        # 3. 통계 계산 (비동기 래핑 - CPU 연산이 무거운 작업은 별도 스레드로 실행)
        # [최적화] 패널 수가 많으면 통계 계산도 샘플링하여 속도 향상
        stats_panels = panels_data
        if len(panels_data) > 200:
            # 200개 이상이면 200개만 샘플링하여 통계 계산 (시연 속도 최적화)
            random.seed(42)
            stats_panels = random.sample(panels_data, 200)
        
        loop = asyncio.get_running_loop()
        statistics = await loop.run_in_executor(
            None, 
            lambda: self.stats_calculator.calculate(stats_panels, focus_areas)
        )
        statistics_context = self._format_statistics(statistics)
        
        # 4. 메타데이터 조회
        metadata = await self.metadata_loader.load_metadata(focus_areas, session)
        metadata_context = self._format_metadata(metadata)
        
        # 5. 카테고리 그룹 정보 추가
        category_groups = await self.metadata_loader.load_category_groups(focus_areas, session)
        category_groups_context = self._format_category_groups(category_groups)
        
        # 6. 비교군 검색
        comparison_groups = await self.comparison_finder.find_comparison_groups(panels_data, session)
        comparison_context = self._format_comparison(comparison_groups)
        
        return {
            "total_count": len(panels_data),
            "panels_data_summary": panels_data_summary,
            "panels_text_summary": panels_text_summary,
            "statistics_context": statistics_context,
            "metadata_context": metadata_context,
            "category_groups_context": category_groups_context,
            "category_groups": category_groups,  # 카테고리 정보를 프롬프트에 전달
            "comparison_context": comparison_context,
        }
    
    def _format_panels_data_summary(self, panels_data: List[Dict[str, Any]]) -> str:
        """정형 데이터 분포 요약 (age, gender, income 등)"""
        if not panels_data:
            return "패널 데이터가 없습니다."
        
        lines = []
        lines.append(f"총 패널 수: {len(panels_data)}명")
        lines.append("")
        
        # 인구통계
        lines.append("[인구통계]")
        gender_dist = self._count_distribution(panels_data, "gender")
        lines.append(f"  성별: {gender_dist}")
        
        # 나이 통계 (타입 안전성 확보)
        ages: List[int] = []
        for p in panels_data:
            age = p.get("age")
            if age is not None and isinstance(age, (int, float)):
                ages.append(int(age))
        if ages:
            avg_age = sum(ages) / len(ages)
            lines.append(f"  평균 나이: {avg_age:.1f}세")
        
        region_dist = self._get_top_values(panels_data, "region_city", 5)
        lines.append(f"  주요 지역: {region_dist}")
        
        marital_dist = self._count_distribution(panels_data, "marital_status")
        lines.append(f"  결혼 여부: {marital_dist}")
        lines.append("")
        
        # 경제력
        lines.append("[경제력]")
        incomes: List[int] = []
        for p in panels_data:
            income = p.get("monthly_household_income")
            if income is not None and isinstance(income, (int, float)):
                incomes.append(int(income))
        if incomes:
            avg_income = sum(incomes) / len(incomes)
            median_income = sorted(incomes)[len(incomes) // 2] if incomes else 0
            lines.append(f"  평균 가구소득: {avg_income:.0f}만원")
            lines.append(f"  중앙값 가구소득: {median_income:.0f}만원")
        
        car_owners = sum(1 for p in panels_data if p.get("car_ownership") is True)
        lines.append(f"  차량 소유: {car_owners}명 ({car_owners/len(panels_data)*100:.1f}%)")
        lines.append("")
        
        # 디지털/라이프스타일
        lines.append("[디지털/라이프스타일]")
        phone_brand_dist = self._get_top_values(panels_data, "phone_brand", 3)
        lines.append(f"  휴대폰 브랜드: {phone_brand_dist}")
        
        # 배열 필드 처리 (owned_electronics)
        electronics = {}
        for p in panels_data:
            if p.get("owned_electronics"):
                items = p["owned_electronics"] if isinstance(p["owned_electronics"], list) else []
                for item in items:
                    electronics[item] = electronics.get(item, 0) + 1
        if electronics:
            top_electronics = sorted(electronics.items(), key=lambda x: -x[1])[:5]
            lines.append(f"  보유 전자제품: {', '.join(f'{k}({v})' for k, v in top_electronics)}")
        
        return "\n".join(lines)
    
    def _format_panels_text_summary(self, panels_data: List[Dict[str, Any]]) -> str:
        """비정형 데이터 (panel_summary_text) 요약
        
        패널 수에 따라 동적으로 샘플 개수를 조정하여 더 정확한 인사이트 도출
        """
        summaries = [p.get("panel_summary_text") for p in panels_data if p.get("panel_summary_text")]
        if not summaries:
            return "비정형 데이터 요약이 없습니다."
        
        # None 값 필터링
        valid_summaries = [s for s in summaries if s]
        if not valid_summaries:
            return "비정형 데이터 요약이 없습니다."
        
        total_count = len(valid_summaries)
        
        # [최적화] 샘플 개수 대폭 축소 (시연 속도 최적화)
        # 20개 -> 10개로 줄여서 프롬프트 길이 단축 및 LLM 응답 속도 향상
        if total_count <= 5:
            # 5개 이하: 모두 포함
            sample_count = total_count
        else:
            # 최대 10개까지만 보여줘도 충분합니다. (시연 속도 최적화)
            sample_count = min(10, int(total_count * 0.05) + 3)
        
        # 랜덤 샘플링으로 대표성 향상 (단순히 처음 N개가 아닌)
        # 시드 고정으로 재현 가능성 보장 (패널 ID 기반 시드로 더 일관된 샘플링)
        # 패널 ID의 해시값을 시드로 사용하여 같은 패널 그룹에서는 항상 같은 샘플 선택
        if panels_data and len(panels_data) > 0:
            # 첫 번째 패널 ID를 기반으로 시드 생성 (같은 패널 그룹 = 같은 샘플)
            first_panel_id = str(panels_data[0].get("panel_id", ""))
            seed_value = hash(first_panel_id) % 10000  # 0-9999 범위로 정규화
            random.seed(seed_value)
        else:
            random.seed(42)  # 기본 시드
        
        sampled_summaries = random.sample(valid_summaries, min(sample_count, len(valid_summaries)))
        
        lines = [f"총 {total_count}개 패널에 요약 텍스트가 있습니다."]
        lines.append(f"대표 샘플 {len(sampled_summaries)}개 (랜덤 샘플링):")
        for i, summary in enumerate(sampled_summaries, 1):
            if summary:  # None 체크
                # 각 샘플은 100자로 제한 (시연 속도 최적화 - 프롬프트 길이 단축)
                # 너무 짧으면 정보 부족, 너무 길면 컨텍스트 과다
                truncated = summary[:100] if len(summary) > 100 else summary
                lines.append(f"  {i}. {truncated}{'...' if len(summary) > 100 else ''}")
        
        return "\n".join(lines)
    
    def _format_statistics(self, statistics: Dict[str, Dict[str, Any]]) -> str:
        """통계 데이터 포맷팅 (토큰 절약을 위해 요약)"""
        if not statistics:
            return "통계 데이터가 없습니다."
        
        lines = []
        # 주요 카테고리만 포함 (토큰 절약)
        priority_categories = ["demographics", "economic", "digital", "lifestyle"]
        
        for category in priority_categories:
            if category in statistics:
                stats = statistics[category]
                lines.append(f"[{category}]")
                # 주요 통계만 포함 (최대 5개)
                for idx, (key, value) in enumerate(stats.items()):
                    if idx >= 5:  # 카테고리당 최대 5개 통계만
                        break
                    if isinstance(value, dict):
                        # dict는 간단히 요약
                        value_str = json.dumps(value, ensure_ascii=False)
                        if len(value_str) > 200:  # 너무 길면 요약
                            value_str = value_str[:200] + "..."
                        lines.append(f"  {key}: {value_str}")
                    else:
                        lines.append(f"  {key}: {value}")
                lines.append("")
        
        # 기타 카테고리는 요약만
        other_categories = [cat for cat in statistics.keys() if cat not in priority_categories]
        if other_categories:
            lines.append(f"[기타 카테고리: {', '.join(other_categories)}]")
            lines.append("  (상세 통계는 생략)")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """메타데이터 포맷팅"""
        if not metadata:
            return "메타데이터가 없습니다."
        
        lines = []
        column_metadata = metadata.get("column_metadata", {})
        
        # 주요 컬럼만 포함 (토큰 절약)
        high_priority_cols = [
            col for col, meta in column_metadata.items()
            if meta.get("analysis_priority") == "high"
        ][:10]
        
        for col_name in high_priority_cols:
            meta = column_metadata.get(col_name, {})
            lines.append(f"  {col_name} ({meta.get('name_ko', '')}): {meta.get('type', '')} - {meta.get('description', '')}")
        
        return "\n".join(lines) if lines else "메타데이터가 없습니다."
    
    def _format_category_groups(self, category_groups: Dict[str, Dict[str, Any]]) -> str:
        """카테고리 그룹 정보 포맷팅 (토큰 절약을 위해 간소화)"""
        if not category_groups:
            return "카테고리 그룹 정보가 없습니다."
        
        lines = []
        lines.append("카테고리별 분석 가이드:")
        lines.append("**반드시 각 카테고리별로 인사이트를 도출하세요:**")
        lines.append("")
        
        # 주요 카테고리와 기타 카테고리 구분
        main_categories = ["demographics", "economic", "digital", "lifestyle", "health_wellness", "tech_digital_life", "consumption_finance"]
        other_categories = [k for k in category_groups.keys() if k not in main_categories and k != "summary"]
        
        lines.append("**주요 카테고리 (각 카테고리당 최소 2-3개 인사이트 필수):**")
        for group_key in main_categories:
            if group_key in category_groups:
                group_data = category_groups[group_key]
                name_ko = group_data.get("name_ko", group_key)
                # 간소화: 이름과 분석 포커스만 포함
                analysis_focus = group_data.get("analysis_focus", [])
                focus_str = ', '.join(analysis_focus[:3]) if analysis_focus else "일반 분석"
                lines.append(f"  • {name_ko} ({group_key}): {focus_str}")
        
        lines.append("")
        lines.append("**기타 카테고리 (각 카테고리당 최소 1-2개 인사이트 권장):**")
        for group_key in other_categories:
            if group_key in category_groups:
                group_data = category_groups[group_key]
                name_ko = group_data.get("name_ko", group_key)
                lines.append(f"  • {name_ko} ({group_key})")
        
        return "\n".join(lines) if lines else "카테고리 그룹 정보가 없습니다."
    
    def _format_comparison(self, comparison_groups: List[Dict[str, Any]]) -> str:
        """비교군 정보 포맷팅"""
        if not comparison_groups:
            return "비교군이 없습니다."
        
        lines = []
        for group in comparison_groups[:3]:  # 상위 3개만
            lines.append(f"  {group.get('type', 'unknown')}: {len(group.get('panel_ids', []))}개 패널 - {group.get('reason', '')}")
        
        return "\n".join(lines) if lines else "비교군이 없습니다."
    
    def _count_distribution(self, panels_data: List[Dict], field: str) -> str:
        """필드별 분포 카운트"""
        counts = {}
        for p in panels_data:
            val = p.get(field)
            if val:
                counts[val] = counts.get(val, 0) + 1
        return ", ".join(f"{k}({v}명)" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    
    def _get_top_values(self, panels_data: List[Dict], field: str, top_n: int) -> str:
        """상위 N개 값"""
        counts = {}
        for p in panels_data:
            val = p.get(field)
            if val:
                counts[val] = counts.get(val, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        return ", ".join(f"{k}({v})" for k, v in top)
    
    async def _generate_analysis(
        self,
        panels_data: List[Dict[str, Any]],
        context: Dict[str, Any],
        query: Optional[str] = None,
        requested_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """LLM을 이용한 분석 생성"""
        if not self.llm:
            # LLM이 없으면 기본 통계만 반환 (비동기 처리)
            loop = asyncio.get_running_loop()
            statistics = await loop.run_in_executor(
                None,
                lambda: self.stats_calculator.calculate(panels_data)
            )
            return {
                "summary": {
                    "total_panels": context.get("total_count", 0),
                    "key_insights": [],
                    "notable_findings": [],
                },
                "statistics": statistics,
                "insights": [],
                "chart_recommendations": [],
                "comparison_groups": [],
            }
        
        try:
            # 프롬프트 구성 (개선: 숨겨진 인사이트 발견 강조)
            system_prompt = """당신은 패널 데이터 분석 전문가이자 비즈니스 인사이트 도출 전문가입니다. 
단순한 통계 요약이 아닌, 데이터 속에 숨겨진 패턴과 의미를 발견하여 실용적인 비즈니스 인사이트를 제공하세요.

**중요: 출력 형식**
- 반드시 유효한 JSON 형식으로만 응답하세요.
- 설명이나 서문 없이 JSON만 반환하세요.
- JSON이 완전해야 합니다 (모든 중괄호와 배열이 닫혀있어야 함).

**일관성 원칙 (매우 중요):**
- 모든 인사이트에서 동일한 품질과 길이 기준을 유지하세요.
- 비즈니스 함의와 추천 사항은 모든 인사이트에서 반드시 포함되어야 하며, 길이와 상세도가 일관되어야 합니다.
- 데이터가 부족한 카테고리에서도 가능한 한 인사이트를 도출하되, 품질을 유지하세요.

**분석 원칙:**
1. **숨겨진 패턴 발견**: 표면적인 통계가 아닌, 변수 간 상관관계와 예상치 못한 연관성을 찾아내세요.
   - 예: "고소득층이 특정 브랜드를 선호한다"는 단순 통계가 아니라, "고소득층 중에서도 30대 기혼 남성이 특정 브랜드 선호도가 높다"는 구체적 패턴
   - 예: "OTT 이용률이 높다"는 단순 통계가 아니라, "OTT 이용률이 높은 그룹은 특정 라이프스타일 특성을 공유한다"는 연관성

2. **핵심 인사이트 vs 상세 인사이트 구분**:
   - **핵심 인사이트 (key_insights)**: 전체 데이터를 종합한 큰 그림, 전략적 관점의 발견 (3-5개)
   - **상세 인사이트 (insights)**: 구체적인 발견사항, 실행 가능한 액션 아이템 (10-15개)
   - 핵심 인사이트와 상세 인사이트는 중복되지 않아야 합니다.

3. **상세한 코멘트 제공**: 각 인사이트마다 다음을 포함하세요:
   - 발견 사항 (finding): 구체적이고 측정 가능한 사실
   - 중요도 (significance): high/medium/low (비즈니스 영향도 기준)
   - 비즈니스 함의 (business_implication): 이 발견이 비즈니스에 미치는 영향과 의미 (2-3문장)
   - 추천 사항 (recommendation): 구체적인 실행 방안이나 추가 분석 제안 (2-3문장)

4. **비교 분석**: 전체 패널 대비 이 그룹의 특이점을 강조하세요.

메타데이터 정보:
{metadata_context}

통계 정보:
{statistics_context}

비교군 정보:
{comparison_context}"""

            # 명수 정보 추가
            count_info = ""
            if requested_count is not None:
                count_info = f"\n**원본 질의에서 요청한 명수: {requested_count}명**\n"
            
            query_info = ""
            if query:
                query_info = f"\n**원본 질의: {query}**\n"
            
            user_prompt_template = """다음 패널 그룹 데이터를 심층 분석해주세요:""" + query_info + count_info + """
총 패널 수: {total_count}명

[정형 데이터 분포]
{panels_data_summary}

[비정형 데이터 요약 (LLM 생성 요약 텍스트)]
{panels_text_summary}

[계산된 통계]
{statistics_context}

[메타데이터 정보]
{metadata_context}

[카테고리 그룹 정보]
{category_groups_context}

[비교군 정보]
{comparison_context}

**분석 요청 (중요):**

1. **숨겨진 패턴 발견**:
   - 정형 데이터와 비정형 데이터를 교차 분석하여 예상치 못한 연관성 찾기
   - 변수 간 상관관계 분석 (예: 소득과 라이프스타일, 연령과 디지털 수용도)
   - 특이한 조합이나 예외적 패턴 발견

2. **핵심 인사이트 도출** (5-7개):
   - 전체 데이터를 종합한 전략적 관점의 발견
   - 이 패널 그룹의 핵심 특성과 차별점
   - 비즈니스 전략 수립에 중요한 큰 그림
   - **중요**: 각 핵심 인사이트는 서로 다른 관점이나 측면을 다루어야 합니다 (예: 인구통계, 소비패턴, 라이프스타일, 디지털 수용도 등)
   - **길이**: 각 인사이트는 간결하면서도 핵심을 담아야 합니다 (너무 짧거나 길지 않게 적절한 길이로 작성)

3. **카테고리별 상세 인사이트 도출** (**반드시 최소 15개 이상, 목표 20-30개**):
   - **중요**: insights 배열에는 반드시 최소 15개 이상의 인사이트를 포함해야 합니다. 15개 미만이면 분석이 불완전한 것으로 간주됩니다.
   - **반드시 각 카테고리별로 분석**: 카테고리 그룹 정보를 참고하여 모든 카테고리에서 인사이트 도출
   - **실무 활용 중심**: 리서치 기업에서 실무에 바로 활용할 수 있는 구체적이고 실행 가능한 발견사항 제시
   - **중복 방지**: 
     * 같은 주제나 측면을 다루는 인사이트는 한 번만 작성
     * 각 인사이트는 서로 다른 관점, 변수, 또는 측면을 다루어야 함
     * 예: "차량 보유율"에 대한 인사이트가 있으면, 같은 내용을 다른 수치로 반복하지 않음
   - **카테고리별 실무 중심 분배** (최소 개수 보장):
     * demographics (인구통계): 최소 2개 이상 (연령/성별/지역별 세분화, 결혼/가족 구조 특성)
     * economic (경제력): 최소 2개 이상 (소득 분포, 소비 여력, 경제적 특성)
     * digital (디지털): 최소 2개 이상 (디지털 기기 보유, 디지털 서비스 이용, 디지털 수용도)
     * lifestyle (라이프스타일): 최소 2개 이상 (생활 패턴, 여가 활동, 소비 습관)
     * health_wellness (건강/신체관리): 최소 1개 이상 (건강 관리 관심도, 운동/식습관, 건강 관련 소비)
     * tech_digital_life (기술 및 디지털 라이프): 최소 1개 이상 (기술 제품 선호, 디지털 라이프스타일)
     * consumption_finance (소비 및 재테크): 최소 1개 이상 (소비 패턴, 금융 상품 이용, 재테크 관심)
     * travel_culture (여행 및 문화생활): 최소 1개 이상 (여행 빈도/선호지, 문화 활동)
     * psychology_stress (심리 및 스트레스 관리): 최소 1개 이상 (스트레스 요인, 심리적 특성)
     * daily_habits (일상생활 태도 및 습관): 최소 1개 이상 (일상 습관, 생활 태도)
     * values_experience (경험 및 가치관): 최소 1개 이상 (가치관, 경험 선호도)
   - 각 인사이트의 finding은 구체적이고 측정 가능한 사실을 제시하세요 (적절한 길이로 충분히 상세하게)
   - **소제목 다양화**: 같은 카테고리 내에서도 서로 다른 측면을 다루도록 소제목/관점을 다양하게 설정

4. **상세한 코멘트 제공**:
   - 각 인사이트마다 비즈니스 함의(business_implication)를 상세히 작성하세요 (2-3문장, 충분히 설명)
   - 각 인사이트마다 추천 사항(recommendation)을 구체적인 실행 방안으로 제시하세요 (2-3문장, 실무 활용 가능하게)
   - 단순 나열이 아닌, 의미 해석과 실행 방안 포함
   - **중요**: 비즈니스 함의와 추천 사항은 반드시 포함되어야 하며, 모든 인사이트에서 일관된 품질을 유지하세요
   - **품질 기준**: 각 코멘트는 구체적인 수치나 사례를 포함하고, 실무에 바로 활용 가능한 수준이어야 합니다

5. **차트 추천** (최소 2개 필수): 데이터를 효과적으로 시각화할 수 있는 차트 타입과 이유
   - **반드시 최소 2개의 차트를 추천**하세요
   - 각 차트는 서로 다른 필드나 관점을 다루어야 합니다
   - 차트 타입은 데이터 특성에 맞게 선택 (pie, bar, histogram 등)
   - 각 차트마다 제목, 설명, 데이터 필드, 집계 방식을 명시하세요

6. **비교군 추천** (2~3개 필수): 추가 분석을 위한 유사/대조 패널 그룹 추천
   - **반드시 2~3개의 비교군을 추천**하세요
   - 유사 그룹(similar): 비슷한 특성을 가진 그룹으로 추가 분석 가능
   - 대조 그룹(contrast): 반대 특성을 가진 그룹으로 차이점 분석 가능
   - 보완 그룹(complement): 다른 관점에서 보완 분석 가능한 그룹
   - 각 비교군마다 추천 이유와 **자연어 형태의 검색어**를 명시하세요
   - **중요**: `query_suggestion`은 자연어 문장 형태로 작성하세요 (예: "30대 미혼 여성 중 고소득층", "서울 거주 비흡연자", "기혼 남성 중 차량 보유자" 등)
   - **명수 포함 규칙 (매우 중요)**: 
     * 위 프롬프트 시작 부분에 "원본 질의에서 요청한 명수: N명" 정보가 있으면, 비교군 추천 검색어에도 반드시 동일한 명수를 포함하세요 (예: "30대 미혼 여성 중 고소득층 N명")
     * "원본 질의에서 요청한 명수" 정보가 없으면, 비교군 추천 검색어에도 명수를 포함하지 마세요 (예: "30대 미혼 여성 중 고소득층" - 명수 없음)

**출력 형식 (JSON - 반드시 이 형식으로만 응답):**
**중요: 설명이나 서문 없이 아래 JSON 형식으로만 응답하세요. JSON만 반환하세요.**

{{
    "key_insights": [
        "전체 데이터를 종합한 전략적 관점의 핵심 발견 1 (간결하면서도 핵심을 담은 문장)",
        "전체 데이터를 종합한 전략적 관점의 핵심 발견 2 (간결하면서도 핵심을 담은 문장)",
        "... (총 5-7개)"
    ],
    "insights": [
        {{
            "category": "{category_list}",
            "finding": "구체적이고 측정 가능한 발견 사항 (충분히 상세하게, 예: '30대 기혼 남성의 65%가 프리미엄 브랜드를 선호하며, 이는 전체 평균(35%)보다 30%p 높음. 특히 월 가구소득 700만원 이상 그룹에서 이 선호도가 80%로 더욱 높게 나타남')",
            "significance": "high|medium|low",
            "business_implication": "이 발견이 비즈니스에 미치는 영향과 의미를 상세히 설명 (2-3문장). 예: '이 그룹은 프리미엄 제품 마케팅의 핵심 타겟으로, 높은 구매력과 브랜드 충성도를 보여준다. 따라서 이 그룹을 대상으로 한 맞춤형 마케팅 전략이 효과적일 것으로 예상된다. 특히 이 그룹의 라이프스타일 특성을 반영한 제품 포지셔닝이 중요하다.'",
            "recommendation": "구체적인 실행 방안이나 추가 분석 제안을 작성 (2-3문장). 예: '프리미엄 브랜드 포지셔닝 강화와 함께, 이 그룹의 라이프스타일 특성을 반영한 제품 개발을 권장한다. 또한 유사한 특성을 가진 다른 세그먼트에 대한 추가 분석을 통해 시장 확장 가능성을 검토해야 한다. 마케팅 채널 선호도와 소비 패턴에 대한 심층 분석도 함께 진행하는 것이 좋겠다.'"
        }}
    ],
    "chart_recommendations": [
        {{
            "type": "pie|bar|histogram|box|treemap",
            "title": "차트 제목",
            "description": "차트 설명 및 왜 이 차트가 유용한지 설명",
            "category": "{category_list}",
            "data_spec": {{
                "field": "column_name",
                "aggregation": "count|mean|distribution"
            }}
        }},
        {{
            "type": "pie|bar|histogram|box|treemap",
            "title": "차트 제목 2",
            "description": "차트 설명 및 왜 이 차트가 유용한지 설명",
            "category": "{category_list}",
            "data_spec": {{
                "field": "column_name",
                "aggregation": "count|mean|distribution"
            }}
        }}
    ],
    "comparison_suggestions": [
        {{
            "type": "similar|contrast|complement",
            "reason": "추천 이유를 상세히 설명",
            "query_suggestion": "자연어 형태의 검색어 (원본 질의에 명수가 있으면 명수 포함, 없으면 명수 없음. 예: '30대 미혼 여성 중 고소득층 100명' 또는 '30대 미혼 여성 중 고소득층')"
        }},
        {{
            "type": "similar|contrast|complement",
            "reason": "추천 이유를 상세히 설명",
            "query_suggestion": "자연어 형태의 검색어 (원본 질의에 명수가 있으면 명수 포함, 없으면 명수 없음. 예: '서울 거주 비흡연자 100명' 또는 '서울 거주 비흡연자')"
        }},
        {{
            "type": "similar|contrast|complement",
            "reason": "추천 이유를 상세히 설명",
            "query_suggestion": "자연어 형태의 검색어 (원본 질의에 명수가 있으면 명수 포함, 없으면 명수 없음. 예: '기혼 남성 중 차량 보유자 100명' 또는 '기혼 남성 중 차량 보유자')"
        }}
    ]
}}

**중요**: 
1. **중복 방지 (최우선)**: 
   - key_insights와 insights는 완전히 다른 주제나 관점을 다루어야 합니다
   - 같은 주제의 인사이트는 한 번만 작성하세요. 예를 들어, "기아/현대 차량 선호도"에 대한 인사이트가 key_insights에 있으면, insights에 같은 내용을 다른 수치로 반복하지 마세요
   - 각 인사이트는 서로 다른 변수, 측면, 또는 관점을 다루어야 합니다 (예: 하나는 "차량 보유율", 다른 하나는 "차량 브랜드 선호도"는 가능하지만, 둘 다 "차량 보유율"은 불가)
   
2. **수치 일관성**: 
   - 같은 통계를 다르게 표현하지 마세요. 예를 들어, "기아/현대 차량 선호도 62.5%"와 "기아/현대 차량 선호도 72.6%"처럼 같은 주제에 다른 수치를 제시하지 마세요
   - 정확한 수치를 한 번만 사용하세요
   
3. **계산 기준 명확화**: 
   - 수치를 제시할 때는 계산 기준을 명확히 하세요. 예를 들어, "전체 패널 기준"인지 "차량 보유자 기준"인지 명시하세요
   
4. **실무 활용성**: 
   - 리서치 기업에서 실무에 바로 활용할 수 있는 구체적이고 실행 가능한 인사이트를 제공하세요
   - 단순 통계 나열이 아닌, 비즈니스 의사결정에 도움이 되는 인사이트를 도출하세요
   - 각 인사이트는 마케팅, 제품 개발, 타겟팅 등 실무에 활용 가능한 형태로 작성하세요
   
5. **차트 추천 필수**: 
   - chart_recommendations는 반드시 최소 2개 이상이어야 합니다
   - 각 차트는 서로 다른 필드나 관점을 다루어야 합니다
   
6. **비교군 추천 필수**: 
   - comparison_suggestions는 반드시 2~3개 이상이어야 합니다
   - 유사 그룹, 대조 그룹, 보완 그룹 등 다양한 관점의 비교군을 제시하세요

7. **길이 가이드라인 (유연한 기준)**: 
   - key_insights: 간결하면서도 핵심을 담은 문장 (너무 짧거나 길지 않게)
   - insights.finding: 구체적이고 측정 가능한 사실을 충분히 상세하게 제시
   - insights.business_implication: 상세한 설명 (2-3문장, 의미를 충분히 전달)
   - insights.recommendation: 구체적인 실행 방안 (2-3문장, 실무에 활용 가능하게)
   - **중요**: 내용의 질과 완성도에 집중하세요. 글자 수보다는 의미 있는 인사이트를 제공하는 것이 우선입니다.

8. **카테고리별 분석 필수 (최소 개수 보장)**: 
   - **반드시 최소 10개 이상의 insights를 생성하세요. 10개 미만이면 분석이 불완전합니다. (시연 속도 최적화)**
   - 모든 카테고리 그룹에서 최소 1개 이상의 인사이트를 도출하세요 (시연 속도 최적화)
   - 주요 카테고리(demographics, economic, digital, lifestyle)는 각각 최소 1개 이상 (시연 속도 최적화)
   - 기타 카테고리(health_wellness, tech_digital_life, consumption_finance 등)는 각각 최소 1개 이상
   - 카테고리별로 고르게 분배하여 총 최소 10개 이상, 목표 15-20개의 상세 인사이트를 생성하세요 (시연 속도 최적화)
   - **데이터가 부족한 카테고리도 가능한 한 인사이트를 도출하세요. 통계나 패턴이 없어도 전체적인 특성을 분석할 수 있습니다.**"""

            # 카테고리 목록 생성 (프롬프트에 주입)
            category_groups = context.get("category_groups", {})
            category_list = "|".join(category_groups.keys()) if category_groups else "demographics|economic|digital|lifestyle|health_wellness|tech_digital_life|consumption_finance|travel_culture|psychology_stress|daily_habits|values_experience|summary"
            
            # context에 category_list 추가
            context["category_list"] = category_list
            
            # 프롬프트 포맷팅 (context 값으로 채우기)
            user_prompt_formatted = user_prompt_template.format(**context)
            
            # 컨텍스트 길이 확인 및 로깅 (토큰 절약 확인)
            context_lengths = {
                "panels_data_summary": len(context.get("panels_data_summary", "")),
                "panels_text_summary": len(context.get("panels_text_summary", "")),
                "statistics_context": len(context.get("statistics_context", "")),
                "metadata_context": len(context.get("metadata_context", "")),
                "category_groups_context": len(context.get("category_groups_context", "")),
                "comparison_context": len(context.get("comparison_context", "")),
            }
            total_context_length = sum(context_lengths.values())
            user_prompt_length = len(user_prompt_formatted)
            
            print(f"📊 RAG 컨텍스트 크기:")
            for key, length in context_lengths.items():
                print(f"  - {key}: {length:,}자")
            print(f"  - user_prompt: {user_prompt_length:,}자")
            print(f"  - 총 컨텍스트: {total_context_length:,}자")
            
            # 컨텍스트가 너무 길면 경고 및 자동 요약 (시연 속도 최적화)
            MAX_CONTEXT_LENGTH = 30000  # 약 7,500 토큰 (한글 기준 1자 = 0.25토큰, 시연용으로 축소)
            if total_context_length > MAX_CONTEXT_LENGTH:
                print(f"⚠️ 경고: RAG 컨텍스트가 너무 깁니다! ({total_context_length:,}자)")
                print(f"  LLM이 모든 정보를 처리하지 못할 수 있습니다.")
                # 컨텍스트 정규화: 가장 긴 부분을 요약 (시연 속도 최적화)
                if len(context.get("panels_text_summary", "")) > 10000:
                    # 비정형 데이터 요약이 너무 길면 추가로 요약
                    original_text = context.get("panels_text_summary", "")
                    # 샘플 개수를 줄여서 요약
                    lines = original_text.split("\n")
                    if len(lines) > 15:
                        # 처음 10개와 마지막 5개만 유지 (시연 속도 최적화)
                        context["panels_text_summary"] = "\n".join(lines[:10] + ["... (중간 생략) ..."] + lines[-5:])
                        print(f"  ✅ 비정형 데이터 요약 축소: {len(original_text):,}자 → {len(context['panels_text_summary']):,}자")
                
                if len(context.get("statistics_context", "")) > 8000:
                    # 통계 컨텍스트가 너무 길면 주요 카테고리만 유지 (시연 속도 최적화)
                    stats_lines = context.get("statistics_context", "").split("\n")
                    # 주요 카테고리만 추출 (demographics, economic, digital, lifestyle)
                    filtered_lines = []
                    current_category = None
                    for line in stats_lines:
                        if any(cat in line for cat in ["demographics", "economic", "digital", "lifestyle"]):
                            current_category = line
                            filtered_lines.append(line)
                        elif current_category and line.strip() and not line.startswith("["):
                            filtered_lines.append(line)
                            if len(filtered_lines) > 30:  # 최대 30줄 (시연 속도 최적화)
                                break
                    context["statistics_context"] = "\n".join(filtered_lines)
                    print(f"  ✅ 통계 컨텍스트 축소: 주요 카테고리만 유지")
            
            # system_prompt도 먼저 포맷팅 (metadata_context, statistics_context, comparison_context 포함)
            system_prompt_formatted = system_prompt.format(
                metadata_context=context.get("metadata_context", ""),
                statistics_context=context.get("statistics_context", ""),
                comparison_context=context.get("comparison_context", "")
            )
            
            # LLM 호출 (재시도 로직 포함)
            # 직접 메시지 리스트를 LLM에 전달 (ChatPromptTemplate 사용하지 않음)
            from langchain_core.messages import SystemMessage, HumanMessage
            
            max_retries = 3
            retry_delay = 2  # 초
            result_text = None
            
            # 메시지 리스트 생성
            messages = [
                SystemMessage(content=system_prompt_formatted),
                HumanMessage(content=user_prompt_formatted)
            ]
            
            for attempt in range(max_retries):
                try:
                    # LLM에 직접 메시지 전달 (ChatPromptTemplate 없이)
                    response = await self.llm.ainvoke(messages)
                    # AIMessage에서 텍스트 추출
                    result_text = response.content if hasattr(response, 'content') else str(response)
                    break  # 성공하면 루프 탈출
                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code == "ThrottlingException" and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # 지수 백오프
                        print(f"⚠️ ThrottlingException 발생. {wait_time}초 후 재시도 ({attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise  # 다른 에러이거나 최대 재시도 횟수 초과
                except Exception as e:
                    # ThrottlingException이 아닌 다른 에러는 즉시 재발생
                    raise
            
            # 재시도 실패 시 예외 발생
            if result_text is None:
                raise Exception("LLM 분석 실패: 최대 재시도 횟수 초과")
            
            # result_text를 문자열로 변환 (타입 안전성)
            if not isinstance(result_text, str):
                if isinstance(result_text, list):
                    # 리스트인 경우 첫 번째 요소가 문자열이면 사용
                    result_text = str(result_text[0]) if result_text and isinstance(result_text[0], str) else str(result_text)
                else:
                    result_text = str(result_text)
            
            # 텍스트에서 JSON 부분 추출 (스택 기반 파서 - 완벽한 JSON 추출)
            def extract_json_from_text(text: str) -> Dict[str, Any]:
                """스택 기반으로 가장 바깥쪽의 온전한 JSON 객체를 추출 (탐욕 문제 해결)"""
                import re
                import json
                
                text = text.strip()
                
                # 1. 마크다운 코드 블록 제거
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
                if match:
                    text = match.group(1).strip()
                
                # 2. 스택 기반 괄호 짝 찾기 (Nested structure 지원)
                # 정규식의 탐욕 문제를 해결하기 위해 스택을 사용하여 정확한 JSON 경계를 찾습니다.
                stack = []
                start_index = -1
                
                for i, char in enumerate(text):
                    if char == '{':
                        if not stack:
                            start_index = i  # 첫 번째 { 발견
                        stack.append('{')
                    elif char == '}':
                        if stack:
                            stack.pop()
                            if not stack:
                                # 스택이 비워지는 순간이 가장 바깥쪽 JSON의 끝
                                json_str = text[start_index : i+1]
                                try:
                                    parsed = json.loads(json_str)
                                    print(f"✅ JSON 추출 성공 (스택 방식): {len(json_str)}자")
                                    return parsed
                                except json.JSONDecodeError:
                                    # 실패하면 계속 탐색 (혹시 뒤에 또 다른 JSON이 있을 수 있음)
                                    start_index = -1
                                    continue
                
                # 3. 스택 방식 실패 시 최후의 수단 (Non-greedy Regex)
                # .*? 를 사용하여 가장 먼저 닫히는 구간을 찾음
                match = re.search(r"(\{.*?\})", text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    try:
                        parsed = json.loads(json_str)
                        print(f"✅ JSON 추출 성공 (Non-greedy Regex): {len(json_str)}자")
                        return parsed
                    except json.JSONDecodeError:
                        # 파싱 실패 시 제어 문자 청소 후 재시도
                        try:
                            cleaned_str = json_str.replace('\n', ' ').replace('\r', '')
                            parsed = json.loads(cleaned_str)
                            print(f"✅ JSON 추출 성공 (청소 후): {len(cleaned_str)}자")
                            return parsed
                        except:
                            pass
                
                # 4. 최후의 수단: key_insights만이라도 추출 시도
                try:
                    key_insights_match = re.search(r'"key_insights"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                    if key_insights_match:
                        insights_content = key_insights_match.group(1)
                        # 문자열 추출 (이스케이프된 따옴표 처리)
                        insight_strings = re.findall(r'"((?:[^"\\]|\\.)*)"', insights_content)
                        if insight_strings:
                            print(f"⚠️ 부분 JSON 복구: key_insights {len(insight_strings)}개만 추출")
                            return {
                                "key_insights": insight_strings,
                                "insights": [],
                                "chart_recommendations": [],
                                "comparison_suggestions": []
                            }
                except Exception as e3:
                    print(f"❌ 부분 JSON 복구도 실패: {e3}")
                
                raise ValueError("유효한 JSON을 추출할 수 없습니다.")
            
            # JSON 추출
            try:
                result = extract_json_from_text(result_text)
            except Exception as e:
                print(f"❌ JSON 추출 실패: {e}")
                print(f"📝 LLM 응답 전체 (처음 2000자): {result_text[:2000] if result_text else 'None'}")
                print(f"📝 LLM 응답 전체 (마지막 1000자): {result_text[-1000:] if result_text and len(result_text) > 1000 else 'None'}")
                # 빈 결과라도 반환하여 프론트엔드에서 처리 가능하도록
                print(f"⚠️ JSON 추출 실패로 빈 인사이트 반환")
                raise Exception(f"LLM 응답에서 JSON을 추출할 수 없습니다: {str(e)}")
            
            # 디버깅: LLM 결과 확인
            print(f"📊 LLM 분석 결과 수신:")
            print(f"  - result 타입: {type(result)}")
            print(f"  - result 키 목록: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            print(f"  - key_insights 개수: {len(result.get('key_insights', []))}")
            print(f"  - insights 개수: {len(result.get('insights', []))}")
            print(f"  - chart_recommendations 개수: {len(result.get('chart_recommendations', []))}")
            print(f"  - comparison_suggestions 개수: {len(result.get('comparison_suggestions', []))}")
            
            # 인사이트가 비어있는 경우 상세 로깅
            if not result.get('key_insights') and not result.get('insights'):
                print(f"⚠️ 경고: key_insights와 insights가 모두 비어있습니다!")
                print(f"  - result 전체 구조: {json.dumps(result, ensure_ascii=False, indent=2)[:2000]}")
            
            # insights 필드 검증 및 정리
            insights_raw = result.get('insights', [])
            insights_validated = []
            missing_fields_count = {"category": 0, "finding": 0, "significance": 0, "business_implication": 0, "recommendation": 0}
            
            for idx, insight in enumerate(insights_raw):
                if not isinstance(insight, dict):
                    print(f"  ⚠️ insights[{idx}]가 dict가 아닙니다: {type(insight)}")
                    continue
                
                # 필수 필드 확인
                category = insight.get("category")
                finding = insight.get("finding")
                significance = insight.get("significance")
                business_implication = insight.get("business_implication")
                recommendation = insight.get("recommendation")
                
                # 필수 필드 누락 체크
                if not category:
                    missing_fields_count["category"] += 1
                    print(f"  ⚠️ insights[{idx}]: category 누락")
                if not finding:
                    missing_fields_count["finding"] += 1
                    print(f"  ⚠️ insights[{idx}]: finding 누락")
                if not significance:
                    missing_fields_count["significance"] += 1
                    print(f"  ⚠️ insights[{idx}]: significance 누락")
                if not business_implication:
                    missing_fields_count["business_implication"] += 1
                    print(f"  ⚠️ insights[{idx}]: business_implication 누락")
                if not recommendation:
                    missing_fields_count["recommendation"] += 1
                    print(f"  ⚠️ insights[{idx}]: recommendation 누락")
                
                # 최소한 category와 finding이 있으면 포함 (나머지는 빈 문자열로 채움)
                if category and finding:
                    # 길이 검증 및 품질 체크
                    finding_len = len(finding)
                    implication_len = len(business_implication) if business_implication else 0
                    recommendation_len = len(recommendation) if recommendation else 0
                    
                    # 길이 경고 (일관성 체크)
                    if finding_len < 70 or finding_len > 130:
                        # 글자 수 검증 완화: 내용의 질에 집중
                        if finding_len < 30:
                            print(f"  ⚠️ insights[{idx}]: finding이 너무 짧습니다 ({finding_len}자, 최소 30자 권장)")
                    if business_implication and (implication_len < 100 or implication_len > 200):
                        # 글자 수 검증 완화: 내용의 질에 집중
                        if implication_len < 50:
                            print(f"  ⚠️ insights[{idx}]: business_implication이 너무 짧습니다 ({implication_len}자, 최소 50자 권장)")
                    if recommendation and (recommendation_len < 100 or recommendation_len > 200):
                        # 글자 수 검증 완화: 내용의 질에 집중
                        if recommendation_len < 50:
                            print(f"  ⚠️ insights[{idx}]: recommendation이 너무 짧습니다 ({recommendation_len}자, 최소 50자 권장)")
                    
                    insights_validated.append({
                        "category": category,
                        "finding": finding,
                        "significance": significance or "medium",
                        "business_implication": business_implication or "",
                        "recommendation": recommendation or ""
                    })
                else:
                    print(f"  ⚠️ insights[{idx}]: 필수 필드(category, finding) 누락으로 제외")
            
            # 검증 결과 로깅
            if any(count > 0 for count in missing_fields_count.values()):
                print(f"⚠️ insights 필드 누락 통계:")
                for field, count in missing_fields_count.items():
                    if count > 0:
                        print(f"  - {field}: {count}개 누락")
            
            # 검증된 insights로 교체
            result['insights'] = insights_validated
            print(f"  ✅ 검증 완료: {len(insights_validated)}개 insights 유효 (원본: {len(insights_raw)}개)")
            
            # 인사이트 개수 검증 (최소 10개 이상 권장 - 시연 속도 최적화)
            insights_count = len(insights_validated)
            if insights_count < 10:
                print(f"⚠️ 경고: insights 개수가 부족합니다! (현재: {insights_count}개, 권장: 10개 이상)")
                print(f"  - 패널 수: {len(panels_data)}개")
                print(f"  - 통계 데이터: {len(context.get('statistics_context', ''))}자")
                print(f"  - 비정형 데이터 샘플: {len(context.get('panels_text_summary', ''))}자")
                if insights_count < 5:
                    print(f"  ⚠️ 심각: insights가 5개 미만입니다. LLM이 프롬프트 요구사항을 제대로 따르지 않았을 수 있습니다.")
            
            # chart_recommendations 검증
            chart_recommendations = result.get('chart_recommendations', [])
            if len(chart_recommendations) < 2:
                print(f"⚠️ 경고: chart_recommendations가 부족합니다! (현재: {len(chart_recommendations)}개, 요구: 2개 이상)")
            
            # comparison_suggestions 검증
            comparison_suggestions = result.get('comparison_suggestions', [])
            if len(comparison_suggestions) < 2:
                print(f"⚠️ 경고: comparison_suggestions가 부족합니다! (현재: {len(comparison_suggestions)}개, 요구: 2-3개 이상)")
            
            if result.get('key_insights'):
                print(f"  - key_insights 샘플: {result.get('key_insights', [])[:2]}")
            if result.get('insights'):
                print(f"  - insights 샘플 (검증 후): {result.get('insights', [])[:2]}")
            
            # 통계 데이터 추가 (비동기 처리)
            loop = asyncio.get_running_loop()
            statistics = await loop.run_in_executor(
                None,
                lambda: self.stats_calculator.calculate(panels_data)
            )
            
            # chart_recommendations 검증 및 정리
            chart_recommendations_raw = result.get('chart_recommendations', [])
            chart_recommendations_validated = []
            for idx, chart in enumerate(chart_recommendations_raw):
                if not isinstance(chart, dict):
                    print(f"  ⚠️ chart_recommendations[{idx}]가 dict가 아닙니다: {type(chart)}")
                    continue
                
                # 필수 필드 확인
                chart_type = chart.get("type")
                title = chart.get("title")
                description = chart.get("description")
                category = chart.get("category")
                data_spec = chart.get("data_spec")
                
                # 최소한 type과 title이 있으면 포함
                if chart_type and title:
                    chart_recommendations_validated.append({
                        "type": chart_type,
                        "title": title,
                        "description": description or "",
                        "category": category or "기타",
                        "data_spec": data_spec or {}
                    })
                else:
                    print(f"  ⚠️ chart_recommendations[{idx}]: 필수 필드(type, title) 누락으로 제외")
            
            if len(chart_recommendations_validated) < 2:
                print(f"⚠️ 경고: chart_recommendations가 부족합니다! (현재: {len(chart_recommendations_validated)}개, 요구: 2개 이상)")
            
            result['chart_recommendations'] = chart_recommendations_validated
            
            # 비교군 정보 추가 및 검증
            comparison_groups = []
            comparison_suggestions_raw = result.get("comparison_suggestions", [])
            for idx, suggestion in enumerate(comparison_suggestions_raw):
                if not isinstance(suggestion, dict):
                    print(f"  ⚠️ comparison_suggestions[{idx}]가 dict가 아닙니다: {type(suggestion)}")
                    continue
                
                # 필수 필드 확인
                comp_type = suggestion.get("type")
                reason = suggestion.get("reason")
                query_suggestion = suggestion.get("query_suggestion")
                
                # 최소한 type이 있으면 포함
                if comp_type:
                    comparison_groups.append({
                        "type": comp_type,
                        "reason": reason or "",
                        "query_suggestion": query_suggestion or ""
                    })
                else:
                    print(f"  ⚠️ comparison_suggestions[{idx}]: 필수 필드(type) 누락으로 제외")
            
            if len(comparison_groups) < 2:
                print(f"⚠️ 경고: comparison_groups가 부족합니다! (현재: {len(comparison_groups)}개, 요구: 2-3개 이상)")
            
            # key_insights와 insights 분리 처리 (중복 제거 강화)
            key_insights = result.get("key_insights", [])
            detailed_insights = result.get("insights", [])
            
            print(f"📊 인사이트 처리 전:")
            print(f"  - key_insights 원본: {len(key_insights)}개")
            print(f"  - detailed_insights 원본: {len(detailed_insights)}개")
            
            # key_insights가 없으면 high significance 인사이트를 key_insights로 사용 (하위 호환성)
            if not key_insights:
                print(f"  ⚠️ key_insights가 비어있어 detailed_insights에서 high significance 추출 시도")
                key_insights = [
                    insight.get("finding", "") for insight in detailed_insights
                    if insight.get("significance") == "high"
                ][:5]
                print(f"  - 추출된 key_insights: {len(key_insights)}개")
            
            # detailed_insights가 비어있는 경우 경고
            if not detailed_insights:
                print(f"  ⚠️ 경고: detailed_insights가 비어있습니다!")
                print(f"  - result 전체: {json.dumps(result, ensure_ascii=False, indent=2)[:3000]}")
            
            # key_insights 내부 중복 제거 (같은 주제의 다른 수치 표현 제거)
            def normalize_text(text: str) -> str:
                """텍스트 정규화: 숫자, 특수문자 제거하여 핵심 키워드만 추출"""
                import re
                # 숫자와 퍼센트 제거
                text = re.sub(r'\d+\.?\d*%?', '', text)
                # 특수문자 제거
                text = re.sub(r'[^\w\s]', '', text)
                # 공백 정리
                text = ' '.join(text.split())
                return text.lower()
            
            # key_insights 중복 제거 (완화된 기준)
            unique_key_insights = []
            seen_normalized = set()
            for ki in key_insights:
                normalized = normalize_text(ki)
                # 핵심 키워드 추출 (예: "기아/현대 차량 선호도" -> "기아 현대 차량 선호도")
                if normalized and normalized not in seen_normalized:
                    # 유사한 텍스트가 이미 있는지 확인 (부분 일치 체크 - 완화: 4개 이상 키워드 겹치면 중복)
                    # 너무 엄격하면 유사하지만 다른 인사이트도 제거됨
                    is_similar = any(
                        len(set(normalized.split()) & set(existing.split())) >= 4  # 4개 이상 키워드 겹치면 중복 (3개 → 4개로 완화)
                        for existing in seen_normalized
                    )
                    if not is_similar:
                        unique_key_insights.append(ki)
                        seen_normalized.add(normalized)
            
            # 최대 개수 제한 완화 (5개 → 7개)
            key_insights = unique_key_insights[:7]  # 최대 7개 (5개 → 7개로 증가)
            
            # notable_findings는 key_insights와 중복되지 않는 모든 인사이트 (significance 제한 완화)
            notable_findings = []
            key_insights_lower = [normalize_text(ki) for ki in key_insights]
            for insight in detailed_insights:
                finding = insight.get("finding", "")
                # significance 제한 완화: "low"도 포함 (원래는 "high", "medium"만)
                if finding:  # significance 제한 제거
                    finding_normalized = normalize_text(finding)
                    # key_insights와 중복되지 않는 경우만 추가 (완화된 기준)
                    is_duplicate = any(
                        finding_normalized in ki or ki in finding_normalized or
                        len(set(finding_normalized.split()) & set(ki.split())) >= 4  # 4개 이상 키워드 겹치면 중복 (3개 → 4개로 완화)
                        for ki in key_insights_lower
                    )
                    if not is_duplicate:
                        notable_findings.append(finding)
            
            # notable_findings 내부 중복 제거 (완화된 기준)
            unique_notable_findings = []
            seen_notable = set()
            for nf in notable_findings:
                nf_normalized = normalize_text(nf)
                is_similar = any(
                    len(set(nf_normalized.split()) & set(existing.split())) >= 4  # 4개 이상 키워드 겹치면 중복 (3개 → 4개로 완화)
                    for existing in seen_notable
                )
                if not is_similar:
                    unique_notable_findings.append(nf)
                    seen_notable.add(nf_normalized)
            
            # 최대 개수 제한 완화 (10개 → 15개)
            notable_findings = unique_notable_findings[:15]  # 최대 15개 (10개 → 15개로 증가)
            
            # 최종 결과 확인
            final_result = {
                "summary": {
                    "total_panels": context.get("total_count", 0),
                    "key_insights": key_insights[:7],  # 최대 7개 (5개 → 7개로 증가)
                    "notable_findings": notable_findings[:15],  # 최대 15개 (10개 → 15개로 증가)
                },
                "statistics": statistics,
                "insights": detailed_insights,  # 모든 상세 인사이트 (필터링 없이 모두 포함)
                "chart_recommendations": result.get("chart_recommendations", []),
                "comparison_groups": comparison_groups,
            }
            
            # 최종 결과 상세 검증 및 로깅
            print(f"✅ 최종 분석 결과:")
            print(f"  - summary.key_insights: {len(final_result['summary']['key_insights'])}개")
            print(f"  - summary.notable_findings: {len(final_result['summary']['notable_findings'])}개")
            print(f"  - insights: {len(final_result['insights'])}개")
            print(f"  - chart_recommendations: {len(final_result['chart_recommendations'])}개")
            print(f"  - comparison_groups: {len(final_result['comparison_groups'])}개")
            
            # 각 인사이트 타입별 상세 검증
            print(f"\n📋 인사이트 타입별 상세 검증:")
            
            # 1. 상세 인사이트 (insights) 검증
            insights_with_recommendation = sum(1 for i in final_result['insights'] if i.get('recommendation'))
            insights_with_implication = sum(1 for i in final_result['insights'] if i.get('business_implication'))
            
            # 길이 일관성 검증
            finding_lengths = [len(i.get('finding', '')) for i in final_result['insights'] if i.get('finding')]
            implication_lengths = [len(i.get('business_implication', '')) for i in final_result['insights'] if i.get('business_implication')]
            recommendation_lengths = [len(i.get('recommendation', '')) for i in final_result['insights'] if i.get('recommendation')]
            
            print(f"  [상세 인사이트 (insights)]")
            print(f"    - 총 개수: {len(final_result['insights'])}개")
            print(f"    - 추천사항(recommendation) 포함: {insights_with_recommendation}개 ({insights_with_recommendation/len(final_result['insights'])*100:.1f}%)" if final_result['insights'] else "    - 추천사항 포함: 0개")
            print(f"    - 비즈니스 함의(business_implication) 포함: {insights_with_implication}개 ({insights_with_implication/len(final_result['insights'])*100:.1f}%)" if final_result['insights'] else "    - 비즈니스 함의 포함: 0개")
            
            # 길이 일관성 체크
            if finding_lengths:
                avg_finding_len = sum(finding_lengths) / len(finding_lengths)
                min_finding_len = min(finding_lengths)
                max_finding_len = max(finding_lengths)
                print(f"    - finding 길이: 평균 {avg_finding_len:.1f}자 (범위: {min_finding_len}-{max_finding_len}자)")
                if min_finding_len < 30:
                    print(f"    ⚠️ 경고: 일부 finding이 너무 짧습니다 (최소 30자 권장)")
            
            if implication_lengths:
                avg_impl_len = sum(implication_lengths) / len(implication_lengths)
                min_impl_len = min(implication_lengths)
                max_impl_len = max(implication_lengths)
                print(f"    - business_implication 길이: 평균 {avg_impl_len:.1f}자 (범위: {min_impl_len}-{max_impl_len}자)")
                if min_impl_len < 50:
                    print(f"    ⚠️ 경고: 일부 business_implication이 너무 짧습니다 (최소 50자 권장)")
            
            if recommendation_lengths:
                avg_rec_len = sum(recommendation_lengths) / len(recommendation_lengths)
                min_rec_len = min(recommendation_lengths)
                max_rec_len = max(recommendation_lengths)
                print(f"    - recommendation 길이: 평균 {avg_rec_len:.1f}자 (범위: {min_rec_len}-{max_rec_len}자)")
                if min_rec_len < 50:
                    print(f"    ⚠️ 경고: 일부 recommendation이 너무 짧습니다 (최소 50자 권장)")
            
            if len(final_result['insights']) < 10:
                print(f"    ⚠️ 경고: 최소 10개 이상 요구되나 {len(final_result['insights'])}개만 생성됨")
            
            # 2. 핵심 인사이트 (key_insights) 검증
            print(f"  [핵심 인사이트 (key_insights)]")
            print(f"    - 총 개수: {len(final_result['summary']['key_insights'])}개")
            if final_result['summary']['key_insights']:
                print(f"    - 샘플: {final_result['summary']['key_insights'][0][:50]}...")
            else:
                print(f"    ⚠️ 경고: 핵심 인사이트가 생성되지 않음")
            
            # 3. 특이사항 (notable_findings) 검증
            print(f"  [특이사항 (notable_findings)]")
            print(f"    - 총 개수: {len(final_result['summary']['notable_findings'])}개")
            if final_result['summary']['notable_findings']:
                print(f"    - 샘플: {final_result['summary']['notable_findings'][0][:50]}...")
            else:
                print(f"    ⚠️ 경고: 특이사항이 생성되지 않음")
            
            # 4. 차트 추천 (chart_recommendations) 검증
            print(f"  [차트 추천 (chart_recommendations)]")
            print(f"    - 총 개수: {len(final_result['chart_recommendations'])}개")
            if final_result['chart_recommendations']:
                for idx, chart in enumerate(final_result['chart_recommendations'][:3], 1):
                    print(f"    - [{idx}] {chart.get('type', 'N/A')}: {chart.get('title', 'N/A')}")
            else:
                print(f"    ⚠️ 경고: 차트 추천이 생성되지 않음")
            if len(final_result['chart_recommendations']) < 2:
                print(f"    ⚠️ 경고: 최소 2개 이상 요구되나 {len(final_result['chart_recommendations'])}개만 생성됨")
            
            # 5. 비교군 추천 (comparison_groups) 검증
            print(f"  [비교군 추천 (comparison_groups)]")
            print(f"    - 총 개수: {len(final_result['comparison_groups'])}개")
            if final_result['comparison_groups']:
                for idx, comp in enumerate(final_result['comparison_groups'][:3], 1):
                    print(f"    - [{idx}] {comp.get('type', 'N/A')}: {comp.get('query_suggestion', 'N/A')[:50]}...")
            else:
                print(f"    ⚠️ 경고: 비교군 추천이 생성되지 않음")
            if len(final_result['comparison_groups']) < 2:
                print(f"    ⚠️ 경고: 최소 2개 이상 요구되나 {len(final_result['comparison_groups'])}개만 생성됨")
            
            # 전체 요약
            print(f"\n📊 인사이트 생성 요약:")
            total_expected = 10 + 5 + 10 + 2 + 2  # insights(10) + key_insights(5) + notable_findings(10) + charts(2) + comparison(2) (시연 속도 최적화)
            total_actual = (
                len(final_result['insights']) +
                len(final_result['summary']['key_insights']) +
                len(final_result['summary']['notable_findings']) +
                len(final_result['chart_recommendations']) +
                len(final_result['comparison_groups'])
            )
            print(f"  - 예상 총 개수: {total_expected}개 이상")
            print(f"  - 실제 생성 개수: {total_actual}개")
            print(f"  - 생성률: {total_actual/total_expected*100:.1f}%" if total_expected > 0 else "  - 생성률: N/A")
            
            return final_result
        except Exception as e:
            print(f"❌ LLM 분석 실패: {e}")
            print(f"  - 패널 수: {len(panels_data)}개")
            print(f"  - 컨텍스트 키: {list(context.keys()) if isinstance(context, dict) else 'N/A'}")
            import traceback
            traceback.print_exc()
            # 폴백: 통계만 반환 (비동기 처리)
            loop = asyncio.get_running_loop()
            statistics = await loop.run_in_executor(
                None,
                lambda: self.stats_calculator.calculate(panels_data)
            )
            fallback_result = {
                "summary": {
                    "total_panels": context.get("total_count", 0),
                    "key_insights": [],
                    "notable_findings": [],
                },
                "statistics": statistics,
                "insights": [],
                "chart_recommendations": [],
                "comparison_groups": [],
            }
            print(f"⚠️ 폴백 결과 반환 (인사이트 없음)")
            return fallback_result

