"""
분석 API 라우터

RAG 기반 패널 데이터 분석 엔드포인트
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.database import get_db
from backend.services.analysis_service import AnalysisService

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ===== 요청/응답 모델 =====
class AnalyzeRequest(BaseModel):
    """분석 요청 모델"""
    panel_ids: List[str]
    analysis_type: str = "comprehensive"  # "basic" | "comprehensive" | "custom"
    focus_areas: Optional[List[str]] = None  # ["demographics", "economic", "digital", "lifestyle"]
    include_comparison: bool = True
    include_charts: bool = True
    query: Optional[str] = None  # 원본 질의
    requested_count: Optional[int] = None  # 질의에서 추출한 명수 (명시된 경우만)


class InsightItem(BaseModel):
    """인사이트 항목"""
    category: str
    finding: str
    significance: str  # "high" | "medium" | "low"
    business_implication: Optional[str] = None
    recommendation: Optional[str] = None


class ChartRecommendation(BaseModel):
    """차트 추천"""
    type: str
    title: str
    description: str
    category: str
    data_spec: dict


class ComparisonGroup(BaseModel):
    """비교군"""
    type: str  # "similar" | "contrast" | "complement"
    reason: str
    query_suggestion: Optional[str] = None


class AnalysisResponse(BaseModel):
    """분석 응답 모델"""
    summary: dict
    statistics: dict
    insights: List[InsightItem]
    chart_recommendations: List[ChartRecommendation]
    comparison_groups: List[ComparisonGroup]


# ===== API 엔드포인트 =====
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_panels(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db)
):
    """패널 데이터 분석
    
    Args:
        request: 분석 요청 (패널 ID 리스트, 분석 타입 등)
        db: DB 세션
    
    Returns:
        분석 결과 (인사이트, 차트 추천, 비교군 등)
    """
    if not request.panel_ids:
        raise HTTPException(status_code=400, detail="panel_ids가 비어있습니다.")
    
    # [수정 1] 제한 해제 (서비스 내부에서 샘플링하므로 10만 개도 OK)
    # 다만, HTTP 요청 바디 크기 보호를 위해 10만 개 정도로 넉넉하게 잡습니다.
    if len(request.panel_ids) > 100000:
        raise HTTPException(status_code=400, detail="한 번에 분석 가능한 패널은 최대 10만 명입니다.")
    
    try:
        analysis_service = AnalysisService()
        result = await analysis_service.analyze_panels(
            panel_ids=request.panel_ids,
            analysis_type=request.analysis_type,
            focus_areas=request.focus_areas,
            query=request.query,
            requested_count=request.requested_count,
            session=db,
        )
        
        # 디버깅: 서비스 결과 확인
        print(f"📤 분석 서비스 결과:")
        print(f"  - summary: {result.get('summary', {})}")
        print(f"  - insights 개수: {len(result.get('insights', []))}")
        print(f"  - key_insights 개수: {len(result.get('summary', {}).get('key_insights', []))}")
        
        # [수정 2] 안전한 Pydantic 모델 변환
        # 서비스에서 리턴한 dict가 모델과 100% 안 맞을 수 있으므로 안전장치 추가
        # (예: None이 오면 안 되는데 None이 온 경우 등)
        
        def safe_insight(item):
            """필수 필드 누락 시 기본값 처리"""
            if not isinstance(item, dict):
                item = {}
            return InsightItem(
                category=item.get("category", "기타"),
                finding=item.get("finding", "내용 없음"),
                significance=item.get("significance", "medium"),
                business_implication=item.get("business_implication") or "",
                recommendation=item.get("recommendation") or ""
            )
        
        def safe_chart(item):
            """차트 추천 안전 변환"""
            if not isinstance(item, dict):
                item = {}
            return ChartRecommendation(
                type=item.get("type", "bar"),
                title=item.get("title", "차트"),
                description=item.get("description", ""),
                category=item.get("category", "기타"),
                data_spec=item.get("data_spec", {})
            )
        
        def safe_comparison(item):
            """비교군 안전 변환"""
            if not isinstance(item, dict):
                item = {}
            return ComparisonGroup(
                type=item.get("type", "similar"),
                reason=item.get("reason", ""),
                query_suggestion=item.get("query_suggestion") or ""
            )
        
        # 응답 모델로 변환 (안전장치 적용)
        response = AnalysisResponse(
            summary=result.get("summary", {}),
            statistics=result.get("statistics", {}),
            insights=[safe_insight(i) for i in result.get("insights", [])],
            chart_recommendations=[safe_chart(c) for c in result.get("chart_recommendations", [])],
            comparison_groups=[safe_comparison(g) for g in result.get("comparison_groups", [])],
        )
        
        # 디버깅: 최종 응답 확인
        print(f"📤 최종 API 응답:")
        print(f"  - summary.key_insights: {len(response.summary.get('key_insights', []))}개")
        print(f"  - insights: {len(response.insights)}개")
        
        return response
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ 분석 라우터 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

