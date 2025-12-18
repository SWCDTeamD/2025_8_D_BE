"""Search Router (v2) - Optimized"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.database import get_db
from backend.services.search_service import (
    NLQueryRequest,
    SearchResponse,
    natural_language_panel_search,
)


router = APIRouter(prefix="/search", tags=["search"])


@router.post("/nl", response_model=SearchResponse)
async def search_by_natural_language(
    payload: NLQueryRequest,
    db: AsyncSession = Depends(get_db)  # [개선 1] DB 세션 주입 (연결 풀 관리)
) -> SearchResponse:
    """자연어 질의를 통한 패널 데이터 검색 엔드포인트"""
    
    # [개선 2] 빈 검색어 즉시 차단 (불필요한 리소스 낭비 방지)
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="검색어를 입력해주세요.")
    
    try:
        # [개선 3] 서비스 함수에 세션 전달
        result = await natural_language_panel_search(payload, session=db)
        return result
    
    except HTTPException as he:
        # HTTPException은 그대로 전달
        raise he
    except Exception as e:
        # [개선 4] 에러 로그 출력 및 명확한 에러 메시지 반환
        print(f"❌ 검색 API 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # 사용자에게는 "서버 오류"라고 알리고, 상세 내용은 로그로 남김
        raise HTTPException(
            status_code=500, 
            detail=f"검색 처리 중 오류가 발생했습니다: {str(e)}"
        )
