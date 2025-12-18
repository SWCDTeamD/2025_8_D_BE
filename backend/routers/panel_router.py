"""Panel Router (v2)"""

from typing import Any, AsyncGenerator, Dict, List

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.database import AsyncSessionLocal


router = APIRouter(prefix="/panels", tags=["panels"])


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """DB 세션"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


@router.get("/")
async def list_panels(limit: int = 50) -> List[Dict[str, Any]]:
    """패널 리스트 조회"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT panel_id, gender, age, region_city, region_gu, marital_status,
                       children_count, occupation, panel_summary_text
                FROM panels
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        rows = result.fetchall()
        
        columns = [
            "panel_id", "gender", "age", "region_city", "region_gu", "marital_status",
            "children_count", "occupation", "panel_summary_text"
        ]
        
        return [dict(zip(columns, row)) for row in rows]


@router.get("/{panel_id}")
async def get_panel(panel_id: str) -> Dict[str, Any]:
    """단일 패널의 설문 응답 세그먼트만 조회 (최적화: 기본 패널 데이터는 프론트엔드에서 이미 가지고 있음)"""
    async with AsyncSessionLocal() as session:
        # 세그먼트 정보만 조회 (JOIN 불필요, 속도 향상)
        result = await session.execute(
            text("""
                SELECT 
                    segment_name, summary_text
                FROM panel_summary_segments
                WHERE panel_id = :panel_id
                ORDER BY segment_name
            """),
            {"panel_id": panel_id}
        )
        rows = result.fetchall()
        
        # 패널 존재 여부 확인 (세그먼트가 없어도 패널은 존재할 수 있음)
        panel_check = await session.execute(
            text("SELECT panel_id FROM panels WHERE panel_id = :panel_id"),
            {"panel_id": panel_id}
        )
        if not panel_check.fetchone():
            raise HTTPException(status_code=404, detail="Panel not found")
        
        # 세그먼트 정보 수집
        summary_segments = {}
        for row in rows:
            segment_name = row[0]
            summary_text = row[1]
            if segment_name and summary_text:
                summary_segments[segment_name] = summary_text
        
        return {
            "panel_id": panel_id,
            "summary_segments": summary_segments
        }
