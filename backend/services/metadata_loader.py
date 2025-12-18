"""
메타데이터 로더

DB에서 컬럼 메타데이터, 라벨 값, 카테고리 그룹을 조회하는 모듈
"""

from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.repositories.database import AsyncSessionLocal


class MetadataLoader:
    """메타데이터 조회 클래스"""
    
    async def load_column_metadata(
        self,
        focus_areas: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Dict[str, Any]]:
        """컬럼 메타데이터 조회
        
        Args:
            focus_areas: 분석 대상 카테고리 그룹 (예: ["demographics", "economic"])
            session: DB 세션 (없으면 새로 생성)
        
        Returns:
            {column_name: {name_ko, name_en, type, description, ...}}
        """
        close_session = False
        if session is None:
            session = AsyncSessionLocal()
            close_session = True
        
        try:
            # focus_areas가 있으면 해당 그룹의 컬럼만 조회
            if focus_areas:
                query = text("""
                    SELECT DISTINCT cm.*
                    FROM column_metadata cm
                    JOIN category_group_columns cgc ON cm.column_name = cgc.column_name
                    JOIN category_groups cg ON cgc.group_key = cg.group_key
                    WHERE cg.group_key = ANY(:focus_areas)
                    ORDER BY cm.analysis_priority DESC, cm.column_name
                """)
                result = await session.execute(query, {"focus_areas": focus_areas})
            else:
                # 모든 컬럼 메타데이터 조회
                query = text("""
                    SELECT * FROM column_metadata
                    ORDER BY analysis_priority DESC, column_name
                """)
                result = await session.execute(query)
            
            metadata = {}
            for row in result:
                metadata[row.column_name] = {
                    "name_ko": row.name_ko,
                    "name_en": row.name_en,
                    "type": row.type,
                    "description": row.description,
                    "unit": row.unit,
                    "range_min": row.range_min,
                    "range_max": row.range_max,
                    "analysis_priority": row.analysis_priority,
                    "chart_types": row.chart_types or [],
                    "statistics": row.statistics or [],
                }
            
            return metadata
        finally:
            if close_session:
                await session.close()
    
    async def load_label_values(
        self,
        column_names: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, List[str]]:
        """라벨 값 조회
        
        Args:
            column_names: 조회할 컬럼명 리스트 (없으면 모든 컬럼)
            session: DB 세션
        
        Returns:
            {column_name: [value1, value2, ...]}
        """
        close_session = False
        if session is None:
            session = AsyncSessionLocal()
            close_session = True
        
        try:
            if column_names:
                query = text("""
                    SELECT column_name, value, display_order
                    FROM label_values
                    WHERE column_name = ANY(:column_names) AND is_active = TRUE
                    ORDER BY column_name, display_order
                """)
                result = await session.execute(query, {"column_names": column_names})
            else:
                query = text("""
                    SELECT column_name, value, display_order
                    FROM label_values
                    WHERE is_active = TRUE
                    ORDER BY column_name, display_order
                """)
                result = await session.execute(query)
            
            label_values: Dict[str, List[str]] = {}
            for row in result:
                if row.column_name not in label_values:
                    label_values[row.column_name] = []
                label_values[row.column_name].append(row.value)
            
            return label_values
        finally:
            if close_session:
                await session.close()
    
    async def load_category_groups(
        self,
        group_keys: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Dict[str, Any]]:
        """카테고리 그룹 조회
        
        Args:
            group_keys: 조회할 그룹 키 리스트 (없으면 모든 그룹)
            session: DB 세션
        
        Returns:
            {group_key: {name_ko, name_en, description, fields, ...}}
        """
        close_session = False
        if session is None:
            session = AsyncSessionLocal()
            close_session = True
        
        try:
            if group_keys:
                query = text("""
                    SELECT cg.*, 
                           ARRAY_AGG(cgc.column_name ORDER BY cgc.display_order) as fields
                    FROM category_groups cg
                    LEFT JOIN category_group_columns cgc ON cg.group_key = cgc.group_key
                    WHERE cg.group_key = ANY(:group_keys)
                    GROUP BY cg.group_key
                    ORDER BY cg.group_key
                """)
                result = await session.execute(query, {"group_keys": group_keys})
            else:
                query = text("""
                    SELECT cg.*, 
                           ARRAY_AGG(cgc.column_name ORDER BY cgc.display_order) as fields
                    FROM category_groups cg
                    LEFT JOIN category_group_columns cgc ON cg.group_key = cgc.group_key
                    GROUP BY cg.group_key
                    ORDER BY cg.group_key
                """)
                result = await session.execute(query)
            
            groups = {}
            for row in result:
                groups[row.group_key] = {
                    "name_ko": row.name_ko,
                    "name_en": row.name_en,
                    "description": row.description,
                    "analysis_focus": row.analysis_focus or [],
                    "fields": row.fields or [],
                }
            
            return groups
        finally:
            if close_session:
                await session.close()
    
    async def load_metadata(
        self,
        focus_areas: Optional[List[str]] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """전체 메타데이터 조회 (통합)
        
        Args:
            focus_areas: 분석 대상 카테고리 그룹
            session: DB 세션
        
        Returns:
            {
                "column_metadata": {...},
                "label_values": {...},
                "category_groups": {...}
            }
        """
        column_metadata = await self.load_column_metadata(focus_areas, session)
        
        # focus_areas가 있으면 해당 그룹의 컬럼만 조회
        column_names = None
        if focus_areas:
            category_groups = await self.load_category_groups(focus_areas, session)
            column_names = []
            for group in category_groups.values():
                column_names.extend(group.get("fields", []))
            column_names = list(set(column_names)) if column_names else None
        
        label_values = await self.load_label_values(column_names, session)
        category_groups = await self.load_category_groups(focus_areas, session)
        
        return {
            "column_metadata": column_metadata,
            "label_values": label_values,
            "category_groups": category_groups,
        }

