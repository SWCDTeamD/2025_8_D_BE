-- 비정형 검색 최적화를 위한 인덱스 개선 스크립트 (수동 실행용)
-- 벡터 검색과 FTS 검색 성능 향상

-- ==========================================
-- 주의: 인덱스 생성 전에 maintenance_work_mem 증가 필요
-- ==========================================
-- SET maintenance_work_mem = '1GB';  -- 인덱스 생성 전에 실행 (세션 레벨)
-- 또는 PostgreSQL 설정 파일(postgresql.conf)에서 수정

-- ==========================================
-- 1. 벡터 인덱스 최적화 (IVFFlat)
-- ==========================================

-- 기존 인덱스 삭제
DROP INDEX IF EXISTS idx_segments_embedding;

-- IVFFlat 인덱스 재생성 (lists 파라미터 최적화)
-- 성능 우선: lists=1500 (정확도 약간 희생, 속도 향상)
-- 주의: 인덱스 생성은 시간이 걸릴 수 있습니다 (약 140만 개 레코드 기준)
CREATE INDEX idx_segments_embedding 
ON panel_summary_segments 
USING ivfflat(embedding vector_cosine_ops)
WITH (lists = 1500);

-- ==========================================
-- 2. 복합 인덱스 (segment_name + panel_id)
-- ==========================================

-- 세그먼트 필터링과 패널 조회를 위한 복합 인덱스
CREATE INDEX IF NOT EXISTS idx_segments_name_panel 
ON panel_summary_segments(segment_name, panel_id);

-- ==========================================
-- 3. 통계 정보 업데이트 (쿼리 플래너 최적화)
-- ==========================================

ANALYZE panel_summary_segments;

