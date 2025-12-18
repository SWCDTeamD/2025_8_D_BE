-- 비정형 검색 최적화를 위한 인덱스 개선 스크립트
-- 벡터 검색과 FTS 검색 성능 향상

-- ==========================================
-- 0. 메모리 설정 증가 (인덱스 생성 전 필수)
-- ==========================================

-- 인덱스 생성에 필요한 메모리 증가
-- 주의: 세션 레벨 설정이므로 각 연결마다 설정 필요
SET maintenance_work_mem = '1GB';

-- ==========================================
-- 1. 벡터 인덱스 최적화 (IVFFlat)
-- ==========================================

-- 기존 인덱스 삭제
DROP INDEX IF EXISTS idx_segments_embedding;

-- IVFFlat 인덱스 재생성 (lists 파라미터 최적화)
-- 현재 데이터 규모에 맞게 lists 파라미터 조정
-- lists = sqrt(총 레코드 수) 정도가 적절
-- 더 많은 lists = 더 정확하지만 느림, 적은 lists = 빠르지만 덜 정확
-- 현재 데이터: 약 201,554개 행
-- sqrt(201554) ≈ 449
-- 권장 범위: 359 ~ 673
-- 균형잡힌 선택: lists=500 (정확도와 속도의 균형)
CREATE INDEX idx_segments_embedding 
ON panel_summary_segments 
USING ivfflat(embedding vector_cosine_ops)
WITH (lists = 500);

-- ==========================================
-- 2. 세그먼트별 부분 인덱스 (벡터 검색 최적화)
-- ==========================================

-- 자주 검색되는 세그먼트에 대한 부분 인덱스 생성
-- segment_name을 먼저 필터링하면 벡터 검색 범위가 줄어들어 속도 향상

-- 주요 세그먼트별 부분 인덱스 (선택적)
-- 주의: 부분 인덱스는 데이터가 많은 세그먼트에만 유용
-- CREATE INDEX idx_segments_embedding_fitness 
-- ON panel_summary_segments 
-- USING ivfflat(embedding vector_cosine_ops)
-- WITH (lists = 500)
-- WHERE segment_name = 'FITNESS_MANAGEMENT_METHOD';

-- ==========================================
-- 3. FTS 인덱스 최적화 (GIN)
-- ==========================================

-- 기존 FTS 인덱스 확인 및 재생성 (필요시)
-- GIN 인덱스는 이미 최적화되어 있지만, 필요시 재생성
-- DROP INDEX IF EXISTS idx_segments_ts_vector_korean;
-- CREATE INDEX idx_segments_ts_vector_korean 
-- ON panel_summary_segments 
-- USING GIN(ts_vector_korean);

-- ==========================================
-- 4. 복합 인덱스 (segment_name + panel_id)
-- ==========================================

-- 세그먼트 필터링과 패널 조회를 위한 복합 인덱스
-- 벡터 검색 후 패널별 그룹화 시 성능 향상
CREATE INDEX IF NOT EXISTS idx_segments_name_panel 
ON panel_summary_segments(segment_name, panel_id);

-- ==========================================
-- 5. 통계 정보 업데이트 (쿼리 플래너 최적화)
-- ==========================================

-- PostgreSQL 쿼리 플래너가 최적의 실행 계획을 선택하도록 통계 정보 업데이트
ANALYZE panel_summary_segments;

-- ==========================================
-- 참고사항
-- ==========================================

-- 1. IVFFlat 인덱스는 데이터가 변경되면 재생성이 필요할 수 있습니다.
--    대량 데이터 삽입/업데이트 후에는 인덱스 재생성을 고려하세요.

-- 2. lists 파라미터는 데이터 규모에 따라 조정이 필요합니다.
--    - 데이터가 증가하면 lists도 증가시켜야 정확도 유지
--    - 현재 설정(500)은 약 20만 개 레코드 기준
--    - 데이터가 2배 증가하면 lists도 약 1.4배 증가 권장 (sqrt 비례)

-- 3. 부분 인덱스는 특정 세그먼트가 매우 자주 검색될 때만 유용합니다.
--    일반적으로는 전체 인덱스가 더 효율적입니다.

-- 4. 인덱스 생성은 시간이 걸릴 수 있습니다 (약 20만 개 레코드 기준, 예상 5-10분).
--    운영 시간 외에 실행하는 것을 권장합니다.

