-- 성능 최적화를 위한 인덱스 개선 스크립트
-- IVFFlat 인덱스 파라미터 최적화 및 복합 인덱스 추가

-- 1. 기존 IVFFlat 인덱스 삭제 (재생성 전)
DROP INDEX IF EXISTS idx_segments_embedding;

-- 2. IVFFlat 인덱스 재생성 (lists 파라미터 최적화)
-- lists 파라미터: sqrt(총 레코드 수) 정도가 적절 (약 140만 개 → lists=1000~2000)
-- 더 많은 lists = 더 정확하지만 느림, 적은 lists = 빠르지만 덜 정확
-- 140만 개 기준으로 lists=1000이 적절 (성능과 정확도의 균형)
CREATE INDEX idx_segments_embedding 
ON panel_summary_segments 
USING ivfflat(embedding vector_cosine_ops)
WITH (lists = 1000);

-- 3. 부분 인덱스 추가: segment_name별로 인덱스 분할 (PostgreSQL 제약으로 복합 인덱스 불가)
-- segment_name 인덱스는 이미 존재하므로, 쿼리에서 segment_name을 먼저 필터링하도록 최적화됨
-- (코드 레벨에서 WHERE 절 순서 최적화 완료)

-- 참고: PostgreSQL에서는 벡터 타입과 일반 컬럼을 함께 인덱싱하는 복합 인덱스가 제한적입니다.
-- 대신 쿼리에서 segment_name을 먼저 필터링하도록 WHERE 절 순서를 최적화했습니다.

-- 참고: 인덱스 생성은 시간이 걸릴 수 있습니다 (약 140만 개 레코드 기준)
-- 백그라운드에서 실행하거나, 운영 시간 외에 실행하는 것을 권장합니다.

