from __future__ import annotations
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from backend.repositories.database import AsyncSessionLocal
from backend.repositories.morphology_utils import normalize_query_morphology
import os

# SentenceTransformers (로컬 모델)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_EMBEDDING = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _HAS_EMBEDDING = False

# OpenAI 임베딩 (선택적)
try:
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

_EMBEDDING_MODEL: Optional[Any] = None
_OPENAI_CLIENT: Optional[Any] = None

def get_embedding_model():
    """임베딩 모델 반환
    
    환경 변수 EMBEDDING_MODEL로 선택:
    - "openai" → OpenAI text-embedding-3-small (1536차원)
    - "kosimcse" or None → KoSimCSE (768차원, 기본값)
    
    주의: DB에 저장된 임베딩과 동일한 모델을 사용해야 벡터 검색이 정상 작동함!
    """
    global _EMBEDDING_MODEL, _OPENAI_CLIENT
    
    embedding_model_type = os.getenv("EMBEDDING_MODEL", "kosimcse").lower()
    
    if embedding_model_type == "openai":
        # OpenAI 임베딩 사용
        if not _HAS_OPENAI or OpenAI is None:
            print("⚠️ OpenAI 라이브러리가 설치되지 않았습니다. pip install openai")
            return None
        
        if _OPENAI_CLIENT is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
                return None
            _OPENAI_CLIENT = OpenAI(api_key=api_key)
            print("✅ OpenAI 임베딩 모델 사용 (text-embedding-3-small, 1536차원)")
        
        return _OPENAI_CLIENT
    else:
        # KoSimCSE 임베딩 사용 (기본값)
        if not _HAS_EMBEDDING or SentenceTransformer is None:
            return None
        if _EMBEDDING_MODEL is None:
            # ETL 파이프라인과 동일한 모델 사용 (BM-K/KoSimCSE-roberta-multitask)
            _EMBEDDING_MODEL = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
            print("✅ KoSimCSE 임베딩 모델 사용 (BM-K/KoSimCSE-roberta-multitask, 768차원)")
        return _EMBEDDING_MODEL

class DocumentRepository:
    def __init__(self, session: Optional[AsyncSession] = None) -> None:
        self.session = session

    async def _get_session(self) -> AsyncSession:
        """세션 가져오기 (하위 호환성 유지, 하지만 사용하지 않는 것을 권장)"""
        if self.session:
            return self.session
        return AsyncSessionLocal()

    async def semantic_search(
        self, 
        query: str, 
        limit: int, 
        exclude_negative: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        segment_filter: Optional[List[str]] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """벡터 유사도 검색
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            exclude_negative: True면 부정 표현이 포함된 세그먼트 제외 (예: "키워본 적이 없다")
            segment_filter: 검색할 세그먼트 목록 (None이면 모든 세그먼트 검색)
        """
        # [수정 1] 세션 관리 로직 추가
        # self.session이 있으면 쓰고, 없으면 새로 만들고 '반드시 닫는다'
        session = self.session if self.session else AsyncSessionLocal()
        should_close = self.session is None
        
        embedding_model = get_embedding_model()
        if not embedding_model:
            if should_close:
                await session.close()
            print("  ⚠️ 임베딩 모델이 없어 벡터 검색을 건너뜁니다.")
            return []
        
        try:
            # 임베딩 생성 (모델 타입에 따라 다름)
            embedding_model_type = os.getenv("EMBEDDING_MODEL", "kosimcse").lower()
            
            if embedding_model_type == "openai":
                # OpenAI 임베딩 API 호출
                response = embedding_model.embeddings.create(
                    model="text-embedding-3-small",  # 1536차원
                    input=query
                )
                query_embedding = response.data[0].embedding
            else:
                # KoSimCSE (로컬 모델)
                query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
            
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            
            # 부정 표현 필터링 조건 추가 (메타데이터 기반)
            negative_filter = ""
            negative_filter_params = {}  # SQL 인젝션 방지를 위한 파라미터 딕셔너리
            if exclude_negative:
                # 메타데이터에서 제공된 exclude_patterns 우선 사용
                if exclude_patterns:
                    # SQL 인젝션 방지: 파라미터 바인딩 사용
                    pattern_conditions = " OR ".join([
                        f"summary_text LIKE :neg_pattern_{i}"
                        for i in range(len(exclude_patterns))
                    ])
                    negative_filter = f"""
                      AND NOT ({pattern_conditions})
                    """
                    # 패턴을 파라미터로 추가 (% 포함하여 전달)
                    for i, pattern in enumerate(exclude_patterns):
                        negative_filter_params[f"neg_pattern_{i}"] = f"%{pattern}%"
                else:
                    # 기본 패턴 (하위 호환성)
                    positive_keywords = ["키우는", "키운", "키워", "보유", "있", "한다", "중이다"]
                    negative_keywords = ["없다", "없음", "안", "못", "하지 않", "하지 않는다"]
                    
                    # 쿼리에 긍정 키워드가 있고 부정 키워드가 없으면 부정 표현 제외
                    has_positive = any(kw in query for kw in positive_keywords)
                    has_negative = any(kw in query for kw in negative_keywords)
                    
                    if has_positive and not has_negative:
                        # 부정 표현이 포함된 세그먼트 제외 (강화)
                        negative_filter = """
                          AND NOT (
                            summary_text LIKE '%없다%' 
                            OR summary_text LIKE '%없음%'
                            OR summary_text LIKE '%키워본 적이 없다%'
                            OR summary_text LIKE '%키운 적이 없다%'
                            OR summary_text LIKE '%하지 않는다%'
                            OR summary_text LIKE '%하지 않음%'
                            OR summary_text LIKE '%받지 않는다%'
                            OR summary_text LIKE '%이용하지 않는다%'
                            OR summary_text LIKE '%사용하지 않는다%'
                            OR summary_text LIKE '%선호하지 않는다%'
                            OR summary_text LIKE '%안 한다%'
                            OR summary_text LIKE '%안한다%'
                          )
                        """
            
            # 세그먼트 필터 조건
            segment_filter_clause = ""
            if segment_filter and len(segment_filter) > 0:
                segment_filter_clause = "AND segment_name = ANY(:segment_filter_array)"
            
            # 유사도 임계값 설정
            if min_similarity is None:
                try:
                    min_similarity = float(os.getenv("EMBEDDING_MIN_SIM", "0.60"))
                except Exception:
                    min_similarity = 0.60
            
            # 쿼리 최적화: segment_filter가 있으면 먼저 필터링 (인덱스 활용)
            # WHERE 절 순서 최적화: segment_name 필터를 먼저 적용하면 벡터 검색 범위 축소
            if segment_filter and len(segment_filter) > 0:
                # segment_name을 먼저 필터링하여 벡터 검색 범위 축소
                sql_query = text(f"""
                    SELECT 
                        panel_id,
                        segment_name,
                        summary_text,
                        1 - (embedding <=> CAST(:embedding_str AS vector)) as similarity
                    FROM panel_summary_segments
                    WHERE segment_name = ANY(:segment_filter_array)
                      AND embedding IS NOT NULL
                      {negative_filter}
                      AND (1 - (embedding <=> CAST(:embedding_str AS vector))) >= :min_sim
                    ORDER BY embedding <=> CAST(:embedding_str AS vector)
                    LIMIT :limit_val
                """)
            else:
                # segment_filter가 없으면 기존 방식 사용
                sql_query = text(f"""
                    SELECT 
                        panel_id,
                        segment_name,
                        summary_text,
                        1 - (embedding <=> CAST(:embedding_str AS vector)) as similarity
                    FROM panel_summary_segments
                    WHERE embedding IS NOT NULL
                      {negative_filter}
                      AND (1 - (embedding <=> CAST(:embedding_str AS vector))) >= :min_sim
                    ORDER BY embedding <=> CAST(:embedding_str AS vector)
                    LIMIT :limit_val
                """)
            
            # limit 최적화: 불필요하게 많이 가져오지 않음
            # 벡터 검색은 정확도가 높으므로 limit * 2 정도면 충분
            # RRF 통합을 위해 약간 여유를 두되, 과도하게 많이 가져오지 않음
            effective_limit = min(limit * 2, 10000)  # 최대 10,000개로 제한
            
            params = {
                "embedding_str": embedding_str,
                "limit_val": effective_limit,
                "min_sim": min_similarity
            }
            if segment_filter and len(segment_filter) > 0:
                params["segment_filter_array"] = segment_filter
            # SQL 인젝션 방지: 부정 패턴 파라미터 추가
            params.update(negative_filter_params)
            
            try:
                result = await session.execute(sql_query, params)
                rows = result.fetchall()
            except Exception as e:
                # 에러 발생 시 트랜잭션 롤백
                await session.rollback()
                print(f"  ❌ 벡터 검색 실행 오류: {e}")
                raise
            
            if not rows:
                print(f"  ⚠️ 벡터 검색 결과가 없습니다. (DB에 embedding 데이터가 있는지 확인 필요)")
            
            # 패널별 점수 처리: 여러 세그먼트에서 매칭되면 점수 중첩 (합산)
            # 여러 세그먼트에서 매칭되는 것은 더 관련성이 높다는 신호
            panel_scores: Dict[str, List[float]] = {}  # 패널별 점수 리스트
            panel_segments: Dict[str, Dict[str, float]] = {}  # 패널별 세그먼트별 점수 (가장 높은 점수의 세그먼트 선택용)
            
            for panel_id, segment_name, summary_text, similarity in rows:
                if panel_id:
                    panel_id_str = str(panel_id)
                    similarity_float = float(similarity) if similarity else 0.0
                    if panel_id_str not in panel_scores:
                        panel_scores[panel_id_str] = []
                    panel_scores[panel_id_str].append(similarity_float)
                    
                    # 세그먼트별 최고 점수 추적
                    if panel_id_str not in panel_segments:
                        panel_segments[panel_id_str] = {}
                    seg_name = segment_name or ""
                    if seg_name not in panel_segments[panel_id_str] or similarity_float > panel_segments[panel_id_str][seg_name]:
                        panel_segments[panel_id_str][seg_name] = similarity_float
            
            # 패널별 최종 점수 계산: 평균 + 최고값 가중합 (여러 세그먼트 매칭 보너스)
            best_by_panel: Dict[str, float] = {}
            best_segment_by_panel: Dict[str, Optional[str]] = {}  # 패널별 대표 세그먼트
            
            for panel_id_str, scores in panel_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    # 평균 70% + 최고값 30% (여러 세그먼트 매칭 시 보너스)
                    # 세그먼트가 많을수록 평균이 높아져서 점수 상승
                    final_score = avg_score * 0.7 + max_score * 0.3
                    # 세그먼트 개수 보너스 (최대 1.2배)
                    segment_bonus = min(1.0 + (len(scores) - 1) * 0.1, 1.2)
                    best_by_panel[panel_id_str] = final_score * segment_bonus
                    
                    # 가장 높은 점수의 세그먼트 선택
                    if panel_id_str in panel_segments:
                        best_seg = max(panel_segments[panel_id_str].items(), key=lambda x: x[1])[0]
                        best_segment_by_panel[panel_id_str] = best_seg if best_seg else None
            
            sorted_items = sorted(best_by_panel.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [
                {
                    "panel_id": pid, 
                    "score": score, 
                    "source": "vector",
                    "segment_name": best_segment_by_panel.get(pid)  # 세그먼트 정보 추가
                }
                for pid, score in sorted_items
            ]
        except Exception as e:
            print(f"  ❌ 벡터 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # [중요] 여기서 세션을 닫아줍니다!
            if should_close:
                await session.close()

    async def fulltext_search(
        self, 
        query: str, 
        limit: int, 
        use_or: bool = False,  # 하위 호환성을 위해 유지 (실제로는 무시됨)
        exclude_negative: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        segment_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Full-Text Search (FTS) 검색
        
        Args:
            query: 검색 쿼리
            limit: 반환할 최대 결과 수
            use_or: (deprecated) 하위 호환성을 위해 유지하지만 항상 AND 연산 사용
            exclude_negative: 부정 표현 제외 여부
            exclude_patterns: 제외할 패턴 목록 (메타데이터 기반)
            segment_filter: 검색할 세그먼트 목록
            
        Returns:
            검색 결과 리스트
            
        Note:
            - AND 연산만 사용 (모든 키워드 포함 필요)
            - 부정어는 FTS 쿼리에 직접 포함 (LIKE 사후 필터링 대신)
            - 한국어 형태소 분석 ('korean' dictionary 사용)
        """
        # [수정 1] 세션 관리 로직 추가
        session = self.session if self.session else AsyncSessionLocal()
        should_close = self.session is None
        
        from sqlalchemy import bindparam
        try:
            # 부정 표현 필터링 조건 (메타데이터 기반)
            negative_filter = ""
            negative_filter_params = {}  # SQL 인젝션 방지를 위한 파라미터 딕셔너리
            if exclude_negative:
                # 메타데이터에서 제공된 exclude_patterns 우선 사용
                if exclude_patterns:
                    # SQL 인젝션 방지: 파라미터 바인딩 사용
                    pattern_conditions = " OR ".join([
                        f"summary_text LIKE :neg_pattern_{i}"
                        for i in range(len(exclude_patterns))
                    ])
                    negative_filter = f"""
                      AND NOT ({pattern_conditions})
                    """
                    # 패턴을 파라미터로 추가 (% 포함하여 전달)
                    for i, pattern in enumerate(exclude_patterns):
                        negative_filter_params[f"neg_pattern_{i}"] = f"%{pattern}%"
                else:
                    # 기본 패턴 (하위 호환성)
                    positive_keywords = ["키우는", "키운", "키워", "보유", "있", "한다", "중이다"]
                    negative_keywords = ["없다", "없음", "안", "못", "하지 않", "하지 않는다"]
                    
                    has_positive = any(kw in query for kw in positive_keywords)
                    has_negative = any(kw in query for kw in negative_keywords)
                    
                    if has_positive and not has_negative:
                        # 부정 표현만 제외 (현재 상태와 과거 경험 모두 포함)
                        # "키운 적이 있다"는 과거 경험이지만 현재도 키울 수 있으므로 포함
                        # 단, 명확한 부정 표현만 제외
                        negative_filter = """
                          AND NOT (
                            summary_text LIKE '%없다%' 
                            OR summary_text LIKE '%없음%'
                            OR summary_text LIKE '%키워본 적이 없다%'
                            OR summary_text LIKE '%키운 적이 없다%'
                            OR summary_text LIKE '%하지 않는다%'
                            OR summary_text LIKE '%하지 않음%'
                            OR summary_text LIKE '%받지 않는다%'
                            OR summary_text LIKE '%이용하지 않는다%'
                            OR summary_text LIKE '%사용하지 않는다%'
                            OR summary_text LIKE '%선호하지 않는다%'
                            OR summary_text LIKE '%안 한다%'
                            OR summary_text LIKE '%안한다%'
                          )
                        """
            
            # FTS 쿼리 생성 (형태소 분석 적용)
            # 검색어도 형태소 분석을 거쳐 인덱싱된 토큰과 매칭되도록 함
            print(f"      🔤 FTS 키워드 추출 시작: 쿼리='{query}'")
            
            # 형태소 분석을 통한 키워드 정규화
            normalized_query = normalize_query_morphology(query)
            
            if not normalized_query:
                # 형태소 분석 실패 시 기존 방식 사용 (하위 호환성)
                import re
                keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query)
                keywords = [k for k in keywords if len(k) >= 2]
                if not keywords:
                    keywords = [k.strip() for k in query.split() if k.strip() and len(k.strip()) >= 2]
                
                if not keywords:
                    print(f"      ⚠️ FTS 검색: 키워드 추출 실패 (쿼리: {query})")
                    return []
                
                print(f"      ⚠️ 형태소 분석 실패, 기본 방식 사용: {keywords}")
            else:
                # 형태소 분석 성공: 정규화된 키워드 사용
                keywords = normalized_query.split()
                print(f"      ✅ 형태소 분석 완료: {keywords}")
            
            # [수정] plainto_tsquery를 사용하므로 키워드를 공백으로 구분하여 전달
            # plainto_tsquery는 자동으로 AND 연산을 수행하므로 ' & ' 연결 불필요
            final_query = ' '.join(keywords)
            print(f"      📝 FTS 쿼리 (plainto_tsquery용): {final_query}")
            
            # 주의: plainto_tsquery는 부정어(!) 연산자를 직접 지원하지 않으므로
            # 부정어 필터링은 negative_filter (LIKE 절)로 처리합니다.
            
            # SQL 쿼리 실행 (부정어는 쿼리에 포함되어 있으므로 negative_filter 제거)
            # [수정] to_tsquery 대신 plainto_tsquery 사용 (공백으로 구분된 키워드를 AND 연산으로 처리)
            # plainto_tsquery는 사용자 입력을 안전하게 처리하고 타임아웃을 방지합니다.
            if segment_filter and len(segment_filter) > 0:
                # segment_name을 먼저 필터링하여 FTS 검색 범위 축소
                sql_query = text(f"""
                    SELECT 
                        panel_id,
                        ts_rank(ts_vector_korean, plainto_tsquery('korean', :query_str)) as rank
                    FROM panel_summary_segments
                    WHERE segment_name = ANY(:segment_filter_array)
                      AND ts_vector_korean IS NOT NULL
                      AND ts_vector_korean @@ plainto_tsquery('korean', :query_str)
                      {negative_filter}
                    ORDER BY rank DESC
                    LIMIT :limit_val
                """)
            else:
                # segment_filter가 없으면 기존 방식 사용
                sql_query = text(f"""
                    SELECT 
                        panel_id,
                        ts_rank(ts_vector_korean, plainto_tsquery('korean', :query_str)) as rank
                    FROM panel_summary_segments
                    WHERE ts_vector_korean IS NOT NULL
                      AND ts_vector_korean @@ plainto_tsquery('korean', :query_str)
                      {negative_filter}
                    ORDER BY rank DESC
                    LIMIT :limit_val
                """)
            
            # limit 최적화: 불필요하게 많이 가져오지 않음
            # FTS 검색도 limit * 2 정도면 충분
            effective_limit = min(limit * 2, 10000)  # 최대 10,000개로 제한
            
            params = {
                "query_str": final_query,
                "limit_val": effective_limit
            }
            if segment_filter and len(segment_filter) > 0:
                params["segment_filter_array"] = segment_filter
            # SQL 인젝션 방지: 부정 패턴 파라미터 추가
            params.update(negative_filter_params)
            
            try:
                result = await session.execute(sql_query, params)
                rows = result.fetchall()
            except Exception as e:
                # 에러 발생 시 트랜잭션 롤백
                await session.rollback()
                print(f"    ❌ FTS 검색 실행 오류: {e}")
                raise
            
            if not rows:
                print(f"    ⚠️ FTS 검색 결과가 없습니다. (쿼리: {query[:50]}, DB에 ts_vector_korean 데이터가 있는지 확인 필요)")
            
            # 패널별 점수 처리: 여러 세그먼트에서 매칭되면 점수 중첩 (합산)
            # 여러 세그먼트에서 매칭되는 것은 더 관련성이 높다는 신호
            panel_scores: Dict[str, List[float]] = {}  # 패널별 점수 리스트
            for panel_id, rank in rows:
                if panel_id:
                    panel_id_str = str(panel_id)
                    rank_float = float(rank) if rank else 0.0
                    if panel_id_str not in panel_scores:
                        panel_scores[panel_id_str] = []
                    panel_scores[panel_id_str].append(rank_float)
            
            # 패널별 최종 점수 계산: 평균 + 최고값 가중합 (여러 세그먼트 매칭 보너스)
            best_by_panel: Dict[str, float] = {}
            for panel_id_str, scores in panel_scores.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    max_score = max(scores)
                    # 평균 70% + 최고값 30% (여러 세그먼트 매칭 시 보너스)
                    final_score = avg_score * 0.7 + max_score * 0.3
                    # 세그먼트 개수 보너스 (최대 1.2배)
                    segment_bonus = min(1.0 + (len(scores) - 1) * 0.1, 1.2)
                    best_by_panel[panel_id_str] = final_score * segment_bonus
            sorted_items = sorted(best_by_panel.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [
                {"panel_id": pid, "score": score, "source": "fts"}
                for pid, score in sorted_items
            ]
        except Exception as e:
            print(f"    ❌ FTS 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # [중요] 세션 닫기
            if should_close:
                await session.close()

    async def semantic_search_with_filters(
        self, query: str, candidate_ids: List[str], limit: int
    ) -> List[Dict[str, Any]]:
        # [수정] 세션 관리 로직 추가
        session = self.session if self.session else AsyncSessionLocal()
        should_close = self.session is None
        
        try:
            embedding_model = get_embedding_model()
            if not embedding_model or not candidate_ids:
                return []
            
            query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            sql_query = text(f"""
                SELECT 
                    panel_id,
                    1 - (embedding <=> CAST(:embedding_str AS vector)) as similarity
                FROM panel_summary_segments
                WHERE embedding IS NOT NULL
                  AND panel_id = ANY(:candidate_ids)
                ORDER BY embedding <=> CAST(:embedding_str AS vector)
                LIMIT :limit_val
            """)
            result = await session.execute(sql_query, {
                "embedding_str": embedding_str,
                "candidate_ids": candidate_ids,
                "limit_val": limit * 5
            })
            rows = result.fetchall()
            best_by_panel: Dict[str, float] = {}
            for panel_id, similarity in rows:
                if panel_id:
                    panel_id_str = str(panel_id)
                    similarity_float = float(similarity) if similarity else 0.0
                    if panel_id_str not in best_by_panel or similarity_float > best_by_panel[panel_id_str]:
                        best_by_panel[panel_id_str] = similarity_float
            
            sorted_items = sorted(best_by_panel.items(), key=lambda x: x[1], reverse=True)[:limit]
            return [
                {"panel_id": pid, "score": score, "source": "vector_filtered"}
                for pid, score in sorted_items
            ]
        except Exception as e:
            print(f"  ❌ 필터링된 벡터 검색 오류: {e}")
            return []
        finally:
            # [중요] 세션 닫기
            if should_close:
                await session.close()
