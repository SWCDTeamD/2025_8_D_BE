"""
임베딩 재생성 스크립트

질문-답변 형식으로 변경된 summary_text에 대해 임베딩을 재생성합니다.

사용 예시:
    python backend/scripts/regenerate_embeddings.py --batch-size 1000
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from sqlalchemy import text

# 프로젝트 루트 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.repositories.database import AsyncSessionLocal

load_dotenv(Path(PROJECT_ROOT) / ".env")

# KoSimCSE 임베딩 모델
try:
    from sentence_transformers import SentenceTransformer
    _HAS_KOSIMCSE = True
except ImportError:
    SentenceTransformer = None
    _HAS_KOSIMCSE = False


async def regenerate_embeddings(
    batch_size: int = 2000,  # 1000 → 2000 (2배 증가, 임베딩 모델 부하 고려)
    start_from: int = 0,
    dry_run: bool = False
):
    """임베딩 재생성
    
    Args:
        batch_size: 한 번에 처리할 레코드 수
        start_from: 시작 오프셋 (재시작용)
        dry_run: 실제 업데이트 없이 테스트만
    """
    if not _HAS_KOSIMCSE:
        print("❌ sentence_transformers가 설치되지 않았습니다.")
        print("   설치: pip install sentence-transformers")
        return
    
    print("="*80)
    print("🔍 임베딩 재생성 시작")
    print("="*80)
    print(f"배치 크기: {batch_size}개")
    print(f"시작 오프셋: {start_from}")
    print(f"드라이런 모드: {dry_run}")
    print()
    
    # 임베딩 모델 로드
    print("📚 임베딩 모델 로드 중...")
    try:
        embedding_model = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
        print("✅ KoSimCSE 모델 로드 완료\n")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    async with AsyncSessionLocal() as session:
        # 전체 개수 확인
        count_query = text("""
            SELECT COUNT(*) 
            FROM panel_summary_segments 
            WHERE summary_text IS NOT NULL
        """)
        result = await session.execute(count_query)
        total_count = result.scalar()
        print(f"📊 총 처리 대상: {total_count:,}개\n")
        
        start_time = time.time()
        processed = 0
        updated = 0
        errors = 0
        offset = start_from
        
        while offset < total_count:
            batch_start_time = time.time()
            
            # 배치 조회
            query = text("""
                SELECT panel_id, segment_name, summary_text
                FROM panel_summary_segments
                WHERE summary_text IS NOT NULL
                ORDER BY panel_id, segment_name
                LIMIT :limit_val OFFSET :offset_val
            """)
            result = await session.execute(query, {
                "limit_val": batch_size,
                "offset_val": offset
            })
            rows = result.fetchall()
            
            if not rows:
                break
            
            # 배치 단위로 임베딩 생성
            batch_texts = [row[2] for row in rows]
            try:
                embeddings = embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=32
                )
            except Exception as e:
                print(f"  ❌ 임베딩 생성 실패: {e}")
                errors += len(rows)
                offset += batch_size
                continue
            
            # 배치 업데이트 - bulk update로 최적화
            update_params = []
            for i, (panel_id, segment_name, summary_text) in enumerate(rows):
                try:
                    embedding = embeddings[i]
                    embedding_str = f"[{','.join(map(str, embedding))}]"
                    
                    update_params.append({
                        "embedding": embedding_str,
                        "panel_id": panel_id,
                        "segment_name": segment_name
                    })
                    
                    updated += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"  ❌ 오류 (Panel {panel_id}, Segment {segment_name}): {e}", flush=True)
            
            # Bulk update 실행 (executemany 사용)
            if not dry_run and update_params:
                try:
                    update_query = text("""
                        UPDATE panel_summary_segments
                        SET embedding = CAST(:embedding AS vector),
                            updated_at = NOW()
                        WHERE panel_id = :panel_id 
                          AND segment_name = :segment_name
                    """)
                    # executemany로 한 번에 실행
                    await session.execute(update_query, update_params)
                    await session.commit()
                except Exception as e:
                    print(f"  ❌ 배치 업데이트 실패: {e}", flush=True)
                    await session.rollback()
                    errors += len(update_params)
            
            processed += len(rows)
            batch_time = time.time() - batch_start_time
            
            # 진행 상황 출력
            progress = (processed / total_count) * 100
            elapsed = time.time() - start_time
            avg_time_per_record = elapsed / processed if processed > 0 else 0
            remaining = (total_count - processed) * avg_time_per_record
            
            print(f"--- 배치 {offset // batch_size + 1} (오프셋: {offset:,} ~ {offset + len(rows):,}) ---", flush=True)
            print(f"  ✅ 처리: {len(rows)}개 (업데이트: {updated}개, 오류: {errors}개)", flush=True)
            print(f"  ⏱️ 배치 시간: {batch_time:.2f}초", flush=True)
            print(f"  📈 전체 진행: {processed:,}/{total_count:,} ({progress:.1f}%)", flush=True)
            print(f"  ⏳ 예상 남은 시간: {remaining/60:.1f}분", flush=True)
            
            offset += batch_size
            
            # 배치 간 짧은 대기 (GPU/CPU 부하 방지)
            if not dry_run:
                await asyncio.sleep(0.1)
        
        # 최종 통계
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("✅ 임베딩 재생성 완료!")
        print("="*80)
        print(f"총 처리: {processed:,}개")
        print(f"업데이트: {updated:,}개")
        print(f"오류: {errors}개")
        print(f"총 소요 시간: {total_time/60:.1f}분")
        print(f"평균 처리 속도: {processed/total_time:.1f}개/초")
        
        if dry_run:
            print("\n⚠️ 드라이런 모드였습니다. 실제 업데이트는 수행되지 않았습니다.")


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="임베딩 재생성")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="한 번에 처리할 레코드 수 (기본값: 1000)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="시작 오프셋 (재시작용, 기본값: 0)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 업데이트 없이 테스트만 수행"
    )
    
    args = parser.parse_args()
    
    try:
        await regenerate_embeddings(
            batch_size=args.batch_size,
            start_from=args.start_from,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
        print(f"다음 실행 시 --start-from {args.start_from} 옵션으로 재시작하세요.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

