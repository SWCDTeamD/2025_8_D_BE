"""
TSVECTOR 형태소 분석 마이그레이션 스크립트

기존 panel_summary_segments 테이블의 summary_text를 형태소 분석하여
ts_vector_korean 컬럼을 재생성합니다.

사용 예시:
    python backend/scripts/migrate_tsvector_morphology.py --batch-size 1000
"""
import argparse
import asyncio
import os
import sys
import time
import subprocess
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import text
from backend.repositories.database import AsyncSessionLocal

# 프로젝트 루트 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(Path(PROJECT_ROOT) / ".env")

# 형태소 분석기 (kiwipiepy - Java 불필요!)
_HAS_KIWI = False
Kiwi = None
try:
    from kiwipiepy import Kiwi
    _HAS_KIWI = True
except ImportError:
    print("⚠️ kiwipiepy가 설치되지 않았습니다. 설치: pip install kiwipiepy")
    
# 하위 호환성: konlpy도 시도
_HAS_KONLPY = False
Okt = None
try:
    from konlpy.tag import Okt
    import subprocess
    result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        _HAS_KONLPY = True
except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
    pass


def normalize_text_morphology(text: str, kiwi_tagger=None, okt_tagger=None) -> str:
    """형태소 분석을 통한 텍스트 정규화
    
    Args:
        text: 원본 텍스트
        kiwi_tagger: Kiwi 형태소 분석기 (우선 사용)
        okt_tagger: Okt 형태소 분석기 (하위 호환성)
    
    Returns:
        정규화된 텍스트 (형태소로 분리된 키워드만 추출)
    """
    if not text or not text.strip():
        return ""
    
    # 1순위: Kiwi (Java 불필요)
    if kiwi_tagger:
        try:
            # Kiwi 형태소 분석
            result = kiwi_tagger.analyze(text)
            keywords = []
            
            for token in result[0][0]:  # 첫 번째 문장의 토큰들
                word = token.form  # 형태소
                pos = token.tag   # 품사 태그
                
                # 명사, 동사, 형용사, 영어, 숫자만 포함
                # Kiwi 품사 태그: NNG(일반명사), NNP(고유명사), VV(동사), VA(형용사), SL(외국어), SN(숫자)
                if pos.startswith('NN') or pos.startswith('VV') or pos.startswith('VA') or \
                   pos == 'SL' or pos == 'SN':
                    keywords.append(word)
            
            if keywords:
                normalized = ' '.join(set(keywords))
                return normalized.strip()
        except Exception as e:
            print(f"  ⚠️ Kiwi 형태소 분석 실패: {e}, 다음 방법 시도")
    
    # 2순위: Okt (Java 필요)
    if okt_tagger:
        try:
            # 형태소 분석 (명사, 동사, 형용사만 추출)
            morphs = okt_tagger.morphs(text, stem=True)  # 어간 추출
            
            # 불용어 제거 (조사, 어미 등)
            pos_tags = okt_tagger.pos(text, stem=True)
            keywords = []
            for word, pos in pos_tags:
                # 명사, 동사, 형용사, 영어, 숫자만 포함
                if pos.startswith('N') or pos.startswith('V') or pos.startswith('A') or \
                   pos == 'SL' or pos == 'SN':  # SL: 외국어, SN: 숫자
                    keywords.append(word)
            
            # 키워드가 없으면 형태소만 사용
            if not keywords:
                keywords = morphs
            
            # 중복 제거 및 정리
            normalized = ' '.join(set(keywords))
            return normalized.strip()
        except Exception as e:
            print(f"  ⚠️ Okt 형태소 분석 실패: {e}, 간단한 정규화 사용")
    
    # 형태소 분석기 없으면 간단한 정규화
    # 1. 특수문자 제거
    normalized = re.sub(r'[^\w\s가-힣]', ' ', text)
    
    # 2. 조사/어미 패턴 제거 (간단한 휴리스틱)
    # "을/를", "이/가", "은/는", "의", "에", "에서", "와/과", "로/으로" 등
    common_particles = ['을', '를', '이', '가', '은', '는', '의', '에', '에서', '와', '과', '로', '으로', 
                       '도', '만', '까지', '부터', '에게', '한테', '께', '더러', '에게서', '한테서']
    
    words = normalized.split()
    filtered_words = []
    for word in words:
        # 조사 제거 (단어 끝에 붙은 조사)
        for particle in common_particles:
            if word.endswith(particle) and len(word) > len(particle):
                word = word[:-len(particle)]
                break
        
        # 2글자 이상만 포함 (1글자는 대부분 조사/어미)
        if len(word) >= 2:
            filtered_words.append(word)
    
    # 3. 중복 제거 및 정리
    normalized = ' '.join(set(filtered_words))
    return normalized.strip()


async def create_backup_column(session, dry_run: bool = False):
    """기존 ts_vector_korean을 백업하는 컬럼 생성"""
    backup_column_name = "ts_vector_korean_backup"
    
    # 백업 컬럼 존재 여부 확인
    check_query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'panel_summary_segments' 
          AND column_name = :backup_name
    """)
    result = await session.execute(check_query, {"backup_name": backup_column_name})
    exists = result.scalar() is not None
    
    if exists:
        print(f"✅ 백업 컬럼 '{backup_column_name}' 이미 존재")
        return True
    
    if dry_run:
        print(f"⚠️ 드라이런: 백업 컬럼 '{backup_column_name}' 생성 예정")
        return True
    
    try:
        # 1. 백업 컬럼 생성
        create_backup_query = text(f"""
            ALTER TABLE panel_summary_segments 
            ADD COLUMN {backup_column_name} tsvector
        """)
        await session.execute(create_backup_query)
        await session.commit()
        
        # 2. 데이터 복사
        copy_backup_query = text(f"""
            UPDATE panel_summary_segments 
            SET {backup_column_name} = ts_vector_korean 
            WHERE ts_vector_korean IS NOT NULL
        """)
        await session.execute(copy_backup_query)
        await session.commit()
        
        print(f"✅ 백업 컬럼 '{backup_column_name}' 생성 및 데이터 복사 완료")
        return True
    except Exception as e:
        print(f"❌ 백업 컬럼 생성 실패: {e}")
        await session.rollback()
        return False


async def migrate_tsvector_morphology(
    batch_size: int = 3000,  # 1000 → 3000 (3배 증가, 형태소 분석 부하 고려)
    start_from: int = 0,
    dry_run: bool = False,
    create_backup: bool = True
):
    """TSVECTOR 형태소 분석 마이그레이션
    
    Args:
        batch_size: 한 번에 처리할 레코드 수
        start_from: 시작 오프셋 (재시작용)
        dry_run: 실제 업데이트 없이 테스트만
        create_backup: 백업 컬럼 생성 여부 (기본값: True)
    """
    if not _HAS_KIWI and not _HAS_KONLPY:
        print("⚠️ 형태소 분석기가 없습니다.")
        print("   권장: pip install kiwipiepy (Java 불필요)")
        print("   또는: pip install konlpy + Java 설치")
        print("   간단한 정규화 방법으로 진행합니다...")
    
    print("="*80)
    print("🔍 TSVECTOR 형태소 분석 마이그레이션 시작")
    print("="*80)
    print(f"배치 크기: {batch_size}개")
    print(f"시작 오프셋: {start_from}")
    print(f"드라이런 모드: {dry_run}")
    print(f"백업 생성: {create_backup}")
    print()
    
    if not dry_run:
        print("⚠️ 주의사항:")
        print("   - 기존 'ts_vector_korean' 컬럼의 데이터가 업데이트됩니다")
        if create_backup:
            print("   - 백업 컬럼 'ts_vector_korean_backup'이 자동 생성됩니다")
            print("   - 문제 발생 시 백업에서 복구 가능합니다")
        else:
            print("   - ⚠️ 백업 없이 진행됩니다 (위험!)")
        print()
    
    # 형태소 분석기 초기화 (우선순위: Kiwi > Okt > 간단한 정규화)
    kiwi_tagger = None
    okt_tagger = None
    
    # 사용자 사전에 추가할 신조어/외래어 목록
    user_dictionary = [
        # 신조어/브랜드명
        ("맥시멀리스트", "NNG"),  # 일반명사
        ("ChatGPT", "SL"),  # 외국어
        ("OTT", "SL"),  # 외국어
        ("AI", "SL"),  # 외국어
        ("스킨케어", "NNG"),  # 복합어
        ("라이프스타일", "NNG"),  # 복합어
        ("퀵배송", "NNG"),  # 복합어
        ("전기요금", "NNG"),  # 복합어
        ("선글라스", "NNG"),  # 복합어
        ("반바지", "NNG"),  # 복합어
        ("혼밥", "NNG"),  # 신조어
        ("혼자", "NNG"),  # 명사화
        ("노후", "NNG"),  # 복합어
        ("경제력", "NNG"),  # 복합어
    ]
    
    if _HAS_KIWI:
        print("📚 Kiwi 형태소 분석기 초기화 중... (Java 불필요)")
        try:
            kiwi_tagger = Kiwi()
            
            # 사용자 사전 추가
            print("📖 사용자 사전 추가 중...")
            for word, pos in user_dictionary:
                try:
                    kiwi_tagger.add_user_word(word, pos)
                    print(f"   ✓ {word} ({pos})")
                except Exception as e:
                    print(f"   ⚠️ {word} 추가 실패: {e}")
            
            print("✅ Kiwi 형태소 분석기 준비 완료 (사용자 사전 포함)\n")
        except Exception as e:
            print(f"⚠️ Kiwi 초기화 실패: {e}")
            kiwi_tagger = None
    
    if not kiwi_tagger and _HAS_KONLPY:
        print("📚 Okt 형태소 분석기 초기화 중... (Java 필요)")
        try:
            okt_tagger = Okt()
            print("✅ Okt 형태소 분석기 준비 완료\n")
        except Exception as e:
            print(f"⚠️ Okt 초기화 실패: {e}")
            okt_tagger = None
    
    if not kiwi_tagger and not okt_tagger:
        print("⚠️ 형태소 분석기 없음 - 간단한 정규화 방법 사용\n")
    
    async with AsyncSessionLocal() as session:
        # 백업 컬럼 생성 (안전성)
        if create_backup and not dry_run:
            print("💾 기존 데이터 백업 중...")
            backup_success = await create_backup_column(session, dry_run=False)
            if not backup_success:
                print("⚠️ 백업 실패했지만 계속 진행합니다...")
            print()
        
        # 전체 개수 확인
        count_query = text("""
            SELECT COUNT(*) 
            FROM panel_summary_segments 
            WHERE summary_text IS NOT NULL
        """)
        result = await session.execute(count_query)
        total_count = result.scalar()
        print(f"📊 총 처리 대상: {total_count:,}개\n")
        
        if start_from >= total_count:
            print(f"⚠️ 시작 오프셋({start_from})이 총 개수({total_count})보다 큽니다.")
            return
        
        processed = 0
        updated = 0
        errors = 0
        start_time = time.time()
        
        # 배치 처리
        offset = start_from
        while offset < total_count:
            batch_start_time = time.time()
            
            # 배치 데이터 조회
            select_query = text("""
                SELECT 
                    panel_id,
                    segment_name,
                    summary_text,
                    ts_vector_korean
                FROM panel_summary_segments
                WHERE summary_text IS NOT NULL
                ORDER BY panel_id, segment_name
                LIMIT :limit OFFSET :offset
            """)
            result = await session.execute(select_query, {
                "limit": batch_size,
                "offset": offset
            })
            rows = result.fetchall()
            
            if not rows:
                break
            
            batch_num = offset // batch_size + 1
            print(f"\n--- 배치 {batch_num} (오프셋: {offset:,} ~ {offset + len(rows):,}) ---")
            sys.stdout.flush()  # 출력 즉시 반영
            
            # 배치 내 각 레코드 처리
            batch_updates = []
            update_params = []
            processed_in_batch = 0
            for i, (panel_id, segment_name, summary_text, old_tsvector) in enumerate(rows):
                try:
                    # 형태소 분석 (Kiwi 우선, Okt 하위 호환)
                    normalized_text = normalize_text_morphology(
                        summary_text, 
                        kiwi_tagger=kiwi_tagger,
                        okt_tagger=okt_tagger
                    )
                    
                    if not normalized_text:
                        continue
                    
                    # bulk update를 위한 파라미터 수집
                    update_params.append({
                        "normalized_text": normalized_text,
                        "panel_id": panel_id,
                        "segment_name": segment_name
                    })
                    
                    # 형태소 분석 상세 정보 (Kiwi 사용 시)
                    morphs_info = ""
                    if kiwi_tagger and offset == start_from and i < 5:
                        try:
                            result_kiwi = kiwi_tagger.analyze(summary_text)
                            morphs_list = [f"{t.form}({t.tag})" for t in result_kiwi[0][0][:8]]
                            morphs_info = " | ".join(morphs_list)
                        except:
                            pass
                    
                    batch_updates.append({
                        "panel_id": panel_id,
                        "segment_name": segment_name,
                        "original": summary_text,
                        "normalized": normalized_text,
                        "morphs_info": morphs_info
                    })
                    updated += 1
                    processed_in_batch += 1
                    
                    # 100개마다 진행 상황 출력 (큰 배치의 경우)
                    if batch_size >= 1000 and processed_in_batch % 100 == 0:
                        print(f"    진행: {processed_in_batch}/{len(rows)}개 처리 중...", flush=True)
                    
                except Exception as e:
                    errors += 1
                    print(f"  ❌ 오류 (Panel {panel_id}, Segment {segment_name}): {e}", flush=True)
            
            # Bulk update 실행 (executemany 사용)
            if not dry_run and update_params:
                try:
                    update_query = text("""
                        UPDATE panel_summary_segments
                        SET ts_vector_korean = to_tsvector('korean', :normalized_text)
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
            
            print(f"  ✅ 처리: {len(rows)}개 (업데이트: {len(batch_updates)}개, 오류: {errors}개)", flush=True)
            print(f"  ⏱️ 배치 시간: {batch_time:.2f}초", flush=True)
            print(f"  📈 전체 진행: {processed:,}/{total_count:,} ({progress:.1f}%)", flush=True)
            print(f"  ⏳ 예상 남은 시간: {remaining/60:.1f}분", flush=True)
            
            # 샘플 출력 (첫 배치만, 더 자세하게)
            if offset == start_from and batch_updates:
                print(f"\n  📝 샘플 변환 (상세):")
                for sample in batch_updates[:5]:
                    print(f"\n    Segment: {sample['segment_name']}")
                    print(f"    원본: {sample['original']}")
                    if sample.get('morphs_info'):
                        print(f"    형태소 분석: {sample['morphs_info']}...")
                    print(f"    정규화: {sample['normalized']}")
                    
                    # 변경 여부 확인
                    orig = sample['original'].strip()
                    norm = sample['normalized'].strip()
                    if orig != norm:
                        # 조사/어미가 제거되었는지 확인
                        has_particles = any(p in orig for p in ['을', '를', '이', '가', '은', '는', '합니다', '니다', '하는'])
                        if has_particles and not any(p in norm for p in ['을', '를', '이', '가', '은', '는']):
                            print(f"    ✅ 조사/어미 제거됨 (형태소 분석 성공)")
                        else:
                            print(f"    ✅ 형태소 분석 적용됨 (키워드 추출)")
                    else:
                        print(f"    ℹ️ 변경 없음 (이미 정규화됨)")
            
            offset += batch_size
            
            # 배치 간 짧은 대기 (DB 부하 방지)
            if not dry_run:
                await asyncio.sleep(0.1)
        
        # 최종 통계
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("🎉 마이그레이션 완료!")
        print("="*80)
        print(f"✅ 총 처리: {processed:,}개")
        print(f"✅ 업데이트: {updated:,}개")
        print(f"⚠️ 오류: {errors}개")
        print(f"⏱️ 총 소요 시간: {total_time/60:.1f}분 ({total_time:.1f}초)")
        print(f"📊 평균 처리 속도: {processed/total_time:.1f}개/초")
        
        if dry_run:
            print("\n⚠️ 드라이런 모드였습니다. 실제 업데이트는 수행되지 않았습니다.")
        else:
            print("\n" + "="*80)
            print("✅✅✅ TSVECTOR 형태소 분석 마이그레이션 완료! ✅✅✅")
            print("="*80)
            print(f"📝 {updated:,}개의 세그먼트에 형태소 분석이 적용되었습니다.")
            print("🔍 이제 FTS 검색이 더 정확하게 작동합니다!")
            print("="*80)
        
        if create_backup and not dry_run:
            print("\n💾 백업 정보:")
            print(f"   📦 백업 컬럼: ts_vector_korean_backup")
            print(f"   🔄 복구 방법: UPDATE panel_summary_segments SET ts_vector_korean = ts_vector_korean_backup;")
            print(f"   🗑️ 백업 삭제: ALTER TABLE panel_summary_segments DROP COLUMN ts_vector_korean_backup;")
            print()


async def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="TSVECTOR 형태소 분석 마이그레이션")
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
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="백업 컬럼 생성 안 함 (기본값: 백업 생성)"
    )
    
    args = parser.parse_args()
    
    try:
        await migrate_tsvector_morphology(
            batch_size=args.batch_size,
            start_from=args.start_from,
            dry_run=args.dry_run,
            create_backup=not args.no_backup  # 기본값: True (백업 생성)
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

