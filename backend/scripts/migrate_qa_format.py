"""
질문-답변 형식으로 summary_text 업데이트 마이그레이션 스크립트

기존에 답변만 저장된 summary_text를 "질문 답변" 형식으로 업데이트합니다.

사용 예시:
    python backend/scripts/migrate_qa_format.py --batch-size 1000
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from sqlalchemy import text

# 프로젝트 루트 경로 추가
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.repositories.database import AsyncSessionLocal

load_dotenv(Path(PROJECT_ROOT) / ".env")


# 필드명 → 한국어 매핑 (column_metadata에 없는 필드용)
FIELD_NAME_KO_MAP = {
    "fitness_management_method": "운동 관리 방법",
    "chatbot_experience": "챗봇 사용 경험",
    "chatbot_main_purpose": "챗봇 주요 사용 목적",
    "main_chatbot_used": "주로 사용하는 챗봇",
    "preferred_chatbot": "선호하는 챗봇",
    "ai_usage_field": "AI 사용 분야",
    "main_apps_used": "주로 사용하는 앱",
    "ott_service_count": "이용하는 OTT 서비스 개수",
    "skincare_spending_monthly": "월 스킨케어 지출",
    "skincare_considerations": "스킨케어 제품 선택 시 고려사항",
    "skin_satisfaction": "피부 만족도",
    "most_effective_diet_experience": "가장 효과적이었던 다이어트 경험",
    "most_saved_photos_topic": "가장 많이 저장하는 사진 주제",
    "preferred_spending_category": "선호하는 소비 카테고리",
    "high_spending_category": "높은 지출 카테고리",
    "preferred_new_year_gift": "선호하는 설 선물",
    "preferred_water_play_area": "선호하는 물놀이 장소",
    "preferred_overseas_destination": "선호하는 해외 여행지",
    "preferred_summer_snack": "선호하는 여름 간식",
    "memorable_childhood_winter_activity": "기억에 남는 어린 시절 겨울 활동",
    "travel_style": "여행 스타일",
    "traditional_market_visit_frequency": "전통시장 방문 빈도",
    "main_quick_delivery_products": "주로 주문하는 퀵배송 상품",
    "reward_points_interest": "리워드 포인트 관심도",
    "lifestyle_values": "라이프스타일 가치관",
    "privacy_habits": "개인정보 보호 습관",
    "reducing_plastic_bags": "비닐봉지 사용 줄이기 방법",
    "rainy_day_coping_method": "비 오는 날 대처 방법",
    "late_night_snack_method": "야식 섭취 방법",
    "morning_wakeup_method": "아침 기상 방법",
    "solo_dining_frequency": "혼밥 빈도",
    "preferred_chocolate_situation": "초콜릿을 선호하는 상황",
    "moving_stress_factors": "이사 스트레스 요인",
    "conditions_for_happy_old_age": "행복한 노후를 위한 조건",
    "pets": "반려동물 경험",
    "stress_factors": "스트레스 요인",
    "stress_relief_method": "스트레스 해소 방법",
    "summer_fashion_essential": "여름 패션 필수품",
    "summer_sweat_discomfort": "여름 땀 불편함",
    "summer_worries": "여름 걱정",
    "waste_disposal_method": "쓰레기 처리 방법",
}


def load_column_metadata() -> Dict[str, Dict[str, Any]]:
    """column_metadata.json을 로드하여 필드명 → 메타데이터 매핑 반환"""
    metadata_path = Path(PROJECT_ROOT) / "backend" / "data" / "column_metadata.json"
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ column_metadata.json 로드 실패: {e}")
        return {}


def field_name_to_korean(field_name: str) -> str:
    """필드명을 한국어로 변환"""
    # 매핑 테이블 확인
    if field_name.lower() in FIELD_NAME_KO_MAP:
        return FIELD_NAME_KO_MAP[field_name.lower()]
    
    # snake_case를 한글로 변환 시도
    words = field_name.lower().split("_")
    # 간단한 영어 단어 → 한글 변환
    word_map = {
        "fitness": "운동", "management": "관리", "method": "방법",
        "chatbot": "챗봇", "experience": "경험", "main": "주요", "purpose": "목적",
        "used": "사용", "preferred": "선호", "ai": "AI", "usage": "사용", "field": "분야",
        "apps": "앱", "ott": "OTT", "service": "서비스", "count": "개수",
        "skincare": "스킨케어", "spending": "지출", "monthly": "월",
        "considerations": "고려사항", "skin": "피부", "satisfaction": "만족도",
        "most": "가장", "effective": "효과적인", "diet": "다이어트",
        "saved": "저장한", "photos": "사진", "topic": "주제",
        "spending": "지출", "category": "카테고리", "high": "높은",
        "new": "새해", "year": "년", "gift": "선물",
        "water": "물", "play": "놀이", "area": "장소",
        "overseas": "해외", "destination": "여행지",
        "summer": "여름", "snack": "간식",
        "memorable": "기억에 남는", "childhood": "어린 시절", "winter": "겨울", "activity": "활동",
        "travel": "여행", "style": "스타일",
        "traditional": "전통", "market": "시장", "visit": "방문", "frequency": "빈도",
        "quick": "퀵", "delivery": "배송", "products": "상품",
        "reward": "리워드", "points": "포인트", "interest": "관심도",
        "lifestyle": "라이프스타일", "values": "가치관",
        "privacy": "개인정보", "habits": "습관",
        "reducing": "줄이기", "plastic": "비닐", "bags": "봉지",
        "rainy": "비 오는", "day": "날", "coping": "대처",
        "late": "늦은", "night": "밤",
        "morning": "아침", "wakeup": "기상",
        "solo": "혼자", "dining": "식사",
        "chocolate": "초콜릿", "situation": "상황",
        "moving": "이사", "stress": "스트레스", "factors": "요인",
        "conditions": "조건", "for": "을 위한", "happy": "행복한", "old": "노후", "age": "나이",
        "pets": "반려동물",
    }
    
    translated = []
    for word in words:
        if word in word_map:
            translated.append(word_map[word])
        else:
            translated.append(word)
    
    return " ".join(translated)


def generate_question(field_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """필드명과 메타데이터를 기반으로 질문 생성"""
    if metadata and "name_ko" in metadata:
        field_name_ko = metadata["name_ko"]
    else:
        field_name_ko = field_name_to_korean(field_name)
    
    # 조사 처리
    last_char = field_name_ko[-1]
    if ord(last_char) >= 0xAC00 and ord(last_char) <= 0xD7A3:
        if (ord(last_char) - 0xAC00) % 28 == 0:
            return f"{field_name_ko}는 무엇인가요?"
        else:
            return f"{field_name_ko}은 무엇인가요?"
    else:
        return f"{field_name_ko}은 무엇인가요?"


async def migrate_qa_format(
    batch_size: int = 5000,  # 1000 → 5000 (5배 증가)
    start_from: int = 0,
    dry_run: bool = False
):
    """질문-답변 형식으로 summary_text 업데이트
    
    Args:
        batch_size: 한 번에 처리할 레코드 수
        start_from: 시작 오프셋 (재시작용)
        dry_run: 실제 업데이트 없이 테스트만
    """
    print("="*80)
    print("🔍 질문-답변 형식 마이그레이션 시작")
    print("="*80)
    print(f"배치 크기: {batch_size}개")
    print(f"시작 오프셋: {start_from}")
    print(f"드라이런 모드: {dry_run}")
    print()
    
    # column_metadata 로드
    print("📚 컬럼 메타데이터 로드 중...")
    column_metadata = load_column_metadata()
    print(f"✅ {len(column_metadata)}개 필드 메타데이터 로드 완료\n")
    
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
            
            batch_updates = []
            update_params = []
            
            # 모든 데이터를 먼저 준비
            for panel_id, segment_name, current_text in rows:
                try:
                    # segment_name을 소문자로 변환하여 필드명으로 사용
                    field_name = segment_name.lower()
                    
                    # 이미 질문이 포함되어 있는지 확인 (간단한 휴리스틱)
                    if "은 무엇인가요?" in current_text or "는 무엇인가요?" in current_text or "무엇인가요?" in current_text:
                        # 이미 질문이 포함되어 있으면 스킵
                        continue
                    
                    # 질문 생성
                    metadata = column_metadata.get(field_name)
                    question = generate_question(field_name, metadata)
                    
                    # 질문 + 답변 결합
                    new_text = f"{question} {current_text}"
                    
                    # bulk update를 위한 파라미터 수집
                    update_params.append({
                        "new_text": new_text,
                        "panel_id": panel_id,
                        "segment_name": segment_name
                    })
                    
                    batch_updates.append({
                        "panel_id": panel_id,
                        "segment_name": segment_name,
                        "old": current_text[:50],
                        "new": new_text[:80]
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
                        SET summary_text = :new_text,
                            ts_vector_korean = to_tsvector('korean', :new_text),
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
            print(f"  ✅ 처리: {len(rows)}개 (업데이트: {len(batch_updates)}개, 오류: {errors}개)", flush=True)
            print(f"  ⏱️ 배치 시간: {batch_time:.2f}초", flush=True)
            print(f"  📈 전체 진행: {processed:,}/{total_count:,} ({progress:.1f}%)", flush=True)
            print(f"  ⏳ 예상 남은 시간: {remaining/60:.1f}분", flush=True)
            
            # 샘플 출력 (첫 배치만)
            if offset == start_from and batch_updates:
                print(f"\n  📝 샘플 변환:")
                for sample in batch_updates[:3]:
                    print(f"    {sample['segment_name']}:")
                    print(f"      이전: {sample['old']}...")
                    print(f"      이후: {sample['new']}...")
                print()
            
            offset += batch_size
            
            # 배치 간 짧은 대기 (DB 부하 방지)
            if not dry_run:
                await asyncio.sleep(0.1)
        
        # 최종 통계
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("✅ 마이그레이션 완료!")
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
    parser = argparse.ArgumentParser(description="질문-답변 형식 마이그레이션")
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
        await migrate_qa_format(
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

