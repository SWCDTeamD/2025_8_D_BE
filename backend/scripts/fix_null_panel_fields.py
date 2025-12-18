#!/usr/bin/env python3
"""
DB 데이터와 원본 JSON을 비교하여 JSON 데이터를 기준으로 불일치하는 필드를 업데이트하는 스크립트

- JSON 데이터를 기준(진실)으로 사용
- DB의 값이 JSON과 다르거나 NULL인 경우 JSON 값으로 덮어씌움
- 비교 필드: gender, age, region_city, region_gu, marital_status, children_count, family_size 등

사용법:
    python backend/scripts/fix_null_panel_fields.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 출력 버퍼링 비활성화 (실시간 로그를 위해)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from backend.repositories.database import AsyncSessionLocal, engine
from sqlalchemy import text

# ETL 파이프라인의 변환 함수들 import
from backend.scripts.etl_pipeline import (
    parse_income,
    parse_car_ownership,
    parse_array_field,
    preprocess_panel_data
)


async def fix_null_panel_fields(batch_size: int = 1000, compare_all: bool = True):
    """DB 데이터를 원본 JSON과 비교하여 불일치하는 필드를 업데이트
    
    Args:
        batch_size: 배치 처리 크기
        compare_all: True면 모든 패널 비교, False면 NULL 필드만
    """
    
    json_file = project_root / "backend" / "data" / "panel_data.json"
    
    if not json_file.exists():
        print(f"❌ JSON 파일을 찾을 수 없습니다: {json_file}")
        return
    
    print("=" * 80)
    print("DB 데이터와 JSON 데이터 비교 및 업데이트 스크립트")
    print("=" * 80)
    print()
    print("📌 JSON 데이터를 기준(진실)으로 사용합니다.")
    print()
    
    # 1. DB의 모든 패널 데이터 가져오기 (비교할 모든 필드)
    async with engine.begin() as conn:
        if compare_all:
            result = await conn.execute(text("""
                SELECT panel_id, gender, age, region_city, region_gu, 
                       marital_status, children_count, family_size,
                       education_level, occupation, monthly_personal_income, monthly_household_income,
                       phone_brand, phone_model, car_ownership, car_manufacturer, car_model,
                       owned_electronics, smoking_experience, smoking_brand,
                       e_cig_heated_brand, e_cig_liquid_brand, drinking_experience
                FROM panels
                ORDER BY panel_id
            """))
        else:
            result = await conn.execute(text("""
                SELECT panel_id, gender, age, region_city, region_gu, 
                       marital_status, children_count, family_size,
                       education_level, occupation, monthly_personal_income, monthly_household_income,
                       phone_brand, phone_model, car_ownership, car_manufacturer, car_model,
                       owned_electronics, smoking_experience, smoking_brand,
                       e_cig_heated_brand, e_cig_liquid_brand, drinking_experience
                FROM panels
                WHERE gender IS NULL OR age IS NULL OR region_city IS NULL
                ORDER BY panel_id
            """))
        
        # 모든 필드를 딕셔너리로 변환
        db_panels = {}
        for row in result.fetchall():
            db_panels[row[0]] = {
                'gender': row[1],
                'age': row[2],
                'region_city': row[3],
                'region_gu': row[4],
                'marital_status': row[5],
                'children_count': row[6],
                'family_size': row[7],
                'education_level': row[8],
                'occupation': row[9],
                'monthly_personal_income': row[10],
                'monthly_household_income': row[11],
                'phone_brand': row[12],
                'phone_model': row[13],
                'car_ownership': row[14],
                'car_manufacturer': row[15],
                'car_model': row[16],
                'owned_electronics': row[17],
                'smoking_experience': row[18],
                'smoking_brand': row[19],
                'e_cig_heated_brand': row[20],
                'e_cig_liquid_brand': row[21],
                'drinking_experience': row[22],
            }
    
    total_count = len(db_panels)
    print(f"DB 패널 수: {total_count:,}개")
    print()
    
    if total_count == 0:
        print("✅ 처리할 패널이 없습니다.")
        return
    
    # 2. 원본 JSON 로드
    print("원본 JSON 파일 로딩 중...")
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.loads(f.read())
    
    print(f"JSON 총 패널 수: {len(json_data):,}개")
    print()
    
    # JSON을 딕셔너리로 변환 (ETL 파이프라인의 변환 로직 사용)
    print("JSON 데이터를 DB 형식으로 변환 중...")
    json_panels = {}
    for item in json_data:
        panel_id = item.get('panel_id')
        if panel_id:
            try:
                # ETL 파이프라인의 preprocess_panel_data 함수 사용
                panel_data = preprocess_panel_data(item)
                json_panels[panel_id] = panel_data
            except Exception as e:
                print(f"⚠️ 패널 {panel_id} 변환 실패: {e}")
                continue
    
    print(f"JSON 패널 딕셔너리 생성 완료: {len(json_panels):,}개")
    print()
    
    # 3. 비교 및 업데이트
    updated_count = 0
    not_found_count = 0
    no_change_count = 0
    mismatch_count = 0
    
    async with AsyncSessionLocal() as session:
        panel_ids = list(db_panels.keys())
        
        for i in range(0, total_count, batch_size):
            batch_ids = panel_ids[i:i + batch_size]
            batch_updates = []
            
            for panel_id in batch_ids:
                if panel_id not in json_panels:
                    not_found_count += 1
                    continue
                
                db_data = db_panels[panel_id]
                json_data = json_panels[panel_id]
                
                # 비교할 모든 필드 목록 (panel_summary_text, search_labels 제외)
                fields_to_compare = [
                    'gender', 'age', 'region_city', 'region_gu', 
                    'marital_status', 'children_count', 'family_size',
                    'education_level', 'occupation', 
                    'monthly_personal_income', 'monthly_household_income',
                    'phone_brand', 'phone_model', 
                    'car_ownership', 'car_manufacturer', 'car_model',
                    'owned_electronics', 'smoking_experience', 'smoking_brand',
                    'e_cig_heated_brand', 'e_cig_liquid_brand', 'drinking_experience'
                ]
                
                # 불일치하는 필드 찾기
                updates = {}
                has_mismatch = False
                
                for field in fields_to_compare:
                    db_value = db_data.get(field)
                    json_value = json_data.get(field)
                    
                    # 배열 필드 비교 (순서 무시)
                    if field in ['owned_electronics', 'smoking_experience', 'smoking_brand',
                                'e_cig_heated_brand', 'e_cig_liquid_brand', 'drinking_experience']:
                        # 배열을 정렬하여 비교
                        db_arr = sorted(db_value) if db_value else []
                        json_arr = sorted(json_value) if json_value else []
                        if db_arr != json_arr and json_value is not None:
                            updates[field] = json_value
                            has_mismatch = True
                    else:
                        # 일반 필드 비교
                        if db_value != json_value:
                            # JSON 값이 None이 아닌 경우에만 업데이트
                            if json_value is not None:
                                updates[field] = json_value
                                has_mismatch = True
                
                if has_mismatch:
                    updates['panel_id'] = panel_id
                    batch_updates.append(updates)
                    mismatch_count += 1
                else:
                    no_change_count += 1
            
            # 배치 업데이트 실행
            if batch_updates:
                for update_data in batch_updates:
                    panel_id = update_data.pop('panel_id')
                    
                    # 동적 UPDATE 쿼리 생성
                    set_clauses = []
                    params = {'panel_id': panel_id}
                    
                    for field, value in update_data.items():
                        set_clauses.append(f"{field} = :{field}")
                        params[field] = value
                    
                    if set_clauses:
                        query = f"""
                            UPDATE panels
                            SET {', '.join(set_clauses)}, updated_at = NOW()
                            WHERE panel_id = :panel_id
                        """
                        await session.execute(text(query), params)
                
                await session.commit()
                updated_count += len(batch_updates)
            
            # 진행 상황 출력
            progress = min(i + batch_size, total_count)
            print(f"진행: {progress:,}/{total_count:,} ({progress/total_count*100:.1f}%) - 업데이트: {updated_count:,}개, 불일치: {mismatch_count:,}개")
    
    print()
    print("=" * 80)
    print("업데이트 완료!")
    print("=" * 80)
    print(f"  - 업데이트된 패널: {updated_count:,}개")
    print(f"  - 변경 없음: {no_change_count:,}개")
    print(f"  - JSON에서 찾을 수 없음: {not_found_count:,}개")
    print(f"  - 총 처리: {total_count:,}개")
    print()
    
    # 최종 확인
    async with engine.begin() as conn:
        result = await conn.execute(text("""
            SELECT COUNT(*) as count
            FROM panels
            WHERE gender IS NULL OR age IS NULL OR region_city IS NULL
        """))
        remaining = result.fetchone()[0]
        print(f"남은 NULL 패널: {remaining:,}개")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DB 데이터를 JSON 데이터와 비교하여 업데이트')
    parser.add_argument(
        '--null-only',
        action='store_true',
        help='NULL 필드만 업데이트 (기본값: 모든 불일치 필드 업데이트)'
    )
    args = parser.parse_args()
    
    asyncio.run(fix_null_panel_fields(compare_all=not args.null_only))

