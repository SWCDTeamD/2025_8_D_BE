"""
형태소 분석 유틸리티 모듈
검색 쿼리와 인덱싱에 공통으로 사용되는 형태소 분석 함수
"""
import re
from typing import Optional, Any

# 형태소 분석기 (Kiwi 우선, Okt 하위 호환)
_HAS_KIWI = False
Kiwi = None  # type: ignore
try:
    from kiwipiepy import Kiwi  # type: ignore
    _HAS_KIWI = True
except ImportError:
    pass

_HAS_KONLPY = False
Okt = None  # type: ignore
try:
    from konlpy.tag import Okt  # type: ignore
    import subprocess
    result = subprocess.run(['java', '-version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        _HAS_KONLPY = True
except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
    pass

# 전역 형태소 분석기 인스턴스 (캐싱)
_kiwi_tagger: Optional[Any] = None
_okt_tagger: Optional[Any] = None

# 사용자 사전에 추가할 신조어/외래어 목록
USER_DICTIONARY = [
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


def get_morphology_tagger():
    """형태소 분석기 인스턴스 반환 (캐싱)"""
    global _kiwi_tagger, _okt_tagger
    
    # Kiwi 우선 사용
    if _HAS_KIWI and Kiwi is not None and _kiwi_tagger is None:
        try:
            _kiwi_tagger = Kiwi()
            # 사용자 사전 추가
            if _kiwi_tagger is not None:
                for word, pos in USER_DICTIONARY:
                    try:
                        _kiwi_tagger.add_user_word(word, pos)  # type: ignore
                    except Exception:
                        pass  # 이미 추가되었거나 실패해도 계속 진행
        except Exception:
            _kiwi_tagger = None
    
    # Okt 하위 호환
    if _kiwi_tagger is None and _HAS_KONLPY and Okt is not None and _okt_tagger is None:
        try:
            _okt_tagger = Okt()  # type: ignore
        except Exception:
            _okt_tagger = None
    
    return _kiwi_tagger, _okt_tagger


def normalize_query_morphology(query: str) -> str:
    """검색 쿼리를 형태소 분석하여 정규화된 키워드 추출
    
    Args:
        query: 검색 쿼리 (예: "맛있는 음식")
    
    Returns:
        정규화된 키워드 문자열 (예: "맛있 음식")
    """
    if not query or not query.strip():
        return ""
    
    kiwi_tagger, okt_tagger = get_morphology_tagger()
    
    # 1순위: Kiwi
    if kiwi_tagger:
        try:
            result = kiwi_tagger.analyze(query)
            keywords = []
            
            for token in result[0][0]:  # 첫 번째 문장의 토큰들
                word = token.form  # 형태소
                pos = token.tag   # 품사 태그
                
                # 명사, 동사, 형용사, 영어, 숫자만 포함
                if pos.startswith('NN') or pos.startswith('VV') or pos.startswith('VA') or \
                   pos == 'SL' or pos == 'SN':
                    keywords.append(word)
            
            if keywords:
                return ' '.join(set(keywords))
        except Exception:
            pass
    
    # 2순위: Okt
    if okt_tagger:
        try:
            pos_tags = okt_tagger.pos(query, stem=True)
            keywords = []
            for word, pos in pos_tags:
                if pos.startswith('N') or pos.startswith('V') or pos.startswith('A') or \
                   pos == 'SL' or pos == 'SN':
                    keywords.append(word)
            
            if keywords:
                return ' '.join(set(keywords))
        except Exception:
            pass
    
    # 형태소 분석기 없으면 간단한 정규화
    # 한글, 영문, 숫자만 추출
    keywords = re.findall(r'[가-힣a-zA-Z0-9]+', query)
    # 2글자 이상만 사용
    keywords = [k for k in keywords if len(k) >= 2]
    return ' '.join(set(keywords))

