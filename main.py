import pandas as pd
import numpy as np
import json
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import time
import datetime

# 환경 변수 .env 파일 로드 (api key를 숨기기 위한 용도)
load_dotenv()


# OpenAI 토큰 제한 설정

# gpt-4o-mini 모델 사용
# gpt-4o-mini의 토큰 제한은 분당 토큰 20만개, request 한 번에 128000개
# 데이터 100개당 토큰 약 20만개
MODEL = "gpt-4o-mini"
MAX_TOKENS_PER_REQUEST = 128000
TOKENS_PER_MINUTE = 200000
TOKEN_BUFFER = 1000  # 여유 토큰


def analyze_by_rule(csv_path):
    # 룰 기반 데이터 분석
    
    # 원본 데이터 로드
    df = pd.read_csv(csv_path)

    print("룰 기반 필터링")
    print(f"총 데이터 개수: {len(df)} 개")

    tagged_data = {}

    # 가격이 0이거나 None인 경우 제거
    df = df[(df['가격'] > 0) & (df['가격'].notnull())]

    # 가격 기준 하위 15% 필터링
    price_threshold = np.percentile(df['가격'], 15)
    lowest_price_cars = df[df['가격'] <= price_threshold]
    tagged_data['최저가'] = lowest_price_cars.copy()


    # 신차대비가격 0이거나 None 경우 제거
    df = df[(df['신차대비가격'] > 0) & (df['신차대비가격'].notnull())]

    # 신차대비가격 기준 하위 15% 필터링
    old_per_new_price_threshold = np.percentile(df['신차대비가격'], 15)
    lowest_old_per_new_price_cars = df[df['신차대비가격'] <= old_per_new_price_threshold]
    tagged_data['신차대비최저가'] = lowest_old_per_new_price_cars.copy()


    # 연식 데이터를 숫자로 변환 (yyyy.MM)
    date_df = df.copy()
    date_df['연식'] = date_df['연식'].astype(str).apply(lambda x: int(x[:4]) + int(x[5:7]) / 12)

    # 연식 기준 상위 15% 필터링
    year_threshold = np.percentile(date_df['연식'], 85)
    recent_year_cars = df[date_df['연식'] >= year_threshold]

    # 현재 연도 가져오기
    current_year = datetime.datetime.now().year

    # 연식이 최근 3년 이내인 데이터로만 한정되게 필터링
    recent_year_cars = recent_year_cars[recent_year_cars['연식'] >= (current_year - 3)]
    tagged_data['최신연식'] = recent_year_cars.copy()

    save_rule_tagged_data_to_csv(tagged_data, csv_path)


def save_rule_tagged_data_to_csv(tagged_data: dict[str, any], csv_path):
    print("\n규칙에 따라 분류된 태그별 데이터 저장")

    # 실행 중인 프로젝트 폴더 내부에 result 디렉토리 설정
    output_dir = make_result_dirs(
        csv_path=csv_path,
        classification_method='rule_based',
        output_dir='result')

    # 태그별로 CSV 파일 저장
    for tag in tagged_data:
        data_list = tagged_data[tag]
        if not data_list.empty:
            combined_data = data_list
            # 태그 이름에서 CSV 파일명에 적합하지 않은 문자 제거
            safe_tag_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in tag)
            output_path = os.path.join(output_dir, f"{safe_tag_name}.csv")

            combined_data.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"'{tag}' 태그 파일 저장 완료: {output_path}")


def make_result_dirs(csv_path, classification_method='rule_based', output_dir="result"):
    # 실행 중인 프로젝트 폴더 내부의 result 디렉토리 설정
    base_dir = os.getcwd()  # 현재 프로젝트 디렉토리
    output_dir = os.path.join(base_dir, output_dir)

    # 원본 CSV 파일 이름을 사용하여 저장 폴더 생성
    file_name = os.path.basename(csv_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    output_dir = os.path.join(output_dir, file_name_without_ext)

    # ai based 디렉토리로 나눠서 rule based 디렉토리와 구분
    output_dir = os.path.join(output_dir, classification_method)

    # 출력 디렉토리들 생성
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def estimate_tokens(text):
    # 텍스트의 토큰 수를 대략적으로 추정
    return len(text) // 4 # 평균적으로 토큰 하나는 약 4자


def analyze_dataset_with_dynamic_batching(full_df, text_column, tag_definitions):
    # API 키 설정
    api_key = os.getenv("OPENAI_API_KEY")

    print("LLM 기반 필터링")
    print(f"총 데이터 개수: {len(full_df)} 개")

    # OpenAI 클라이언트 초기화 (한 번만 생성)
    client = OpenAI(api_key=api_key)

    # ID 컬럼 확인/생성
    if 'id' not in full_df.columns:
        full_df['id'] = full_df.index.astype(str)

    # 결과 저장용 리스트
    all_tagged_data = []
    remaining_tokens = TOKENS_PER_MINUTE  # 매 분 사용할 수 있는 토큰 수
    current_batch = 0   # 현재 배치 카운팅용 변수
    batch_size = 10  # 초기 배치 크기
    i = 0

    while i < len(full_df):
        start_idx = i
        batch_size = min(batch_size, len(full_df) - i)  # 아직 남은 데이터에 맞게 batch_size 조정
        end_idx = i + batch_size
        batch_df = full_df.iloc[start_idx:end_idx]
        batch_json = batch_df.to_json(orient='records', indent=2)
        prompt = create_prompt(batch_json, text_column, tag_definitions)
        token_count = estimate_tokens(prompt) + TOKEN_BUFFER

        print(f"\n[배치 {current_batch}] 처리 중... ({start_idx}~{end_idx})")

        if token_count > MAX_TOKENS_PER_REQUEST:
            batch_size = max(1, batch_size // 2)
            print(f"토큰 초과 예상! 배치 크기를 {batch_size}로 조정")
            continue  # 배치 크기 조정 후 다시 시도

        if token_count > remaining_tokens:
            sleep_time = 60  # 1분 대기하여 토큰 제한 회복
            print(f"분당 토큰 제한 초과 예상! {sleep_time}초 대기")
            time.sleep(sleep_time)
            remaining_tokens = TOKENS_PER_MINUTE


        # API 호출
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 데이터 분석 전문가입니다. 텍스트를 분석하고 지정된 태그에 따라 특징적인 데이터만 분류합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content

            # JSON 파싱
            if "```json" in content:
                json_content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_content = content.split("```")[1].split("```")[0].strip()
            else:
                json_content = content

            try:
                batch_tagged_data = json.loads(json_content)
                if batch_tagged_data:
                    all_tagged_data.extend(batch_tagged_data)
                print(f"결과: {batch_tagged_data}")
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류. 원본 응답:")
                print(content)

            remaining_tokens -= token_count
            print(f"처리 완료, 남은 토큰: {remaining_tokens}")
            i += batch_size  # 다음 배치로 이동
            current_batch += 1  # 현재 배치 업데이트

            # batch_size 동적 조절: 토큰이 충분하면 증가, 부족하면 감소
            if token_count < MAX_TOKENS_PER_REQUEST // 2:
                batch_size = min(batch_size * 2, len(full_df) - i)

        except Exception as e:
            print(f"에러 발생: {e}")
            if "maximum context length" in str(e) or "too many tokens" in str(e):
                batch_size = max(1, batch_size // 2)  # 배치 크기 줄이기
                print(f"배치 크기를 {batch_size}로 줄여서 재시도")
            else:
                # 다른 에러시 1분 뒤 재시도
                print("1분 뒤 재시도")
                time.sleep(60)  # 1분 뒤 재시도

    return all_tagged_data



def create_prompt(dataset_json, text_column, tag_definitions):
    prompt = f"""
    # 데이터셋 분석 및 태그 분류 작업

    ## 작업 설명
    아래 제공된 데이터셋을 분석하고, 주요 텍스트 컬럼 '{text_column}'의 내용을 기반으로 특징적인 데이터만 분류해주세요.

    ## 태그 정의
    다음 태그 정의에 따라 데이터를 분류해주세요:
    {tag_definitions}

    ## 분류 규칙
    1. 특별한 특징이 없는 데이터는 태그를 부여하지 않습니다.
    2. 각 데이터는 여러 태그를 가질 수 있습니다.
    3. 주 분석 대상은 '{text_column}' 컬럼이지만, 다른 컬럼의 정보도 참고하여 분석하세요.
    4. 데이터 중 컬럼 일부의 값들이 null인 경우는 그 값들 수집에 실패해서 존재하지 않는 것입니다. 태그 분석 시 이러한 컬럼의 정보는 참고하면 안 됩니다.
    5. 데이터셋 전체 맥락에서 특징적인 항목만 태그를 부여하세요.
    6. 태그 선정 기준이 너무 느슨하지 말고 빡빡해야 합니다.

    ## 데이터셋
    {dataset_json}

    ## 출력 형식
    다음 JSON 형식으로 응답해주세요:
    ```
    [
      {{"id": "데이터ID", "tags": ["태그1", "태그2"], "reason": "이 태그를 부여한 이유"}},
      {{"id": "데이터ID", "tags": ["태그3"], "reason": "이 태그를 부여한 이유"}}
    ]
    ```
    특징이 없는 데이터는 포함하지 마세요.
    """
    return prompt


def analyze_and_save(csv_path, text_column, tag_definitions):
    # 원본 데이터 로드
    df = pd.read_csv(csv_path)

    # API 분석 수행
    tagged_data = analyze_dataset_with_dynamic_batching(df, text_column, tag_definitions)

    if tagged_data:
        # 태그별 CSV 파일 저장
        tag_groups = save_ai_tagged_data_to_csv(tagged_data, df, csv_path)
        return tag_groups
    else:
        print("\n분석 결과가 없습니다.")
        return None



def save_ai_tagged_data_to_csv(tagged_data: list, original_df, csv_path):
    print("\nai로 분류된 태그별 데이터 저장")

    # 실행 중인 프로젝트 폴더 내부에 result 디렉토리 설정
    output_dir = make_result_dirs(
        csv_path=csv_path,
        classification_method='ai_based',
        output_dir='result')

    # 출력 디렉토리들 생성
    os.makedirs(output_dir, exist_ok=True)

    # 태그별 데이터 그룹화
    tag_groups = {}

    for item in tagged_data:
        data_id = item['id']
        tags = item['tags']
        reason = item['reason']

        # 원본 데이터 찾기
        original_data = original_df[original_df['id'] == data_id].copy()
        if original_data.empty:
            continue

        # 분석 이유 추가
        original_data['analysis_reason'] = reason

        # 각 태그에 대해 처리
        for tag in tags:
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append(original_data)

    # 태그별로 CSV 파일 저장
    for tag, data_list in tag_groups.items():
        if data_list:
            combined_data = pd.concat(data_list)
            # 태그 이름에서 CSV 파일명에 적합하지 않은 문자 제거
            safe_tag_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in tag)
            output_path = os.path.join(output_dir, f"{safe_tag_name}.csv")

            combined_data.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"'{tag}' 태그 파일 저장 완료: {output_path}")

    return tag_groups


def main():
    # 타겟 csv 파일 이름
    csv_name = 'genesis_compact.csv'

    # 룰 기반 분석
    analyze_by_rule(csv_path=f"dataset/{csv_name}")
    

    # AI 기반 분석

    # 분류할 태그 정의
    tag_definitions = """
    - "무사고": 사고나 침수 등이 난 적이 없는 매물입니다. 텍스트에서 사고가 없었다는 점을 강조할 확률이 높습니다. 데이터에는 사고 유무, 정비, 사고 관련 수치가 포함되어 있습니다.
    - "튜닝": 차량에 다양한 추가적인 옵션이 많이 달려있는 매물입니다. 텍스트에서 옵션이 많다는 점을 강조할 확률이 높습니다. 데이터에는 각 옵션 유무가 포함되어 있습니다.
    - "1인사용": 차량의 소유주가 변경된 적이 없는 매물입니다. 텍스트에서 1인소유, 1인사용 등을 강조할 확률이 높습니다. 데이터의 '소유자변경' 컬럼을 참고하면 좋습니다.
    - "출퇴근용": 주로 출퇴근 목적으로 사용된 것으로 보이는 차량입니다. 연간 주행거리가 15,000km~25,000km 범위이며, 주행 패턴이 규칙적이거나, 텍스트에 "출퇴근", "통근", "직장", "회사" 등의 단어가 포함되어 있을 확률이 높습니다.
    - "나들이용": 주로 여가 활동이나 주말 사용 목적으로 운행된 차량입니다. 주행거리가 연간 10,000km 이하이고, 텍스트에 "주말", "나들이", "여행", "캠핑", "레저", "가족" 등의 단어가 포함되어 있을 확률이 높습니다.
    - "연식대비적은마일리지": 차량 연식에 비해 주행거리가 현저히 적은 차량입니다. 자세한 기준은 다음과 같습니다:
        * 5년 미만 차량: 연평균 주행거리 10,000km 이하
        * 5-10년 차량: 연평균 주행거리 8,000km 이하
        * 10년 이상 차량: 연평균 주행거리 5,000km 이하
    - "전문정비": 정기적으로 전문 정비소나 제조사 서비스센터에서 관리되었거나 딜러가 상태가 좋다고 보증한 차량입니다. 텍스트에 "제조사", "서비스센터", "정비소", "직영점", "정비기록", "정식 딜러", "딜러 보증" 등의 단어가 포함되어 있을 확률이 높습니다.
    - "급매물": 시장 평균 가격보다 현저히 낮은 가격에 판매되거나, 빠른 판매를 원하는 판매자의 차량입니다. 가격' 컬럼이 동일 차종/연식/조건 대비 15% 이상 저렴하거나 텍스트에 "급매", "급처", "급히", "빨리", "즉시", "네고가능", "가격조정", "대폭할인" 등의 급한 상황(이사, 이직, 급전 등)을 암시하는 단어가 포함되어 있을 확률이 높습니다.
    """
    
    # (rule-based로 하는게 더 나아서 빠진 태그들)
    # "최신연식": 현재 기준으로 비교적 최근에 출시된 모델로, 신형 차량에 속합니다. 현재 연도 기준 3년 이내의 차량이거나, 텍스트에 "신형", "최신형", "신차", "최신 모델" 등의 단어가 포함될 확률이 높습니다.
    

    analyze_and_save(
        csv_path=f"dataset/{csv_name}",
        text_column="설명글",
        tag_definitions=tag_definitions
    )


if __name__ == '__main__':
    main()