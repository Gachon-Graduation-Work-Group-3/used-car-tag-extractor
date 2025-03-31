import pandas as pd
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 .env 파일 로드 (api key를 숨기기 위한 용도)
load_dotenv()

def analyze_large_dataset(csv_path, text_column, tag_definitions, batch_size=100, api_key=None):
    # API 키 설정
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    # OpenAI 클라이언트 초기화 (한 번만 생성)
    client = OpenAI(api_key=api_key)

    # 전체 데이터 로드
    full_df = pd.read_csv(csv_path)

    # ID 컬럼 확인/생성
    if 'id' not in full_df.columns:
        full_df['id'] = full_df.index.astype(str)

    # 결과 저장용 리스트
    all_tagged_data = []

    # 총 배치 개수 정리
    total_batches = (len(full_df) + batch_size - 1) // batch_size

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(full_df))

        print(f"배치 {i + 1}/{total_batches} 처리 중... ({start_idx}~{end_idx})")

        # 배치 추출
        batch_df = full_df.iloc[start_idx:end_idx].copy()

        # 임시 CSV 저장
        temp_csv = f".temp/temp_batch_{i}.csv"
        os.makedirs(".temp/", exist_ok=True)
        batch_df.to_csv(temp_csv, index=False)

        # 배치 데이터를 JSON으로 변환
        batch_json = batch_df.to_json(orient='records', indent=2)

        # 프롬프트 생성
        prompt = create_prompt(batch_json, text_column, tag_definitions)
        print(f"배치 {i + 1} prompt length: {len(prompt)}")

        # API 호출
        try:
            # print("test")
            # response = client.responses.create(
            #     model="gpt-4o-mini",
            #     input="Write a one-sentence bedtime story about a unicorn."
            # )
            # print(response.output_text)
            # return None

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
            except json.JSONDecodeError:
                print(f"배치 {i + 1} JSON 파싱 오류. 원본 응답:")
                print(content)

        except Exception as e:
            print(f"배치 {i + 1} 처리 중 오류 발생: {str(e)}")

        # 임시 파일 삭제
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    # 전체 결과 저장
    if all_tagged_data:
        tag_groups = save_tagged_data_to_csv(all_tagged_data, full_df)
        return tag_groups
    else:
        return None


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
    4. 데이터셋 전체 맥락에서 특징적인 항목만 태그를 부여하세요.

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


def analyze_and_save_tagged_data(csv_path, text_column, tag_definitions, api_key=None):
    # 원본 데이터 로드
    df = pd.read_csv(csv_path)

    # API 분석 수행
    tagged_data = analyze_large_dataset(csv_path, text_column, tag_definitions, api_key=api_key)

    if tagged_data:
        # 태그별 CSV 파일 저장
        tag_groups = save_tagged_data_to_csv(tagged_data, df)
        return tag_groups
    else:
        print("\n분석 결과가 없습니다.")
        return None



def save_tagged_data_to_csv(tagged_data, original_df, output_dir="/tagged_dataset"):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 태그별 데이터 그룹화
    tag_groups = {}

    for item in tagged_data:
        data_id = item['id']
        tags = item['tags']
        reason = item['reason']

        # 원본 데이터 찾기
        original_data = original_df[original_df['id'] == data_id].copy()
        if len(original_data) == 0:
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
            output_path = os.path.join(output_dir, f"tag_{safe_tag_name}.csv")
            combined_data.to_csv(output_path, index=False)
            print(f"'{tag}' 태그 파일 저장 완료: {output_path}")

    return tag_groups


def main():
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
    - "최신연식": 현재 기준으로 비교적 최근에 출시된 모델로, 신형 차량에 속합니다. 현재 연도 기준 3년 이내의 차량이거나, 텍스트에 "신형", "최신형", "신차", "최신 모델" 등의 단어가 포함될 확률이 높습니다.
    - "전문정비": 정기적으로 전문 정비소나 제조사 서비스센터에서 관리되었거나 딜러가 상태가 좋다고 보증한 차량입니다. 텍스트에 "제조사", "서비스센터", "정비소", "직영점", "정비기록", "정식 딜러", "딜러 보증" 등의 단어가 포함되어 있을 확률이 높습니다.
    - "급매물": 시장 평균 가격보다 현저히 낮은 가격에 판매되거나, 빠른 판매를 원하는 판매자의 차량입니다. 가격' 컬럼이 동일 차종/연식/조건 대비 15% 이상 저렴하거나 텍스트에 "급매", "급처", "급히", "빨리", "즉시", "네고가능", "가격조정", "대폭할인" 등의 급한 상황(이사, 이직, 급전 등)을 암시하는 단어가 포함되어 있을 확률이 높습니다.
    """

    # 함수 실행
    analyze_and_save_tagged_data(
        csv_path="dataset/genesis_mid.csv",
        text_column="설명글",
        tag_definitions=tag_definitions
    )


if __name__ == '__main__':
    main()