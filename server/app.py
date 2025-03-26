
import torch
import uvicorn

from fastapi import FastAPI
from transformers import AlbertTokenizer, AlbertForSequenceClassification, M2M100ForConditionalGeneration, M2M100Tokenizer
from fastapi.middleware.cors import CORSMiddleware
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 접근 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)


TOKENIZER = AlbertTokenizer.from_pretrained('./model/mbti')
MBTI_MODEL = AlbertForSequenceClassification.from_pretrained('./model/mbti')

m2m_tokenizer = M2M100Tokenizer.from_pretrained("./model/m2m")
m2m_model = M2M100ForConditionalGeneration.from_pretrained("./model/m2m")
quantized_m2m = torch.quantization.quantize_dynamic(
    m2m_model,  # 모델
    {torch.nn.Linear},  # 양자화할 모듈의 유형
    dtype=torch.qint8  # 양자화할 데이터 타입 (8비트 정수)
)
mbti_labels = ['INTP','ISTP','ENTP','ESTP','INFP','ISFP','ENFP','ESFP', 'INTJ','ISTJ','ENTJ','ESTJ','INFJ','ISFJ','ENFJ','ESFJ']
mbti_probs = {'INTP': 0,
              'ISTP': 0,
              'ENTP': 0,
              'ESTP': 0,
              'INFP': 0,
              'ISFP': 0,
              'ENFP': 0,
              'ESFP': 0,
              'INTJ': 0,
              'ISTJ': 0,
              'ENTJ': 0,
              'ESTJ': 0,
              'INFJ': 0,
              'ISFJ': 0,
              'ENFJ': 0,
              'ESFJ': 0}

class PredictionRequest(BaseModel):
    text: list



@app.post("/predict")
def predict(request: PredictionRequest):
    param = request.dict()
    mbti_result = mbti_probs.copy()

    result = []

    # mbti 질문만 리스트로 저장
    mbti_questions = param['text'][:2]

    # 여행 거리
    # distance = param['text'][3]

    # mbti 예측 및 결과값 저장
    for idx, question in enumerate(mbti_questions):
        text = change_language(question)
        inputs = TOKENIZER(text, return_tensors="pt")
        result = []
        with torch.no_grad():
            outputs = MBTI_MODEL(**inputs)
            logits = outputs.logits

        # 소프트맥스 함수 적용
        softmax_probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        # 각 MBTI 유형에 대한 확률 출력
        for label, prob in zip(mbti_labels, softmax_probs):
            mbti_result[label] += float(Decimal(prob.item()).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
            if idx == 1:
                mbti_result[label] = round(mbti_result[label]/(idx+1), 2)

    sorted_mbti_probs = dict(sorted(mbti_result.items(), key=lambda item: item[1], reverse=True))

    for key, value in sorted_mbti_probs.items():
        if key in mbti_labels:
            result.append((key, value))

    return {"mbti": result}


def change_language(ko_text):
    # 바꿀 문장 선택
    target_language = "en"

    # 소스 언어 설정
    m2m_tokenizer.src_lang = "ko"

    # 입력 문장을 토크나이징
    encoded_input = m2m_tokenizer(ko_text, return_tensors="pt")

    # 타겟 언어의 토큰 ID 설정
    target_lang_token_id = m2m_tokenizer.get_lang_id(target_language)

    # 번역 수행
    generated_tokens = quantized_m2m.generate(
        **encoded_input,
        forced_bos_token_id=target_lang_token_id
    )

    # 번역 결과 디코딩
    translated_text = m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_text[0]


if __name__:
    uvicorn.run(app, host='0.0.0.0', port=8000)
