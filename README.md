# ✈️ MBTI 기반 여행지 추천 서비스

이 프로젝트는 사용자의 자유로운 텍스트 입력을 통해 심리적 성향을 분석하고, 해당 MBTI 유형에 기반하여 맞춤 여행지를 추천해주는 AI 웹 서비스입니다.  

---

## 🔍 프로젝트 개요

- **기능**
  - 사용자 텍스트를 분석하여 MBTI 성격 유형을 추론합니다.
  - MBTI에 따라 여행지를 추천합니다.
  - 프론트엔드는 React, 백엔드는 FastAPI로 구성되어 있습니다.
  - AI 모델은 사전학습된 모델을 기반으로 학습하고, PyTorch 및 Hugging Face Transformers를 사용합니다.

## 🧠 AI 모델 설명

- **사용 모델**
- `mbti_model`: 텍스트 → MBTI 예측
- `m2m_model`: 다국어 텍스트 번역 (m2m100 기반)
- **프레임워크**
- Hugging Face Transformers
- PyTorch
- **모델 파일**
- 용량 제한으로 인해 GitHub에는 포함하지 않았습니다.
- 아래 Google Drive 링크에서 다운로드할 수 있습니다.  
  👉 [모델 다운로드 링크](https://drive.google.com/drive/folders/1Pr7FyM66NTphVIjpQAlgQRzzQbv8g0lU)

---

## 💻 기술 스택

| 구분       | 사용 기술                          |
|------------|------------------------------------|
| 프론트엔드 | React.js                            |
| 백엔드     | FastAPI, Python                    |
| 모델 학습  | PyTorch, Transformers (HuggingFace)|
| 기타       | Google Colab, GitHub               |

---

## 📁 폴더 구조
mbti_travel/
├── chat-mbti/ # 프론트엔드 코드
├── server/ # 백엔드 코드 (FastAPI)
└── README.md # 프로젝트 설명 파일


## 📌 포트폴리오 목적

- 자연어 처리 기반 사용자 분석 기술 구현
- AI 모델을 활용한 개인화 추천 시스템 제작 경험
- 프론트엔드와 백엔드를 연결한 통합 서비스 구현 역량 증명

