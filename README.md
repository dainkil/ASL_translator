# ASL_translator

# Two-Track 기반 End-to-End 수어 번역 시스템

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface" />
  <img src="https://img.shields.io/badge/MediaPipe-Holistic-00BCD4" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

<p align="center">
  Development of a Two-Track Based End-to-End Sign Language Translation System<br>
  고해상도 영상 특징 추출과 경량 랜드마크 모델의 병렬 설계를 통한 성능 최적화 연구
</p>

<p align="center">
  DAT 6기 · 수자인(Sujain) 팀 · 2025.12
</p>

---

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [핵심 기여](#핵심-기여)
- [시스템 아키텍처](#시스템-아키텍처)
- [데이터셋](#데이터셋)
- [실험 결과](#실험-결과)
- [기술 스택](#기술-스택)
- [팀원](#팀원)
- [보고서](#보고서)
- [참고 문헌](#참고-문헌)

---

## 프로젝트 개요

수어(Sign Language)는 손동작뿐 아니라 얼굴 표정, 시선, 신체 방향이 복합적으로 결합된 고차원적 시각 언어입니다. 음성·텍스트 기반 NLP 기법을 직접 적용하기 어렵고, 양손의 비동기적 움직임과 시계열·공간 정보를 동시에 처리해야 한다는 고난도 문제를 내포하고 있습니다.

본 연구는 **RGB 기반(Track A)** 과 **랜드마크 기반(Track B)** 두 가지 접근 방식을 병렬로 실험하는 **Two-Track 전략**을 채택했습니다. 이를 통해 입력 모달리티에 따른 번역 성능 차이를 체계적으로 분석하고, 수어 번역 모델이 지향해야 할 정확도·연산 효율성·실시간성 간의 균형점을 탐색했습니다.

---

## 핵심 기여

- **Two-Track 병렬 설계**: RGB 영상 기반(Track A)과 랜드마크 기반(Track B) 두 파이프라인을 독립적으로 구현하고 동일한 데이터·평가 기준 하에 비교 분석
- **오프라인 특징 추출 전략 (Track A)**: 제한된 GPU 환경에서 VideoMAE-Base를 동결(Frozen)하고 `.npy` 형태로 특징을 사전 저장하여 학습 비용 절감
- **Multi-Stream Embedding (Track B)**: 얼굴·포즈·양손을 별도 스트림으로 분리 처리하는 Conv1d 기반 임베딩으로 신체 부위별 미세 패턴 포착
- **SASS 적용 (Track B)**: Stochastic Adaptive Sequence Sampling으로 가변 길이 수어 시퀀스를 균일하게 정규화하여 학습 안정성 확보
- **LLM 검증 실험**: 명시적 언어학적 피처 주입(품사·구문 정보)이 번역의 의미적 전달력(METEOR +14.6%)에 미치는 효과를 별도 실험군(G0~G3)으로 검증

---

## 시스템 아키텍처

### Track A — RGB 기반 파이프라인
```
Sign Video (RGB)
↓
Frame Sampling (64 frames)
↓
VideoMAE-Base [Frozen] ← Kinetics-400 pretrained
↓  (32 × 768 feature)
Adapter Module
├─ Linear Projection
├─ LayerNorm + ReLU
├─ Positional Encoding (sinusoidal)
└─ Transformer Encoder (4 layers)
↓
T5-Base Decoder (beam search, size=4)
↓
번역 문장 (English)
```
대규모 비전 모델의 표현력을 유지하면서 Adapter만 학습하는 구조를 채택하여, 제한된 GPU 환경에서도 영상 기반 번역이 가능하도록 설계했습니다.

---

### Track B — 랜드마크 기반 파이프라인
```
Sign Video (RGB)
↓
MediaPipe Holistic
├─ Pose        : 33개
├─ L.Hand      : 21개
├─ R.Hand      : 21개
└─ Face        : 468개
→ 총 543개 추출, 핵심 95개로 축소
↓
정규화
├─ Body normalization (신체 중심 기반 스케일링)
├─ Hand min-max normalization
└─ Face min-max normalization
↓
SASS (Stochastic Adaptive Sequence Sampling)
↓
Multi-Stream Embedding (Body / L.Hand / R.Hand / Face)
└─ Conv1d + Residual Connection + GELU
↓
Transformer Encoder (4 layers, 8-head self-attention, d_model=512, FFN=2048)
↓
Projection Layer
↓
T5-Small Decoder
↓
번역 문장 (English)
```
RGB 영상 대비 훨씬 가벼운 랜드마크 좌표만을 사용하되, Multi-Stream 구조를 통해 신체 부위별 구조적 패턴을 효과적으로 학습하고 실시간 추론이 가능하도록 설계했습니다.

---

## 데이터셋

| 데이터셋 | 규모 | 도메인 | 사용 Track |
|---------|------|--------|-----------|
| [How2Sign](https://how2sign.github.io/) (Duarte et al., 2021) | ~80시간 / 35,191 문장 쌍 | 일상 대화·교육 | A, B |
| [DailyMoth-70h](https://www.dailymoth.com/) (Uthus et al., 2023) | ~70시간 / 47,658 문장 쌍 | 뉴스 | B |

두 데이터셋 모두 ASL–영어 병렬 코퍼스로, 문장 단위 수어 영상과 영어 번역문이 정렬되어 있습니다.

**전처리 파이프라인**

1. **영상–텍스트 구간 정합 및 Manifest 구축**: 각 샘플의 영상 경로와 정렬된 영어 문장을 통합 참조 구조(manifest)로 구성. Track A·B가 동일한 영상–텍스트 매칭 정보를 공유하면서 독립적으로 데이터 로딩 가능하도록 설계
2. **랜드마크 추출 및 시계열화 (Track B)**: MediaPipe Holistic으로 프레임당 543개 좌표 추출 → 핵심 95개 선택 → 고정 길이 시계열로 정렬 후 `.npy` 저장
3. **영상 특징 추출 및 벡터화 (Track A)**: 프레임 샘플링 후 VideoMAE로 잠재 표현 추출 → `(32 × 768)` 시계열 벡터로 `.npy` 저장

---

## 실험 결과

### Track B 최종 모델 (Exp-8: lr=5e-5, Encoder Layers=4)

| 지표 | 값 | 비고 |
|------|-----|------|
| BLEU-4 | **44.41** | n-gram 정밀도 |
| ROUGE-L | 13.08 | 문장 구조 재현율 |
| METEOR | 8.26 | 의미적 유사도 |
| Inference Time | 38.74 ms/sample | 실시간 처리 수준 |
| Throughput | 25.81 samples/sec | 엣지 디바이스 구동 가능 |
| Total Parameters | 74,403,840 | 경량 모델 |
| GPU Memory (Peak) | 739.43 MB | 메모리 효율적 |

### Track A 최종 모델

| 지표 | 값 |
|------|-----|
| BLEU-4 | 1.66 |
| 주요 현상 | Overfitting (Epoch 6 이후 Validation Loss 상승), Fluent Hallucination |

### LLM 검증 실험 — 명시적 피처 주입 효과 (ASLG-PC12 기반)

| 그룹 | 설명 | BLEU-4 | METEOR |
|------|------|--------|--------|
| G0 (Baseline) | 기본 Gloss 데이터, 피처 없음 | 2.61 | 13.14 |
| G1 (Augmentation) | 언어학적 데이터 증강 (기능어 삽입, 어순 재배열) | - | - |
| G2 (Features) | 품사·구문 정보를 담은 명시적 피처 벡터 주입 | 2.02 | **15.06** |
| G3 (Hybrid) | G1 + G2 혼합 | - | - |

G2에서 BLEU는 소폭 낮아졌으나 METEOR가 약 14.6% 상승했습니다. 피처 주입이 단순 암기를 넘어 문맥 파악과 유의어 선택(Lexical Choice)을 유도하는 데 효과적임을 확인했습니다.

---

### 분석 및 한계

**Track A**: 일반 행동 인식 데이터(Kinetics-400)로 사전학습된 VideoMAE가 수어의 미세한 손가락 움직임(Fine-grained Motion)을 충분히 포착하지 못해, 시각 정보와 텍스트 간 정렬(Alignment)에 실패하는 *Fluent Hallucination* 현상이 발생했습니다.

**Track B**: BLEU-4(44.41)와 ROUGE-L(13.08)의 큰 괴리가 나타났으며, 정성 분석 결과 T5 Decoder가 랜드마크 특징보다 학습 데이터의 텍스트 분포(뉴스·정치 도메인)에 과도하게 의존하는 *Language Model Bias*가 확인되었습니다.

---

## 기술 스택

| 분야 | 사용 기술 |
|------|----------|
| 딥러닝 프레임워크 | PyTorch 2.0 |
| 모델 | VideoMAE-Base (Kinetics-400), T5-Base / T5-Small |
| 라이브러리 | HuggingFace Transformers |
| 랜드마크 추출 | MediaPipe Holistic |
| 영상 처리 | Decord |
| 실험 환경 | Google Colab Pro, NVIDIA T4 GPU |
| 평가 지표 | sacreBLEU (BLEU-4), ROUGE-L, METEOR |

---


## 보고서

[DAT_6회_캡스톤_프로젝트_보고서_수자인팀.pdf](./DAT_6회_캡스톤_프로젝트_보고서_수자인팀.pdf)

---

## 참고 문헌

- Tong et al. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS.
- Raffel et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.
- Duarte et al. (2021). How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language. CVPR.
- Hwang et al. (2025). An Efficient Gloss-Free Sign Language Translation Using Spatial Configurations and Motion Dynamics with LLMs. NAACL.
- Guewou et al. (2025). SignMusketeers: An Efficient Multi-Stream Approach for Sign Language Translation at Scale. arXiv:2406.08097.
- Kim et al. (2022). Keypoint based Sign Language Translation without Glosses. arXiv:2204.10511.
