# ToxHerb-Net: Donguibogam-inspired Vision-based Discriminator Network for Poisonous vs Medicinal Herb Classification
*(동의보감 기반 독초·약초 이미지 분류 모델)*

ToxHerb-Net은 동의보감 기반 독초 판별 이미지 데이터를 활용해 독초(Poisonous)와 약초(Medicinal)를 구분하는 이미지 분류 모델입니다. 대규모 이미지 데이터셋에서 사전 학습된 벤치마크 모델들을 백본(backbone)으로 선정한 뒤, 약 600GB 규모의 독초 판별 이미지 데이터로 전이학습을 수행하여 최적의 모델을 선택했습니다. EfficientNet 백본 기반 실험에서는 Top-1 Accuracy 88.932%를 기록하며, 독초·약초 분류 성능을 확인했습니다.

## 🏆 Award
### 수상
- **대회명**: 2021 인공지능 동의보감 독초 판별 해커톤
- **기간**: 2021.11 ~ 2021.11
- **주최**: 과학기술정보통신부
- **수상**: 🥉 **3등상**

## ⚙️ Tech Stacks
- EfficientNet-B4
- EfficientNet-B5
- ResNet-50
- Squeeze-and-Excitation (SE) Block
- PyTorch
- Python
- TensorFlow

## ✨ Features
1. EfficientNet 기반 backbone을 중심으로 모델 구성 및 비교 실험
2. SE Block 적용을 통한 분류 성능 향상
3. 약 600GB 규모의 대규모 데이터셋 전이학습을 통해 최적 모델 선정

## 🏗️ Architecture
<img src="https://github.com/MonoHaru/ToxHerb-Net/blob/main/assets/process.png" alt="process" width="800">

## 🔮 Future Work
1. 과적합(Overfitting) 완화 및 성능 향상을 위한 다양한 freeze/fine-tuning 전략 실험
2. 대규모 독초 판별 데이터셋의 클래스/구간별 이미지 수 불균형 완호(수량 정규화)하여 일반화 성능 향상

## 📜 License
The code in this repository is released under the Apache License.