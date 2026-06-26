# BCI & Neuroengineering Self-Study Curriculum

> 백엔드 개발자가 Brain-Computer Interface(BCI)와 신경공학을 처음부터 따라갈 수 있도록 만든 학습 가이드. 환경 셋업부터 SNN, EEG 분석, 비침습 자극 시뮬레이션까지 다룹니다.
>
> **학습 방식**: 각 Step은 ① 개념 영상 → ② 코드 골격(`TODO`로 일부 비어 있음) → ③ 직접 채우기 → ④ 확장 과제(Challenges) 4단계입니다. 빈칸을 채워보고 막히면 [References](#references) 영상을 다시 보세요.

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#)

---

## 📑 Table of Contents

- [Overview](#overview)
- [How to Use](#how-to-use)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Glossary](#glossary)
- [Curriculum](#curriculum)
  - [Part 1. Foundations](#part-1-foundations)
  - [Part 2. EEG & Motor Imagery BCI](#part-2-eeg--motor-imagery-bci)
  - [Part 3. Spiking Neural Networks](#part-3-spiking-neural-networks)
  - [Part 4. Circuit Decoding & Spatial Neurons](#part-4-circuit-decoding--spatial-neurons)
  - [Part 5. Deep Learning for EEG](#part-5-deep-learning-for-eeg)
  - [Part 6. Non-invasive Stimulation](#part-6-non-invasive-stimulation)
  - [Part 7. Real-time Closed-loop Systems](#part-7-real-time-closed-loop-systems)
- [Final Project Ideas](#final-project-ideas)
- [Datasets](#datasets)
- [Library Cheat Sheet](#library-cheat-sheet)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

| Part | Topic | Steps |
|------|-------|-------|
| 1 | Foundations (neuroscience, environment) | 2 |
| 2 | EEG & Motor Imagery BCI | 4 |
| 3 | Spiking Neural Networks | 3 |
| 4 | Circuit Decoding | 2 |
| 5 | Deep Learning for EEG | 2 |
| 6 | Non-invasive Stimulation | 3 |
| 7 | Real-time Closed-loop | 3 |

---

## How to Use

각 Step의 코드 블록은 **완성본이 아니라 골격**입니다. `TODO`와 `...`을 직접 채우세요.

1. **Watch** — 영상으로 개념을 머리에 넣습니다.
2. **Code skeleton** — 빈칸(`TODO`, `...`)을 채워 실행 가능한 코드로 만듭니다.
3. **Run & Verify** — 결과가 영상에서 본 것과 같은지 확인합니다.
4. **Challenges** — 변형·확장 과제를 풀어 응용력을 만듭니다. 최소 1개, 가능하면 3개 이상 풀기를 권장.
5. **Notes** — `notes/stepNN.md`에 막힌 부분과 해결법을 기록하세요. 이게 가장 큰 자산이 됩니다.

> 💡 빈칸을 어떻게 채울지 모를 때: 해당 라이브러리 공식 문서 검색 → API signature 확인 → 영상의 코드와 비교.

---

## Prerequisites

- 기본적인 Python 사용 경험
- 고등학교 수준 미적분 (미분방정식 개념)
- 선형대수 기초 (벡터·행렬)
- Git 사용 가능
- OS: Windows 10+, macOS 11+, Ubuntu 20.04+

---

## Installation

### 1. Conda 환경 생성

```bash
# Anaconda: https://www.anaconda.com/download
conda create -n bci python=3.11 -y
conda activate bci
```

### 2. 라이브러리 설치

```bash
# 공통
pip install numpy scipy matplotlib pandas jupyter scikit-learn seaborn

# EEG 분석
pip install mne moabb braindecode

# 딥러닝 + SNN
pip install torch torchvision
pip install snntorch brian2
```

### 3. 작업 폴더 구조

```
bci_study/
├── step01_neuroscience/
├── step02_mne_first/
├── ...
├── data/                # 모든 데이터셋 저장
└── notes/               # 개념 정리 노트 (필수)
```

### 4. Jupyter 실행

```bash
cd ~/bci_study
jupyter notebook
```

### 5. 동작 확인

```bash
python -c "import mne, torch, snntorch, brian2; print('OK')"
```

---

## Glossary

| 용어 | 한 줄 설명 |
|------|-----------|
| **EEG** | Electroencephalography. 두피 전극으로 측정하는 뇌 표면 전기 신호 (μV 단위). |
| **BCI** | Brain-Computer Interface. 뇌 신호 → 컴퓨터 명령 변환 시스템. |
| **SNN** | Spiking Neural Network. 실제 뉴런처럼 스파이크(0/1)로 통신하는 신경망. |
| **LIF Neuron** | Leaky Integrate-and-Fire. 전압을 누적하다가 임계값을 넘으면 스파이크 발사, 평소엔 leak. |
| **STDP** | Spike-Timing-Dependent Plasticity. 스파이크 발생 순서에 따라 시냅스 가중치가 변하는 학습 규칙. |
| **Euler Method** | 미분방정식 수치해법 — `다음값 = 현재값 + 기울기 × dt`. |
| **ICA** | Independent Component Analysis. EEG 잡음 성분(눈깜빡임·EMG) 분리. |
| **ERD/ERS** | Event-Related De-/Synchronization. 운동 상상 시 특정 주파수 전력 변화. |
| **P300** | 자극 후 ~300ms에 나타나는 양의 EEG 정점. |
| **CSP** | Common Spatial Pattern. 운동 상상 EEG의 클래시컬 특징 추출법. |
| **tDCS / tACS / TI** | 두피 전기 자극 — 직류 / 교류 / Temporal Interference. |
| **MOABB** | Mother of All BCI Benchmarks. BCI 데이터셋 통합 인터페이스. |
| **EEGNet** | 모터 이미저리 EEG 분류용 경량 CNN. BCI 표준 베이스라인. |
| **Place Cell / RSC** | 해마·뇌량팽대후피질의 공간 정보 담당 뉴런. 2014 노벨상. |
| **Surrogate Gradient** | 스파이크 함수의 미분 불가능을 우회하는 SNN 학습 기법. |

---

## Curriculum

### Part 1. Foundations

#### Step 1 — Neuroscience Primer

**Goal**: 뉴런이 어떻게 신호를 보내는지 한 문장으로 설명할 수 있다.

**Watch**
- 🇰🇷 [EBSi 뉴탐스런 생명과학 33강 — 자극의 전달](https://www.youtube.com/watch?v=8B6itAOFzg8)
- 🇰🇷 [북툰 — 15분 만에 뇌과학 입문](https://www.youtube.com/watch?v=_ADYQgL6g_Y)
- 🇺🇸 [Khan Academy — Neuron action potentials](https://www.khanacademy.org/science/biology/human-biology/neuron-nervous-system/v/neuron-action-potential-description)

**Concepts**: membrane potential, action potential, synapse, excitation/inhibition.

**Challenges**
1. 뉴런이 신호를 보내는 과정을 5개 단어(예: depolarization, threshold, …)로만 요약해서 노트에 적기.
2. 흥분성/억제성 시냅스의 차이를 본인 회사 시스템(서비스 호출·차단)에 비유해서 설명.
3. EEG가 측정하는 것이 단일 뉴런인가, 집단인가? 왜 그런지 노트에 정리.

#### Step 2 — Numerical Methods (Euler)

**Goal**: 오일러법으로 미분방정식을 풀 수 있다. SNN 시뮬레이션의 기반.

**Watch**
- 🇰🇷 [칸아카데미 한국어 — 오일러법](https://ko.khanacademy.org/math/integral-calculus/ic-diff-eq/ic-eulers-method/v/eulers-method)
- 🇰🇷 [피토스터디 — 전진/후진 오일러법](https://www.youtube.com/watch?v=Kusxa83osyQ)
- 🇰🇷 [피토스터디 — 수정 오일러법](https://www.youtube.com/watch?v=Q13gjmRt4rc)

**Concepts**: dt (step size), forward/backward Euler, truncation error.

**Code skeleton** — `02_euler.ipynb` (지수 감쇠 `dy/dt = -k·y` 풀기)

```python
import numpy as np
import matplotlib.pyplot as plt

k = 0.5
y0 = 10.0
t_end = 10.0
dt = 0.1

# TODO: forward Euler 루프 작성
#   y[i+1] = y[i] + dt * (-k * y[i])
n = int(t_end / dt)
t = np.linspace(0, t_end, n)
y = np.zeros(n); y[0] = y0
for i in range(n - 1):
    y[i+1] = ...

# 해석해와 비교
y_exact = y0 * np.exp(-k * t)
plt.plot(t, y, label='Euler'); plt.plot(t, y_exact, '--', label='Exact')
plt.legend(); plt.show()
```

**Challenges**
1. `dt`를 0.5, 0.1, 0.01로 바꾸면서 해석해와 오차가 어떻게 변하는지 표로 정리.
2. 같은 ODE를 backward Euler로 다시 풀고 두 결과를 한 그래프에 그리기.
3. 진자 운동 `d²θ/dt² = -(g/L)·sin(θ)`을 2차 시스템 → 1차 두 개로 분해해서 풀고 시각화.

---

### Part 2. EEG & Motor Imagery BCI

#### Step 3 — First EEG Visualization with MNE

**Goal**: EEG raw 데이터를 열어서 채널 수와 샘플링 주파수를 확인하고 그래프를 그린다.

**Watch**
- 🇺🇸 [Basics of EEG BCI](https://www.youtube.com/watch?v=rFFmfNzpfnY)
- 🇺🇸 [MNE Tutorial in Python](https://www.youtube.com/watch?v=lEUmgSFQaAY)
- 🇺🇸 [Berdakh Abibullaev — MNE L8 Part 1](https://www.youtube.com/watch?v=IYuAPisoUeI)

**Dataset**: MNE 샘플 (자동 다운로드, ~1GB)

**Code skeleton** — `01_first_eeg.ipynb`

```python
import mne

sample_path = mne.datasets.sample.data_path()
raw_fname = sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
print(raw.info)

# TODO: 처음 10초 / 20채널만 시각화
raw.plot(duration=..., n_channels=...)

# TODO: 1~40 Hz 대역통과 필터 적용 (raw.filter)
...

# TODO: 필터 전/후 차이를 비교하려면? raw.copy()로 원본을 보존하고 두 객체를 따로 plot
```

**Challenges**
1. `raw.info['sfreq']`, `raw.info['nchan']`을 변수로 저장하고 출력.
2. EEG 채널만 골라서(`raw.pick_types(eeg=True, meg=False)`) 다시 그려보고 채널 수가 얼마나 줄었는지 비교.
3. notch filter 60Hz를 적용한 후 PSD(`raw.plot_psd(fmax=80)`)를 그려서 전원 잡음이 사라졌는지 확인.
4. `preload=False`로 로드하면 어떤 차이가 있는지 적어보기.

#### Step 4 — Motor Imagery (BCI Competition IV 2a)

**Goal**: 4-class 운동 상상 EEG 데이터를 받고 trial 단위로 본다.

**Watch**
- 🇺🇸 [Talha Anwar — Read Motor Imagery EEG BCI Competition IV 2a](https://www.youtube.com/watch?v=Llsfw9RkcIg)
- 🇺🇸 [Ollie D'Amico — Motor Imagery & CSP](https://www.youtube.com/watch?v=tInGKSP65OY)

**Dataset**: BCI Competition IV 2a
- 자동: [MOABB docs](https://moabb.neurotechx.com/docs/index.html)
- 수동: [bbci.de download](https://bbci.de/competition/iv/download/) (420MB)

**Code skeleton** — `02_motor_imagery.ipynb`

```python
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import matplotlib.pyplot as plt

dataset = BNCI2014_001()
dataset.subject_list = [1]

paradigm = MotorImagery(n_classes=4, fmin=8, fmax=30)
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])

print("Shape:", X.shape)
print("Classes:", set(y))

# TODO: 첫 trial의 첫 채널을 그려보기
plt.plot(...)
plt.title(f"Class: {y[0]}"); plt.show()
```

**Challenges**
1. 클래스별 trial 개수를 `collections.Counter`로 출력.
2. `left_hand` 라벨의 trial 5개를 subplot으로 한 figure에 그리기.
3. 클래스별 평균 파형(채널 1개)을 그려서 차이를 눈으로 확인.
4. `subject_list = [1, 2, 3]`으로 늘리고 X 모양이 어떻게 변하는지 적기.
5. `fmin/fmax`를 (1, 40)으로 바꾸면 결과가 어떻게 달라지는지 관찰하고 이유 정리.

#### Step 5 — Filtering & ICA Artifact Removal

**Goal**: ICA로 눈깜빡임 잡음을 식별·제거한다.

**Watch**
- 🇺🇸 [Berdakh — MNE Tutorial Part 1 (ICA)](https://www.youtube.com/watch?v=IYuAPisoUeI)

**Code skeleton** — `03_filter_ica.ipynb`

```python
import mne

sample = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(sample/'MEG'/'sample'/'sample_audvis_raw.fif', preload=True)

# TODO: 60Hz notch + 1~40Hz bandpass 적용
...

ica = mne.preprocessing.ICA(n_components=20, random_state=42)
ica.fit(raw)
ica.plot_components()

# TODO: EOG 성분 자동 검출 → ica.exclude에 넣고 apply
eog_indices, eog_scores = ica.find_bads_eog(raw)
print("EOG components:", eog_indices)
# ica.exclude = ...
# raw_clean = ...
```

**Challenges**
1. `ica.plot_sources(raw)`로 시간축 성분을 그려서 눈깜빡임의 큰 점프가 어디에 있는지 직접 찾기.
2. 자동 검출 결과와 본인이 눈으로 고른 인덱스를 비교.
3. ICA 적용 전/후 raw 데이터를 같은 구간에서 그려 잡음이 줄었는지 확인.
4. `n_components`를 10, 30으로 바꾸면 결과가 어떻게 달라지는지 관찰.

#### Step 6 — Epochs & ERP

**Goal**: 자극 시점의 EEG 평균(evoked) 응답을 시각화한다.

**Watch**
- 🇺🇸 [Pybrain MNE Workshop](https://www.youtube.com/watch?v=t-twhNqgfSY)

**Code skeleton** — `epochs_erp.ipynb`

```python
import mne

sample = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(sample/'MEG'/'sample'/'sample_audvis_raw.fif', preload=True)
raw.filter(1., 40.)

events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'auditory/left': 1, 'auditory/right': 2}

# TODO: 자극 전 200ms ~ 자극 후 500ms 윈도우로 Epochs 생성
epochs = mne.Epochs(raw, events, event_id, tmin=..., tmax=...,
                     preload=True, baseline=(None, 0))

# TODO: auditory/left 평균 evoked 그리기 + 토포맵
evoked = epochs[...].average()
# evoked.plot()
# evoked.plot_topomap(times=[...])
```

**Challenges**
1. `auditory/left`와 `auditory/right`의 evoked를 같은 figure에 겹쳐 그리기 (`mne.viz.plot_compare_evokeds`).
2. visual 자극(event id 3, 4)도 추가하고 청각 vs 시각 토포맵 차이 비교.
3. epoch reject 기준(`reject=dict(eeg=150e-6)`)을 추가해 몇 개 trial이 버려지는지 확인.
4. 특정 시점(P100, N170)의 토포맵을 저장해 노트에 첨부.

---

### Part 3. Spiking Neural Networks

#### Step 7 — LIF Neuron from Scratch

**Goal**: 오일러법으로 LIF 뉴런 하나를 직접 시뮬레이션한다.

**Watch**
- 🇺🇸 [Gerstner Lab — CNS1.3 LIF Model](https://www.youtube.com/watch?v=KGxVwJJC9zs)
- 🇺🇸 [NPTEL — Neuromorphic LIF](https://www.youtube.com/watch?v=gjGTXeJUtCA)
- 🇺🇸 [Arnau Blanco — Integrate-and-Fire in Python](https://www.youtube.com/watch?v=epcQTW9NpPc)

**Code skeleton** — `04_lif_neuron.ipynb`

```python
import numpy as np
import matplotlib.pyplot as plt

# 파라미터
dt, T = 0.1, 200      # ms
tau = 10.0
V_rest, V_th, V_reset = -70, -55, -75
R, I = 10.0, 1.6

n_steps = int(T / dt)
V = np.zeros(n_steps); V[0] = V_rest
spikes = []

# TODO: dV/dt = (-(V - V_rest) + R*I) / tau 를 Euler로 풀기
for t in range(1, n_steps):
    dV = ...
    V[t] = V[t-1] + dV * dt
    # TODO: V가 임계값을 넘으면 spikes에 시각(t*dt) 기록하고 V[t] = V_reset
    if ...:
        ...

plt.plot(np.arange(n_steps) * dt, V)
plt.scatter(spikes, [V_th]*len(spikes), color='red', marker='|', s=200)
plt.xlabel("Time (ms)"); plt.ylabel("V (mV)"); plt.show()
print(f"Total spikes: {len(spikes)}")
```

**Challenges**
1. 입력 전류 `I`를 0.5, 1.0, 1.6, 2.5 nA로 바꿔가며 firing rate(spikes/sec) 곡선을 그리기 (F-I curve).
2. `tau`를 5, 10, 20 ms로 바꾸면 발사 빈도가 어떻게 달라지는지 관찰.
3. 시간에 따라 변하는 입력(`I = 1.6 * np.sin(2*np.pi*5*t/1000)`)을 주입해서 막전위 동기화 시각화.
4. **불응기(refractory period)** 2ms를 구현 — 스파이크 직후 일정 시간 V를 V_reset에 고정.
5. 뉴런 100개를 동시에 시뮬레이션하고 raster plot 그리기 (뉴런마다 다른 `I` 부여).

#### Step 8 — STDP Learning Rule

**Goal**: 스파이크 타이밍에 따라 시냅스 가중치가 어떻게 변하는지 이해한다.

**Watch**
- 🇺🇸 [Gerstner — NDC6.5 STDP](https://www.youtube.com/watch?v=z4vZTeTGiVQ)
- 🇺🇸 [Dan Goodman — Cosyne 2022 SNN](https://www.youtube.com/watch?v=GTXTQ_sOxak)

**Concepts**: pre/post-synaptic spike, LTP, LTD, weight update window.

**Code skeleton** — `08_stdp.ipynb` (STDP 학습 윈도우 함수 그리기)

```python
import numpy as np
import matplotlib.pyplot as plt

A_plus, A_minus = 0.1, 0.12
tau_plus, tau_minus = 20.0, 20.0   # ms

dt = np.linspace(-50, 50, 200)     # post - pre 시각차

# TODO: STDP 함수 작성
#   dt > 0  →  Δw =  A_plus  * exp(-dt / tau_plus)
#   dt < 0  →  Δw = -A_minus * exp( dt / tau_minus)
dw = np.where(dt > 0, ..., ...)

plt.axhline(0, color='gray'); plt.axvline(0, color='gray')
plt.plot(dt, dw); plt.xlabel("t_post - t_pre (ms)"); plt.ylabel("Δw"); plt.show()
```

**Challenges**
1. `tau_plus`, `tau_minus`를 5, 20, 50 ms로 바꿔가며 윈도우 모양 비교.
2. 두 LIF 뉴런(Step 7) 사이에 STDP 시냅스를 만들고 100 trial 동안 weight 변화를 그리기.
3. anti-STDP(부호 반전) 규칙으로 바꾸면 가중치가 어떻게 발산/수렴하는지 관찰.

#### Step 9 — Multi-layer SNN with snntorch

**Goal**: PyTorch 위에서 2-layer SNN을 만든다.

**Watch**
- 🇺🇸 [Jason Eshraghian — Training SNN](https://www.youtube.com/watch?v=zldal7b7sJ4)

**Code skeleton** — `05_snn_basic.ipynb`

```python
import snntorch as snn
import torch
import torch.nn as nn

beta = 0.95

# TODO: 입력 4 → hidden 8 → 출력 2 인 2-layer SNN
# 힌트: nn.Linear → snn.Leaky → nn.Linear → snn.Leaky(output=True)
net = nn.Sequential(
    ...
)

x = torch.randn(1, 4)
spk_rec = []
for step in range(25):
    spk, mem = net(x)
    spk_rec.append(spk)

spk_out = torch.stack(spk_rec)
print("Output spikes:", spk_out.shape)
# TODO: 두 클래스의 스파이크 합을 비교해 더 많이 발사한 쪽을 "예측"으로 출력
```

**Challenges**
1. `beta`를 0.5 / 0.9 / 0.99로 바꿔가며 hidden 뉴런 막전위 trace 시각화.
2. snntorch 공식 튜토리얼의 [Training SNN on MNIST](https://snntorch.readthedocs.io/en/latest/tutorials/index.html)를 끝까지 돌리기.
3. surrogate gradient를 `fast_sigmoid` 대신 `atan`으로 바꿔 학습 곡선 비교.
4. 본인이 만든 LIF(Step 7)와 snntorch `Leaky`의 출력을 같은 입력·파라미터에서 비교.

---

### Part 4. Circuit Decoding & Spatial Neurons

#### Step 10 — Place Cells & Grid Cells

**Goal**: 공간 정보 처리 뉴런과 RSC의 역할을 1분 설명한다.

**Watch**
- 🇺🇸 [Edvard Moser Nobel Lecture — Grid Cells](https://www.youtube.com/watch?v=ywPpiPwUstY)
- 🇺🇸 [O'Keefe Nobel Lecture — Place Cells](https://www.youtube.com/watch?v=PrN1OilSAg8)

**Concepts**: place field, grid cell, head-direction cell, retrosplenial cortex.

**Optional Code skeleton** — `place_field.ipynb` (가상 place field 시뮬레이션)

```python
import numpy as np
import matplotlib.pyplot as plt

# 2m × 2m 환경, 격자 50×50
grid = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(grid, grid)

# TODO: 중심 (cx, cy), 폭 sigma인 가우시안 place field 만들기
cx, cy, sigma = 0.3, -0.2, 0.2
firing_rate = ...

plt.imshow(firing_rate, extent=[-1, 1, -1, 1], origin='lower'); plt.colorbar(); plt.show()
```

**Challenges**
1. place cell 3개를 다른 위치에 놓고 한 figure에 3개 firing map 그리기.
2. 동물 trajectory(랜덤 워크 1000step)를 만들고 각 시점에 firing rate 샘플링 → spike 시뮬레이션.
3. grid cell 패턴(6각 격자)을 sinusoid 합으로 근사해 시각화.
4. RSC가 BCI와 어떤 관련이 있을지 본인 생각을 노트에 3줄로 적기.

#### Step 11 — Population Decoding

**Goal**: N개 뉴런의 동시 활동에서 자극 정보를 분류 알고리즘으로 읽는다.

**Watch**
- 🇺🇸 [Neuromatch Academy — Decoding (W1D3)](https://www.youtube.com/watch?v=KxldhMR5PxA)

**Code skeleton** — `06_population_decoding.ipynb`

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

np.random.seed(0)
n_trials, n_neurons = 100, 50

# TODO: 가상 데이터 만들기
#   X: shape (n_trials, n_neurons), 표준정규
#   y: 절반 0, 절반 1
#   class 1에서만 처음 10개 뉴런에 +1.0 bias (signal)
X = ...
y = ...

clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, X, y, cv=5)
print(f"Decoding accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
```

**Challenges**
1. 신호를 받는 뉴런 개수를 10 → 3, 1로 줄이면 정확도가 어떻게 떨어지는지 그래프로 그리기.
2. bias 크기를 0.1 ~ 2.0 범위에서 sweep하고 SNR과 정확도 관계 시각화.
3. `LogisticRegression` 대신 `RandomForestClassifier`, `SVC(kernel='rbf')`로 성능 비교.
4. confusion matrix를 `sklearn.metrics.confusion_matrix`로 출력.
5. (응용) Step 4의 BCI 데이터를 trial 평균 전력으로 요약한 후 같은 디코더로 분류 시도.

---

### Part 5. Deep Learning for EEG

#### Step 12 — EEGNet on BCI Competition IV 2a

**Goal**: 가장 널리 쓰이는 EEG 분류 CNN인 EEGNet을 학습시킨다.

**Watch**
- 🇺🇸 [Ollie D'Amico — EEGNet](https://www.youtube.com/watch?v=LMQ6Uh-AekY)
- 🇺🇸 [BrainDecode 소개](https://www.youtube.com/watch?v=LwxpDInax44)

**Reference**: [braindecode 공식 예제 — Trialwise Decoding on BCIC IV 2a](https://braindecode.org/0.6/auto_examples/plot_bcic_iv_2a_moabb_trial.html)

**Code skeleton** — `08_eegnet.ipynb`

```python
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor, exponential_moving_standardize
from braindecode.models import EEGNetv4

dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[3])

# TODO: preprocessor 파이프라인 구성
# 1) EEG 채널만 선택, 2) V→μV 스케일, 3) 4~38Hz 필터, 4) exponential standardize
preprocessors = [
    ...
]
preprocess(dataset, preprocessors)

# TODO: EEGNetv4 인스턴스 (22ch, 4 class, 1125 samples)
model = ...
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

**Challenges**
1. 공식 예제를 끝까지 돌리고 cross-subject vs within-subject 정확도 차이를 표로 정리.
2. `ShallowFBCSPNet`도 같은 데이터로 학습해 두 모델 파라미터·정확도 비교.
3. subject 3 외에 9명 전체에 대해 학습 → subject-wise 정확도 분포(boxplot) 그리기.
4. train/val loss 곡선을 저장하고 overfitting 시점 표시.
5. (응용) 학습된 모델 weight를 `torch.save`로 저장 후 Step 18 서버에서 로드해 추론에 사용.

#### Step 13 — Classical Baseline (CSP + LDA)

**Goal**: 딥러닝과 비교할 클래시컬 BCI 파이프라인을 구축한다.

**Code skeleton** — `10_csp_lda.ipynb`

```python
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

paradigm = LeftRightImagery(fmin=8, fmax=30)
X, y, _ = paradigm.get_data(dataset=BNCI2014_001(), subjects=[1])

# TODO: CSP(6 components) → LDA 파이프라인 만들기
clf = Pipeline([
    ...
])

# TODO: 5-fold 평균 정확도 출력
scores = ...
print(f"CSP+LDA accuracy: {scores.mean():.2%}")
```

**Challenges**
1. `n_components`를 2, 4, 6, 8, 10으로 sweep하고 정확도 vs 컴포넌트 수 그래프 그리기.
2. 주파수 대역을 (8,12) μ rhythm만, (13,30) β만, (8,30) 합칠 때 정확도 비교.
3. CSP 필터의 spatial pattern을 `csp.plot_patterns(epochs.info)`로 시각화.
4. Step 12 EEGNet 정확도와 같은 피험자에서 직접 비교 표 만들기.
5. classifier만 `LogisticRegression`, `SVC`로 교체해서 비교.

---

### Part 6. Non-invasive Stimulation

#### Step 14 — TMS / tDCS / TI Basics

**Goal**: 비침습 자극의 종류와 TI의 원리(envelope frequency)를 설명한다.

**Watch**
- 🇺🇸 [Marom Bikson — TI Stimulation](https://www.youtube.com/watch?v=jsyZva93M_8)
- 🇺🇸 [Neuromatch — TI Mechanism](https://www.youtube.com/watch?v=TTl6J7ELsE8)
- 🇺🇸 [NIMH — TMS Basics](https://www.youtube.com/watch?v=io_ganJ0Za0)

**Code skeleton** — `ti_envelope.ipynb` (TI envelope 개념 수치 확인)

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 10000
t = np.linspace(0, 1, fs)

# TODO: 두 고주파 사인파 만들기
#   f1 = 2000 Hz, f2 = 2010 Hz → 차주파수 10 Hz envelope
sig1 = ...
sig2 = ...
sum_sig = sig1 + sig2

plt.plot(t[:500], sum_sig[:500])
plt.title("Sum of 2000Hz + 2010Hz → 10Hz envelope")
plt.show()
```

**Challenges**
1. 두 주파수 차이를 5 / 10 / 40 Hz로 바꿔가며 envelope 주기 확인.
2. envelope을 추출하려면 어떤 신호처리가 필요할지 적어보고 `scipy.signal.hilbert`로 구현.
3. TI가 일반 tACS보다 깊은 영역을 자극할 수 있는 이유를 노트에 3줄로 정리.

#### Step 15 — SimNIBS Installation & GUI Tutorial

**Goal**: SimNIBS로 두피 전극 위치를 잡고 tDCS 시뮬레이션을 실행한다.

**Watch**
- 🇺🇸 [SimNIBS Tutorials playlist](https://www.youtube.com/playlist?list=PLDCjI20ZMvu2G4dGH9CIEztYHTEHg2oam)

**Install**
- [SimNIBS download](https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html)
- [GUI tutorial](https://simnibs.github.io/simnibs/build/html/tutorial/gui.html)

**Dataset**: `m2m_ernie` 예제 (~500MB) — [SimNIBS Datasets](https://simnibs.github.io/simnibs/build/html/dataset.html)

**Challenges**
1. GUI에서 motor cortex(M1) 위 전극(C3) → 반대편(Fp2) tDCS 1mA 시뮬레이션을 끝까지 실행.
2. 결과 .msh 파일을 Gmsh로 열어 전기장 강도를 색깔로 확인.
3. 전류 강도를 0.5 / 1 / 2 mA로 바꿔가며 최대 E-field 변화 기록.

#### Step 16 — Scripting TI Simulation

**Goal**: Python 스크립트로 두 전극쌍에 다른 주파수를 인가하여 TI envelope을 만든다.

**Reference**: [SimNIBS Scripting tutorial](https://simnibs.github.io/simnibs/build/html/tutorial/scripting.html), TI 예제 `simnibs/examples/simulations/TI.py`

**Challenges**
1. TI.py를 복사해서 두 전극쌍 위치를 본인이 정한 좌표로 바꿔 실행.
2. envelope amplitude가 가장 큰 voxel의 위치를 출력.
3. 전극 위치를 1cm 옮기면 envelope hotspot이 얼마나 이동하는지 비교.
4. (응용) 결과를 numpy array로 export해서 matplotlib slice view 만들기.

---

### Part 7. Real-time Closed-loop Systems

#### Step 17 — Closed-loop BCI Concept

**Watch**
- 🇺🇸 [Microsoft Research — Closed-loop BCI](https://www.youtube.com/watch?v=hGJwEU-V-gs)
- 🇺🇸 [gtec — Closed-loop EEG](https://www.youtube.com/watch?v=Dz-9YTFoKwU)
- 🇺🇸 [sentdex — Python OpenBCI](https://www.youtube.com/watch?v=Dgo7F-lpyYE)

**Concepts**: latency, sliding window, online inference.

**Challenges**
1. closed-loop BCI에서 latency가 100ms 넘으면 어떤 문제가 생기는지 영상에서 찾아 정리.
2. 본인이 만들 시스템의 latency budget을 component별로 분배 (acquisition + filtering + inference + stimulation).
3. 본인이 일했던 백엔드 시스템에서 "실시간" 정의를 어떻게 했었는지 BCI 맥락에 비유.

#### Step 18 — Real-time EEG Inference Server

**Goal**: FastAPI + WebSocket으로 EEG 추론 서버를 만든다.

**Code skeleton** — `11_realtime_server.py`

```python
from fastapi import FastAPI, WebSocket
import numpy as np
import torch

app = FastAPI()
window = []                                  # 250 samples = 1초 @ 250Hz

# TODO: Step 12에서 저장한 EEGNet weight를 torch.load로 로드
# model = ...
# model.eval()

@app.websocket("/eeg")
async def eeg_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        sample = await ws.receive_json()      # {"channels": [...]}
        window.append(sample["channels"])
        if len(window) >= 250:
            X = np.array(window[-250:])
            # TODO: X를 (1, n_channels, n_times) tensor로 reshape
            # TODO: model(X) → argmax → 클래스 이름
            prediction = "left_hand"          # placeholder
            await ws.send_json({"class": prediction})
```

```bash
pip install fastapi uvicorn websockets
uvicorn 11_realtime_server:app --reload
```

**Challenges**
1. 클라이언트 스크립트(`client.py`)로 BCI Competition trial을 250ms 간격으로 WebSocket에 전송하고 예측을 받아 정확도 측정.
2. sliding window를 250 → 125 samples(50% overlap)로 바꾸고 latency 차이 측정(`time.time()`).
3. 추론 결과를 SQLite에 로그하고 `/stats` 엔드포인트에서 클래스별 분포 반환.
4. 동시 접속 N명까지 처리 가능한지 `wrk` 또는 `locust`로 부하 테스트.
5. (응용) Redis Pub/Sub으로 추론 결과를 다른 서비스(예: 자극 생성기)에 broadcast — 본인 백엔드 강점 활용.

#### Step 19 — Integration Architecture

```
┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────┐
│ EEG 장비 │ -> │ Decoder     │ -> │ Stimulator   │ -> │ 뇌   │
│ (OpenBCI)│    │ (EEGNet/SNN)│    │ (TI/tDCS)    │    │      │
└──────────┘    └─────────────┘    └──────────────┘    └──────┘
       ▲                                                   │
       └───────────────────────────────────────────────────┘
                       (closed loop)
```

**Challenges**
1. 위 다이어그램에서 각 박스의 latency 예상치를 적어보기 (영상에서 본 수치 기준).
2. 본인 시스템에서 어떤 곳이 bottleneck이 될지 예상하고 측정 계획 적기.
3. failure mode(센서 disconnect, 추론 오류, 자극 over-current) 3가지를 골라 fallback 전략 설계.

---

## Final Project Ideas

1주~1개월 분량 — 한 가지를 골라 끝까지 완성해보기를 권장합니다.

1. **Realtime Motor Imagery Demo**: BCI Competition 데이터를 실시간 스트림처럼 재생 → EEGNet 추론 → 화면에 클래스 표시.
2. **SNN vs CNN Benchmark**: 같은 데이터셋에서 snntorch SNN과 EEGNet의 정확도·파라미터·추론시간 비교 보고서.
3. **TI Stimulation Planner**: SimNIBS Python API로 타깃 영역(좌표) 입력 → 최적 전극 위치 추천 스크립트.
4. **Decoder API as a Service**: FastAPI + Redis + Docker로 EEG 추론 마이크로서비스를 패키징하고 GitHub에 공개.
5. **Custom STDP Visualizer**: brian2로 50개 뉴런 네트워크에 STDP 적용 → 가중치 행렬 변화를 GIF로 저장.

---

## Datasets

| Dataset | Use | Access | Size |
|---------|-----|--------|------|
| MNE Sample (sample_audvis) | EEG/MEG intro | `mne.datasets.sample.data_path()` | ~1GB |
| BCI Competition IV 2a | 4-class motor imagery | [MOABB auto](https://moabb.neurotechx.com/docs/) / [bbci.de](https://bbci.de/competition/iv/download/) | 420MB |
| Allen Brain Observatory | Circuit decoding (optional) | `pip install allensdk` | varies |
| SimNIBS m2m_ernie | TI simulation | [SimNIBS Datasets](https://simnibs.github.io/simnibs/build/html/dataset.html) | ~500MB |

---

## Library Cheat Sheet

| Library | One-liner | Key APIs |
|---------|-----------|----------|
| **MNE-Python** | EEG/MEG 표준 분석 | `mne.io.read_raw_fif`, `raw.filter`, `mne.Epochs`, `mne.preprocessing.ICA` |
| **MOABB** | BCI 데이터셋 통합 | `BNCI2014_001()`, `paradigm.get_data()` |
| **Braindecode** | EEG 딥러닝 | `EEGNetv4`, `MOABBDataset`, `EEGClassifier` |
| **snntorch** | PyTorch SNN | `snn.Leaky(beta)`, `surrogate.fast_sigmoid()` |
| **brian2** | 생물물리 SNN | `NeuronGroup`, `Synapses`, `run()` |
| **scikit-learn** | 클래시컬 ML | `LogisticRegression`, `cross_val_score`, `Pipeline` |
| **SimNIBS** | 비침습 자극 | GUI + Python `simnibs.run_simnibs()` |
| **FastAPI** | 실시간 서버 | `WebSocket`, `uvicorn` |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: No module named mne` | conda 환경 비활성 | `conda activate bci` |
| MOABB 다운로드 멈춤 | 방화벽 | VPN 또는 [수동 다운로드](https://bbci.de/competition/iv/download/) |
| `CUDA out of memory` | GPU 부족 | `.to('cpu')` 또는 batch size 절반 |
| matplotlib 창 안 뜸 | backend 문제 | Jupyter: `%matplotlib inline` / 스크립트: `plt.savefig('out.png')` |
| SimNIBS GUI 안 열림 | 설치 문제 | [공식 설치 가이드](https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html) 재실행 |
| `mne.find_events` 빈 결과 | stim 채널명 오류 | `raw.ch_names` 확인 후 `stim_channel` 인자 수정 |

---

## References

### Korean Lectures

- [EBSi — 자극의 전달](https://www.youtube.com/watch?v=8B6itAOFzg8)
- [북툰 — 뇌과학 15분](https://www.youtube.com/watch?v=_ADYQgL6g_Y)
- [칸아카데미 한국어 — 오일러법](https://ko.khanacademy.org/math/integral-calculus/ic-diff-eq/ic-eulers-method/v/eulers-method)
- [피토스터디 7-2 — 전진/후진 오일러](https://www.youtube.com/watch?v=Kusxa83osyQ)
- [피토스터디 7-3 — 수정 오일러](https://www.youtube.com/watch?v=Q13gjmRt4rc)
- [기계TV — MATLAB 미분방정식](https://www.youtube.com/watch?v=qQANfgZxdeQ)

### SNN / LIF / STDP

- [Gerstner CNS1.3 LIF Model](https://www.youtube.com/watch?v=KGxVwJJC9zs)
- [Gerstner NDC6.5 STDP](https://www.youtube.com/watch?v=z4vZTeTGiVQ)
- [Neuromatch — Biological Neuron Models](https://www.youtube.com/watch?v=rSExvwCVRYg)
- [NPTEL Neuromorphic LIF](https://www.youtube.com/watch?v=gjGTXeJUtCA)
- [Jason Eshraghian Training SNN](https://www.youtube.com/watch?v=zldal7b7sJ4)
- [Dan Goodman Cosyne 2022 SNN](https://www.youtube.com/watch?v=GTXTQ_sOxak)
- [Arnau Blanco IF Python](https://www.youtube.com/watch?v=epcQTW9NpPc)

### EEG / MNE / BCI

- [Berdakh L8 MNE Part 1](https://www.youtube.com/watch?v=IYuAPisoUeI)
- [Armin Abdollahi MNE Tutorial](https://www.youtube.com/watch?v=lEUmgSFQaAY)
- [Pybrain MNE Workshop](https://www.youtube.com/watch?v=t-twhNqgfSY)
- [Talha Anwar — Motor Imagery EEG](https://www.youtube.com/watch?v=Llsfw9RkcIg)
- [Ollie D'Amico — CSP](https://www.youtube.com/watch?v=tInGKSP65OY)
- [Ollie D'Amico — EEGNet](https://www.youtube.com/watch?v=LMQ6Uh-AekY)
- [Basics of EEG BCI](https://www.youtube.com/watch?v=rFFmfNzpfnY)
- [Microsoft Research Closed-loop BCI](https://www.youtube.com/watch?v=hGJwEU-V-gs)
- [sentdex Python OpenBCI](https://www.youtube.com/watch?v=Dgo7F-lpyYE)
- [gtec Closed-loop EEG](https://www.youtube.com/watch?v=Dz-9YTFoKwU)
- [BrainDecode 소개](https://www.youtube.com/watch?v=LwxpDInax44)

### Stimulation

- [Marom Bikson TI](https://www.youtube.com/watch?v=jsyZva93M_8)
- [Neuromatch TI Mechanism](https://www.youtube.com/watch?v=TTl6J7ELsE8)
- [NIMH TMS Basics](https://www.youtube.com/watch?v=io_ganJ0Za0)
- [SimNIBS Tutorials playlist](https://www.youtube.com/playlist?list=PLDCjI20ZMvu2G4dGH9CIEztYHTEHg2oam)

### Spatial Coding

- [Edvard Moser Nobel — Grid Cells](https://www.youtube.com/watch?v=ywPpiPwUstY)
- [O'Keefe Nobel — Place Cells](https://www.youtube.com/watch?v=PrN1OilSAg8)

### Documentation

- [MNE-Python](https://mne.tools/)
- [MOABB](https://moabb.neurotechx.com/docs/)
- [Braindecode](https://braindecode.org/)
- [snntorch](https://snntorch.readthedocs.io/)
- [Brian2](https://brian2.readthedocs.io/)
- [SimNIBS](https://simnibs.github.io/simnibs/)
- [Allen SDK](https://allensdk.readthedocs.io/)

---

## License

MIT
