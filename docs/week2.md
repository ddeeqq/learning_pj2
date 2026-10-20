[← README로 돌아가기](../README.md)

# 2주차 — 전처리와 특징 추출

원신호를 필터링하고 2초 구간을 200ms 창 19개로 잘라, 창마다 시간영역 특징 5종 × 5채널 = 25차원 벡터를 만들었습니다.
시행 단위로 먼저 학습/테스트를 나눈 뒤 특징을 만들고, 학습셋 통계로 표준화해 `dataset.npz`로 저장했습니다.

## 2-1. 실습 단계와 산출물

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1. 필터 적용 | 60Hz 노치(Q=30) + 20–450Hz 4차 Butterworth, `filtfilt` | `results/week2/filter_check.png`, `filter_std.txt` |
| 2. 분석 구간 | 2초(2,000샘플) 세그먼트, 시작점 0.7초 | `results/week2/segment_start.png` |
| 3. 슬라이딩 윈도우 | 200ms 창 · 100ms 이동 → (19, 200, 5) | `results/week2/segment_windows.png` |
| 4. 시간영역 특징 | RMS · MAV · WL · ZC · SSC 직접 구현 → (25,) | `results/week2/zc_threshold.csv` |
| 5. 데이터셋 | 시행 단위 8:2 분할 후 각 집합 안에서만 특징 생성 | `results/week2/dataset_summary.json` |
| 6. 표준화와 저장 | `StandardScaler` 학습셋 fit → 테스트셋 transform | `results/week2/dataset.npz`, `check_log.txt` |

## 2-2. 데이터셋 요약

| 항목 | 값 |
|---|---|
| 학습 / 테스트 시행 | 1,536 / 384 — 피험자×클래스(64그룹)마다 30회 중 24:6 |
| 학습 / 테스트 샘플(창) | (29184, 25) / (7296, 25) — 학습이 테스트의 4.0배 |
| 클래스별 | 모든 클래스 학습 3,648 · 테스트 912 창 |
| 학습 ∩ 테스트 시행 | 0 (겹치는 시행 없음) |
| 표준화 확인 | 학습셋 평균 ≈ 0(1e-14), 표준편차 1.000000 · 테스트셋 평균 절댓값 최대 0.031 |
| 특징 순서 | RMS×5 → MAV×5 → WL×5 → ZC×5 → SSC×5, 각 블록의 채널 순서 ECR · ED · FDS · Biceps · Triceps |

```python
d = np.load('results/week2/dataset.npz')
Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']   # ytr 예: '30_fist'
# 추가 키: trial_tr/te(시행 ID), subject_tr/te, feature_names, classes, scaler_mean, scaler_scale
```

## 2-3. 주요 관찰

![2초 세그먼트와 19개 윈도우](../results/week2/segment_windows.png)

- **세그먼트 시작점을 1.5초에서 0.7초로 옮겼습니다.** 슬라이드도 1.5초는 실습 가정이라 데이터에 맞게 조정하라고 합니다. 시작점별로 2초 구간이 담는 버스트 에너지를 1,920개 시행에서 계산하니 0.7초가 평균 91.4%로 최대였고, 1.5초는 68.3%(하위 10% 시행은 26.3%)에 그쳤습니다.

![시작점별 에너지 포함 비율](../results/week2/segment_start.png)

- **필터는 이미 걸린 상태였습니다.** 저장소의 `bandpass` 열은 노치 + 대역통과가 적용된 신호라 60Hz 봉우리가 처음부터 없습니다. 다시 걸어도 30–400Hz 전력은 1.04~1.09배로 유지되고, 대역 밖(20Hz 아래, 450Hz 위)만 100배 이상 추가로 줄었습니다.
- **ZC 임계값 0.01은 이 신호에 비해 큽니다.** th=0이면 창당 22~29회지만 0.01이면 3~13회로 줄고, 채널-창의 28%가 ZC=0이 됩니다. 슬라이드 조건을 그대로 쓰고 기록만 해 두었습니다.
- **RMS·MAV·WL은 거의 같은 정보입니다.** 클래스별 평균 무늬가 세 블록에서 거의 똑같습니다.

![클래스별 표준화 특징 평균](../results/week2/class_feature_means.png)

- 이 분할은 같은 피험자의 다른 시행이 양쪽에 들어가는 피험자 내 평가입니다. 처음 보는 사람에 대한 성능은 피험자 단위 분할로 따로 확인해야 합니다.
