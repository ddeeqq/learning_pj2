# data/

원 데이터(64개 CSV, 약 930MB)는 용량 때문에 이 저장소에 올리지 않았습니다. 아래 명령으로 받아 이 폴더에 두면 됩니다.

```bash
git clone https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset data/sEMG-Hand-Gesture-Elbow-Angle-Dataset
```

받은 뒤 구조:

```text
data/
└── sEMG-Hand-Gesture-Elbow-Angle-Dataset/
    ├── README.md
    └── data/
        ├── a_30 extor.csv
        ├── a_30 fist.csv
        ├── ...
        └── h_60 pro.csv      # 8명(a~h) × 2각도 × 4동작 = 64파일
```

- 동일한 파일이 https://github.com/Juhyun1123/sEMG_Hand_Gesture_Elbow_Angle 에도 있습니다(줄바꿈 문자만 다름).
- 논문: *Simultaneous Classification of Hand Gestures and Elbow Angles Using sEMG Signals: A Comparison of Machine Learning Performances.* Biomedical Engineering Letters (2026). DOI 10.1007/s13534-026-00607-7
