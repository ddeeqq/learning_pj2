# -*- coding: utf-8 -*-
"""
Step 2 · 2주차 실습 — 전처리와 특징 추출 (6단계)

  단계 1  필터 적용           60Hz 노치(Q=30) + 20–450Hz 4차 Butterworth, filtfilt
                              -> results/week2/filter_check.png, filter_std.txt
  단계 2  분석 구간 잘라내기   2초(2,000샘플) 세그먼트, 시작점은 1주차 관찰로 결정
                              -> results/week2/segment_start.png
  단계 3  슬라이딩 윈도우      200ms 창 · 100ms 이동 -> 세그먼트당 19개
                              -> results/week2/segment_windows.png
  단계 4  시간영역 특징        RMS · MAV · WL · ZC · SSC × 5채널 = 25차원
                              -> results/week2/zc_threshold.csv
  단계 5  데이터셋 만들기      시행 단위로 먼저 8:2 분할 -> 각 집합 안에서만 윈도우·특징
  단계 6  표준화와 저장        StandardScaler(학습셋 fit) -> results/week2/dataset.npz
                              -> results/week2/check_log.txt, dataset_summary.json,
                                 class_feature_means.png

실행: python src/week2_lab.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week1_lab as W1  # noqa: E402  (파일 목록·파일명 파싱·시행 분할은 1주차 코드를 그대로 쓴다)

ROOT = W1.ROOT
RESULTS = ROOT / "results" / "week2"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = W1.FS
CH_NAMES = W1.CH_NAMES
CLASSES = W1.CLASSES
SEED = 42

# 단계 2: 슬라이드 기본값(1.5초)은 '실습 가정'. 1주차에서 근활성이 1.0~2.5초에 있음을 확인했고,
#         아래 scan_segment_start() 로 버스트 에너지를 가장 많이 담는 시작점을 골랐다.
SLIDE_START = 1500
START, LEN = 700, 2000          # 0.7초 지점부터 2초 (0.7 ~ 2.7초)
WIN, HOP = 200, 100            # 200ms 창, 100ms 이동 (50% 오버랩)
ZC_TH = 0.01                   # ZC 임계값 (슬라이드 조건)

FEATURES = ["RMS", "MAV", "WL", "ZC", "SSC"]
FEATURE_NAMES = [f"{f}_{c}" for f in FEATURES for c in CH_NAMES]   # td_features 출력 순서

LOG = []


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------------
# 단계 1. 필터
# ----------------------------------------------------------------------
def preprocess(x, fs=FS):
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def load_filtered_trials(path):
    """파일 하나(30회 연속 기록)를 통째로 필터링한 뒤 mark 기준으로 시행을 자른다.
    시행(5초)마다 따로 거르면 양 끝에 filtfilt 가장자리 효과가 생기므로 연속 신호에서 먼저 거른다."""
    x, marks = W1.load_file(path)
    xf = preprocess(x)
    return W1.split_trials(x, marks), W1.split_trials(xf, marks)


def step1_filter(files):
    raw, filt = {}, {}
    for f in files:
        r, q = load_filtered_trials(f)
        raw[W1.parse_name(f)], filt[W1.parse_name(f)] = r, q
    x0, _ = W1.load_file(files[0])
    xf0 = preprocess(x0)
    log("[단계 1] 필터 적용 —", Path(files[0]).name)
    log("전:", x0.std(0).round(4))
    log("후:", xf0.std(0).round(4))

    # 필터 검증: 1,920개 시행 전체의 평균 PSD (채널별)
    keys = sorted(raw)
    P0 = np.zeros((5, 513)); P1 = np.zeros((5, 513)); n = 0
    for k in keys:
        fr, p0 = welch(raw[k], fs=FS, nperseg=1024, axis=1)     # (30, 513, 5)
        _, p1 = welch(filt[k], fs=FS, nperseg=1024, axis=1)
        P0 += p0.sum(0).T; P1 += p1.sum(0).T; n += len(p0)
    P0 /= n; P1 /= n

    fig, axes = plt.subplots(5, 1, figsize=(9, 9.5), sharex=True)
    for ch, ax in enumerate(axes):
        ax.semilogy(fr, P0[ch], lw=1.0, color="#2a78d6", label="before (저장소 원본 'bandpass' 열)")
        ax.semilogy(fr, P1[ch], lw=1.0, color="#eb6834", alpha=0.9,
                    label="after (notch 60Hz Q=30 + BPF 20–450Hz 4차, filtfilt)")
        for v, st in ((20, ":"), (60, "--"), (450, ":")):
            ax.axvline(v, color="#6B7280", ls=st, lw=0.8)
        ax.set_ylabel(f"ch{ch + 1} {CH_NAMES[ch]}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="upper right", frameon=False)
    axes[0].set_title("필터 전후 평균 PSD — 64개 파일 · 1,920개 시행 (Welch, nperseg=1024)", fontsize=10)
    axes[-1].set_xlabel("Hz", fontsize=8)
    axes[-1].set_xlim(0, 500)
    fig.tight_layout()
    fig.savefig(RESULTS / "filter_check.png", dpi=120)
    plt.close(fig)

    def band(P, lo, hi):
        m = (fr >= lo) & (fr < hi)
        return P[:, m].mean(1)
    i60 = int(np.argmin(np.abs(fr - 60)))
    stats = {
        "std_before": x0.std(0).tolist(), "std_after": xf0.std(0).tolist(),
        "low_ratio": (band(P0, 1, 15) / band(P1, 1, 15)).tolist(),        # 20Hz 아래 감쇠
        "high_ratio": (band(P0, 460, 500) / band(P1, 460, 500)).tolist(),  # 450Hz 위 감쇠
        "p60_ratio": (P0[:, i60] / P1[:, i60]).tolist(),
        "pass_ratio": (band(P0, 30, 400) / band(P1, 30, 400)).tolist(),    # 통과 대역 유지
        "p60_local_before": (P0[:, i60] / band(P0, 50, 70)).tolist(),      # 60Hz 봉우리가 있었는가
    }
    (RESULTS / "filter_std.txt").write_text(
        "\n".join(LOG[-3:]) + "\n", encoding="utf-8")
    return raw, filt, stats


# ----------------------------------------------------------------------
# 단계 2. 분석 구간
# ----------------------------------------------------------------------
def cut(x, start=START, n=LEN):
    return x[start:start + n]


def scan_segment_start(filt):
    """2초 창의 시작점별로 '시행의 버스트 에너지(휴식 기준선 위)'를 몇 % 담는지 계산."""
    T = np.concatenate([filt[k] for k in sorted(filt)])          # (1920, 5000, 5)
    E = (T ** 2).sum(2)
    base = np.median(E[:, 4000:5000], axis=1, keepdims=True)    # 마지막 1초 휴식 = 기준선
    Ex = np.clip(E - base, 0, None)
    tot = Ex.sum(1)
    starts = np.arange(0, 3001, 100)
    cap = np.array([Ex[:, s:s + LEN].sum(1) / tot for s in starts])  # (n_start, 1920)
    return starts, cap


def step2_segment(filt):
    starts, cap = scan_segment_start(filt)
    mean = cap.mean(1); q10 = np.quantile(cap, 0.1, axis=1)
    i_best = int(np.argmax(mean))
    i_ch, i_sl = list(starts).index(START), list(starts).index(SLIDE_START)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(starts / 1000, mean * 100, color="#2a78d6", lw=2, label="평균")
    ax.plot(starts / 1000, q10 * 100, color="#2a78d6", lw=1.2, ls="--", label="하위 10% 시행")
    for i, txt, col in ((i_ch, f"채택 {START / 1000:.1f}s", "#eb6834"),
                        (i_sl, f"슬라이드 {SLIDE_START / 1000:.1f}s", "#6B7280")):
        ax.axvline(starts[i] / 1000, color=col, lw=1.2)
        ax.text(starts[i] / 1000 + 0.05, 12 if col == "#6B7280" else 58,
                f"{txt}\n평균 {mean[i] * 100:.1f}% · 하위10% {q10[i] * 100:.1f}%",
                fontsize=8, color=col, va="center")
    ax.set_xlabel("2초 세그먼트 시작점 (s, mark 기준)", fontsize=9)
    ax.set_ylabel("버스트 에너지 포함 비율 (%)", fontsize=9)
    ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.set_title("시작점별로 2초 구간이 담는 근활성 에너지 — 1,920개 시행", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "segment_start.png", dpi=120)
    plt.close(fig)

    xf = filt[("a", "30", "fist")][0]
    seg = cut(xf)
    log("\n[단계 2] 분석 구간 — START, LEN =", START, LEN)
    log("세그먼트 모양:", seg.shape)
    return {"starts": starts.tolist(), "mean": mean.tolist(), "q10": q10.tolist(),
            "best_start": int(starts[i_best]), "best_mean": float(mean[i_best]),
            "chosen_mean": float(mean[i_ch]), "chosen_q10": float(q10[i_ch]),
            "slide_mean": float(mean[i_sl]), "slide_q10": float(q10[i_sl])}


# ----------------------------------------------------------------------
# 단계 3. 슬라이딩 윈도우
# ----------------------------------------------------------------------
def make_windows(x, win=WIN, hop=HOP):
    n = (len(x) - win) // hop + 1
    return np.stack([x[i * hop: i * hop + win] for i in range(n)])


def step3_windows(filt):
    xf = filt[("a", "30", "fist")][0]
    seg = cut(xf)
    w = make_windows(seg)
    log("\n[단계 3] 슬라이딩 윈도우 — WIN, HOP =", WIN, HOP)
    log("윈도우 배열:", w.shape)
    log("19 = (2000 - 200) / 100 + 1 =", (LEN - WIN) // HOP + 1)

    # 그림: 5초 시행 위에 슬라이드 구간 · 채택 구간 · 19개 윈도우
    t = np.arange(len(xf)) / FS
    fig = plt.figure(figsize=(10, 6.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.75], hspace=0.35)
    for k, ch in enumerate([1, 2]):
        ax = fig.add_subplot(gs[k])
        ax.plot(t, xf[:, ch], lw=0.5, color="#1F2937")
        ax.axvspan(START / FS, (START + LEN) / FS, color="#eb6834", alpha=0.13, lw=0,
                   label=f"채택 세그먼트 {START / FS:.1f}~{(START + LEN) / FS:.1f}s")
        ax.axvline(SLIDE_START / FS, color="#6B7280", ls="--", lw=0.9,
                   label=f"슬라이드 세그먼트 {SLIDE_START / FS:.1f}~{(SLIDE_START + LEN) / FS:.1f}s")
        ax.axvline((SLIDE_START + LEN) / FS, color="#6B7280", ls="--", lw=0.9)
        ax.set_ylabel(f"ch{ch + 1} {CH_NAMES[ch]}", fontsize=8)
        ax.set_xlim(0, 5); ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
        if k == 0:
            ax.legend(fontsize=7.5, loc="upper right", frameon=False)
            ax.set_title("피험자 a · 30° Fist · 1번째 시행 (필터 후) — 2초 세그먼트와 19개 윈도우",
                         fontsize=10)
    ax = fig.add_subplot(gs[2])
    for i in range(w.shape[0]):
        s = (START + i * HOP) / FS
        ax.add_patch(plt.Rectangle((s, 18 - i), WIN / FS, 0.8,
                                   facecolor="#2a78d6" if i % 2 == 0 else "#86b6ef", edgecolor="none"))
    ax.set_xlim(0, 5); ax.set_ylim(-0.5, 19.5)
    ax.set_yticks([18.4, 0.4], ["창 1", "창 19"], fontsize=7.5)
    ax.set_xlabel("time (s)", fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title("200ms 창 · 100ms 이동 → 이웃 창과 50% 겹침", fontsize=9)
    fig.savefig(RESULTS / "segment_windows.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return w


# ----------------------------------------------------------------------
# 단계 4. 시간영역 특징 (numpy만 사용)
# ----------------------------------------------------------------------
def td_features(w, th=ZC_TH):
    """w: (200, 5) -> (25,)
    순서: RMS×5, MAV×5, WL×5, ZC×5, SSC×5  (채널: ECR, ED, FDS, Biceps, Triceps)"""
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d = np.diff(w, axis=0)
    wl = np.sum(np.abs(d), axis=0)
    zc = np.sum(((w[:-1] * w[1:]) < 0) &
                (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1] * d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])


def step4_features(w, filt):
    f = td_features(w[0])
    log("\n[단계 4] 시간영역 특징")
    log("특징 벡터:", f.shape)
    log("NaN 개수:", int(np.isnan(f).sum()))

    # ZC 임계값 관찰: 0 / 0.001 / 0.005 / 0.01
    W = np.concatenate([make_windows(cut(tr)) for k in sorted(filt) for tr in filt[k]])  # (36480, 200, 5)
    d = np.abs(np.diff(W, axis=1))
    rows = []
    for th in (0.0, 0.001, 0.005, 0.01):
        zc = (((W[:, :-1] * W[:, 1:]) < 0) & (d >= th)).sum(1)    # (N, 5)
        rows.append({"th": th, **{f"ZC_{c}": float(zc[:, i].mean()) for i, c in enumerate(CH_NAMES)},
                     "zero_window_ratio": float((zc == 0).mean())})
    zdf = pd.DataFrame(rows)
    zdf.to_csv(RESULTS / "zc_threshold.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    q = np.quantile(d, [0.1, 0.5, 0.9]).tolist()
    return f, zdf, q


# ----------------------------------------------------------------------
# 단계 5. 데이터셋 — 시행 단위로 먼저 분할
# ----------------------------------------------------------------------
def trial_table(files):
    """시행 1,920개의 목록: (파일, 시행 번호, 피험자, 라벨)."""
    rows = []
    for p in files:
        sub, ang, ges = W1.parse_name(p)
        for j in range(30):
            rows.append({"file": p, "trial": j, "subject": sub, "label": f"{ang}_{ges}",
                         "group": f"{sub}_{ang}_{ges}"})
    return pd.DataFrame(rows)


def split_trials_8_2(trials):
    # ===== [분할] 시행 단위로 먼저 8:2 분할 — 윈도우·특징은 이 다음에 만든다 =====
    tr_idx, te_idx = train_test_split(
        np.arange(len(trials)), test_size=0.2,
        stratify=trials["group"],          # 피험자×클래스(64그룹)마다 30회 중 24:6
        random_state=SEED)
    return trials.iloc[tr_idx].reset_index(drop=True), trials.iloc[te_idx].reset_index(drop=True)


def build(trial_rows, filt):
    """주어진 시행들 안에서만 세그먼트 -> 윈도우 -> 특징을 만든다."""
    X, y, tid, sub = [], [], [], []
    for r in trial_rows.itertuples():
        x = filt[W1.parse_name(r.file)][r.trial]          # (5000, 5) 필터 후 시행
        for wnd in make_windows(cut(x)):
            X.append(td_features(wnd)); y.append(r.label)
            tid.append(f"{r.group}#{r.trial:02d}"); sub.append(r.subject)
    return np.stack(X), np.array(y), np.array(tid), np.array(sub)


def step5_dataset(files, filt):
    trials = trial_table(files)
    tr_t, te_t = split_trials_8_2(trials)
    Xtr, ytr, ttr, str_ = build(tr_t, filt)
    Xte, yte, tte, ste = build(te_t, filt)
    overlap = len(set(ttr) & set(tte))
    log("\n[단계 5] 데이터셋 — 시행 단위 분할 후 특징 생성")
    log("학습 시행 / 테스트 시행:", len(tr_t), "/", len(te_t))
    log("Xtr, Xte:", Xtr.shape, Xte.shape)
    log("학습 ∩ 테스트 시행 수:", overlap)
    log("학습/테스트 샘플 비율:", round(len(Xtr) / len(Xte), 2))
    return (Xtr, ytr, ttr, str_), (Xte, yte, tte, ste), tr_t, te_t, overlap


# ----------------------------------------------------------------------
# 단계 6. 표준화와 저장
# ----------------------------------------------------------------------
def step6_scale_save(train, test):
    Xtr, ytr, ttr, str_ = train
    Xte, yte, tte, ste = test
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)     # 학습셋에서 fit
    Xte_s = sc.transform(Xte)         # 테스트엔 적용만
    log("\n[단계 6] 표준화 (StandardScaler, 학습셋 fit)")
    log("학습셋 평균 (|최댓값|):", f"{np.abs(Xtr_s.mean(0)).max():.2e}")
    log("학습셋 표준편차 (최소~최대):", f"{Xtr_s.std(0).min():.6f} ~ {Xtr_s.std(0).max():.6f}")
    log("테스트셋 평균 (|최댓값|):", f"{np.abs(Xte_s.mean(0)).max():.4f}")
    log("상수 특징 수:", int((Xtr.std(0) == 0).sum()))

    out = RESULTS / "dataset.npz"
    np.savez_compressed(
        out, Xtr=Xtr_s, ytr=ytr, Xte=Xte_s, yte=yte,
        trial_tr=ttr, trial_te=tte, subject_tr=str_, subject_te=ste,
        feature_names=np.array(FEATURE_NAMES), classes=np.array(CLASSES),
        scaler_mean=sc.mean_, scaler_scale=sc.scale_)
    log("저장 완료:", out.relative_to(ROOT), f"({out.stat().st_size / 1e6:.1f} MB)")
    return Xtr_s, Xte_s, sc


def class_feature_heatmap(Xtr_s, ytr):
    M = np.stack([Xtr_s[ytr == c].mean(0) for c in CLASSES])   # (8, 25)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    lim = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(8), [f"{c.split('_')[0]}° {W1.GESTURE_EN[c.split('_')[1]]}" for c in CLASSES],
                  fontsize=8)
    ax.set_xticks(range(25), [c for _ in FEATURES for c in CH_NAMES], rotation=90, fontsize=7)
    for i, f in enumerate(FEATURES):
        ax.text(i * 5 + 2, -0.75, f, ha="center", fontsize=9, fontweight="bold")
        if i:
            ax.axvline(i * 5 - 0.5, color="white", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.025, label="표준화 값 평균")
    ax.set_title("클래스별 표준화 특징 평균 (학습셋) — 25차원", fontsize=10, pad=24)
    fig.tight_layout()
    fig.savefig(RESULTS / "class_feature_means.png", dpi=120)
    plt.close(fig)
    return M


def main():
    W1.set_korean_font()
    files = W1.step2_list_files()
    LOG.clear()
    raw, filt, fstats = step1_filter(files)
    del raw
    seg_info = step2_segment(filt)
    w = step3_windows(filt)
    f, zdf, dq = step4_features(w, filt)
    train, test, tr_t, te_t, overlap = step5_dataset(files, filt)
    Xtr_s, Xte_s, sc = step6_scale_save(train, test)
    M = class_feature_heatmap(Xtr_s, train[1])

    ytr, yte = train[1], test[1]
    summary = {
        "start": START, "len": LEN, "win": WIN, "hop": HOP, "zc_th": ZC_TH,
        "n_windows_per_segment": (LEN - WIN) // HOP + 1,
        "filter": fstats, "segment": seg_info,
        "zc_table": zdf.to_dict(orient="records"), "abs_diff_quantiles": dq,
        "n_trials_train": len(tr_t), "n_trials_test": len(te_t),
        "Xtr_shape": list(Xtr_s.shape), "Xte_shape": list(Xte_s.shape),
        "trial_overlap": overlap,
        "class_counts": {c: {"train_trials": int((tr_t.label == c).sum()),
                             "test_trials": int((te_t.label == c).sum()),
                             "train_windows": int((ytr == c).sum()),
                             "test_windows": int((yte == c).sum())} for c in CLASSES},
        "subject_counts": {s: {"train_trials": int((tr_t.subject == s).sum()),
                               "test_trials": int((te_t.subject == s).sum())} for s in W1.SUBJECTS},
        "train_mean_absmax": float(np.abs(Xtr_s.mean(0)).max()),
        "train_std_min": float(Xtr_s.std(0).min()), "train_std_max": float(Xtr_s.std(0).max()),
        "test_mean_absmax": float(np.abs(Xte_s.mean(0)).max()),
        "test_std_min": float(Xte_s.std(0).min()), "test_std_max": float(Xte_s.std(0).max()),
        "feature_names": FEATURE_NAMES,
        "example_feature": f.tolist(),
        "class_feature_means": M.tolist(),
        "npz_mb": (RESULTS / "dataset.npz").stat().st_size / 1e6,
    }
    (RESULTS / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    (RESULTS / "check_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    print("\n완료: results/week2/ 폴더 확인")


if __name__ == "__main__":
    main()
