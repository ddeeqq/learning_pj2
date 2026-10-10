# -*- coding: utf-8 -*-
"""
Step 2 · 1주차 실습 — 문제 정의와 데이터 이해 (5단계)

  단계 1  환경 점검          -> results/env_check.txt
  단계 2  데이터 확보        -> 파일 목록 확인 (64개 CSV)
  단계 3  데이터 구조 파악    -> results/data_summary.json, results/class_counts.csv
  단계 4  채널별 신호 그리기  -> results/ch5_example.png, results/angle_side_by_side.png,
                               results/angle_envelope_grid.png, results/class_channel_rms.png
  단계 5  클래스 설계안 근거  -> results/angle_rms_table.csv (30° vs 60° 수치 비교)

이번 주는 모델을 만들지 않는다. 데이터를 세고, 눈으로 보는 것까지만 한다.

데이터: sEMG Hand Gesture and Elbow Angle Dataset (논문 원 데이터, 8명 × 8클래스 × 30회)
        data/sEMG-Hand-Gesture-Elbow-Angle-Dataset/data/*.csv

실행: python src/week1_lab.py
"""
import json
import os
import platform
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sEMG-Hand-Gesture-Elbow-Angle-Dataset" / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FS = 1000                 # Hz
TRIAL_LEN = 5 * FS        # 5초 시행 = 휴식 2 + 동작 1 + 휴식 2
N_CH = 5
CH_NAMES = ["ECR", "ED", "FDS", "Biceps", "Triceps"]
CH_KO = ["요측수근신근", "총지신근", "천지굴근", "상완이두근", "상완삼두근"]
GESTURES = ["fist", "open", "pro", "extor"]           # 파일명 표기
GESTURE_KO = {"fist": "주먹 쥐기", "open": "손 펴기", "pro": "회내", "extor": "회외"}
GESTURE_EN = {"fist": "Fist", "open": "Open", "pro": "Pronation", "extor": "Supination"}
ANGLES = ["30", "60"]
SUBJECTS = list("abcdefgh")
CLASSES = [f"{a}_{g}" for g in GESTURES for a in ANGLES]  # 4동작 × 2각도 = 8클래스

# 단계 4에서 관찰한 실제 활동 구간 / 휴식 구간 (마커 기준 초)
ACT_WIN = (1.0, 2.5)
REST_WIN = (4.0, 5.0)

COLOR_30 = "#2a78d6"   # 30°
COLOR_60 = "#eb6834"   # 60°


def set_korean_font():
    """Windows(Malgun Gothic) / Linux(Noto Sans CJK KR) 어느 쪽에서도 한글이 깨지지 않게."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "AppleGothic"]:
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# 단계 1. 환경 점검
# ----------------------------------------------------------------------
def step1_env_check():
    import numpy, scipy, sklearn, xgboost  # noqa: E401  (슬라이드 9쪽 확인 코드와 동일)
    lines = [
        f"$ python - <<'EOF'  (import numpy, scipy, sklearn, xgboost)",
        f"numpy    {numpy.__version__}",
        f"scipy    {scipy.__version__}",
        f"sklearn  {sklearn.__version__}",
        f"xgboost  {xgboost.__version__}",
        "",
        f"# python {platform.python_version()} · {platform.system()} {platform.machine()}",
    ]
    text = "\n".join(lines)
    (RESULTS / "env_check.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return text


# ----------------------------------------------------------------------
# 단계 2. 데이터 확보
# ----------------------------------------------------------------------
def step2_list_files():
    files = sorted(glob(str(DATA_DIR / "*.csv")))
    if not files:
        sys.exit(f"데이터가 없습니다: {DATA_DIR}\n"
                 "README의 '데이터 받기'를 먼저 실행하세요.")
    print("파일 수:", len(files))
    print("예시:", os.path.relpath(files[0], ROOT))
    return files


# ----------------------------------------------------------------------
# 단계 3. 데이터 구조 파악
# ----------------------------------------------------------------------
NAME_RE = re.compile(r"^([a-h])_(30|60) (fist|open|pro|extor)$")


def parse_name(path):
    """'a_30 fist.csv' -> ('a', '30', 'fist').
    슬라이드의 label_of 는 '30_fist_trial01.csv' 형식을 가정하지만, 실제 파일은
    '[피험자]_[각도] [동작].csv' 이고 파일 하나에 30회 시행이 이어 붙어 있어 규칙을 바꿨다."""
    m = NAME_RE.match(Path(path).stem)
    if m is None:
        raise ValueError(f"파일명 규칙이 다름: {path}")
    return m.groups()


def label_of(path):
    sub, ang, ges = parse_name(path)
    return f"{ang}_{ges}"          # 예: '30_fist' (슬라이드와 같은 라벨 모양)


def load_file(path):
    """CSV 하나 -> (연속 신호 (N, 5), mark 위치 배열)."""
    d = pd.read_csv(path)
    x = d.iloc[:, :N_CH].to_numpy(dtype=np.float64)
    marks = np.flatnonzero(d["mark"].to_numpy() != 0)
    return x, marks


def split_trials(x, marks):
    """mark(시행 시작 표시)마다 5,000샘플씩 잘라 (30, 5000, 5) 배열을 만든다."""
    trials = [x[s:s + TRIAL_LEN] for s in marks if s + TRIAL_LEN <= len(x)]
    return np.stack(trials)


def step3_structure(files):
    labels = [label_of(f) for f in files]
    file_counts = Counter(labels)
    print("클래스별 파일 수:", dict(file_counts))

    per_file = []
    trial_counts = Counter()
    for f in files:
        x, marks = load_file(f)
        tr = split_trials(x, marks)
        sub, ang, ges = parse_name(f)
        gaps = np.diff(marks)
        per_file.append({
            "file": Path(f).name, "subject": sub, "angle": ang, "gesture": ges,
            "label": f"{ang}_{ges}", "n_samples": int(len(x)), "n_marks": int(len(marks)),
            "n_trials": int(len(tr)), "trial_shape": list(tr.shape[1:]),
            "mark_gap_min": int(gaps.min()), "mark_gap_max": int(gaps.max()),
            "duration_s": round(len(x) / FS, 1),
        })
        trial_counts[f"{ang}_{ges}"] += len(tr)

    df = pd.DataFrame(per_file)
    x0, m0 = load_file(files[0])
    t0 = split_trials(x0, m0)
    rows = []
    for c in CLASSES:
        sub = df[df.label == c]
        rows.append({"class": c, "files": len(sub), "subjects": sub.subject.nunique(),
                     "trials": int(sub.n_trials.sum())})
    pd.DataFrame(rows).to_csv(RESULTS / "class_counts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_files": len(files),
        "n_subjects": int(df.subject.nunique()),
        "n_classes": len(file_counts),
        "file_counts": dict(file_counts),
        "trial_counts": {c: trial_counts[c] for c in CLASSES},
        "total_trials": int(sum(trial_counts.values())),
        "raw_file_shape": [int(x0.shape[0]), int(x0.shape[1])],
        "raw_columns": list(pd.read_csv(files[0], nrows=0).columns),
        "trial_array_shape": list(t0.shape),
        "n_channels": int(t0.shape[2]),
        "samples_min": int(df.n_samples.min()), "samples_max": int(df.n_samples.max()),
        "mark_gap_min": int(df.mark_gap_min.min()), "mark_gap_max": int(df.mark_gap_max.max()),
        "all_files_30_trials": bool((df.n_trials == 30).all()),
        "example_file": Path(files[0]).name,
        "per_file": per_file,
    }
    (RESULTS / "data_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("배열 모양(파일 원본):", x0.shape, " -> 시행 배열:", t0.shape)
    print("채널 수:", t0.shape[2])
    print("클래스별 시행 수:", summary["trial_counts"])
    return summary


# ----------------------------------------------------------------------
# 단계 4. 채널별 신호 그리기
# ----------------------------------------------------------------------
def load_all_trials(files):
    """{(subject, angle, gesture): (30, 5000, 5)}"""
    out = {}
    for f in files:
        x, marks = load_file(f)
        out[parse_name(f)] = split_trials(x, marks)
    return out


def moving_rms(tr, win=100, hop=20):
    """(…, T, C) -> (…, n, C) 이동 RMS 포락선 (100ms 창, 20ms 이동)."""
    n = (tr.shape[-2] - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    seg = tr[..., idx, :]                      # (…, n, win, C)
    return np.sqrt((seg ** 2).mean(axis=-2)), (idx[:, 0] + win / 2) / FS


def window_rms(tr, t0, t1):
    a, b = int(t0 * FS), int(t1 * FS)
    return np.sqrt((tr[..., a:b, :] ** 2).mean(axis=-2))


def plot_5ch(ax_col, x, title=None, ylim=None, show_slide_lines=True):
    t = np.arange(len(x)) / FS
    for ch in range(N_CH):
        ax = ax_col[ch]
        ax.plot(t, x[:, ch], lw=0.5, color="#1F2937")
        ax.set_ylabel(CH_NAMES[ch], fontsize=8)
        ax.axvspan(*ACT_WIN, color="#2a78d6", alpha=0.10, lw=0)
        if show_slide_lines:
            ax.axvline(2.0, color="r", ls="--", lw=0.8)
            ax.axvline(3.0, color="r", ls="--", lw=0.8)
        if ylim is not None:
            ax.set_ylim(-ylim[ch], ylim[ch])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    ax_col[0].set_title(title or "", fontsize=10)
    ax_col[-1].set_xlabel("time (s)", fontsize=8)


def step4_plots(allt, example=("a", "30", "fist"), trial=0):
    # (1) 슬라이드 12쪽 그림 그대로 + 실제 활동 구간 음영
    x = allt[example][trial]
    fig, ax = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
    plot_5ch(ax, x, title=f"피험자 {example[0]} · {example[1]}° · {GESTURE_EN[example[2]]} · "
                          f"{trial + 1}번째 시행 (5초)  —  빨간 점선: 슬라이드 기준 2~3초, "
                          f"파란 음영: 실제 활동 구간 {ACT_WIN[0]}~{ACT_WIN[1]}초")
    plt.tight_layout()
    fig.savefig(RESULTS / "ch5_example.png", dpi=120)
    plt.close(fig)

    # (2) 같은 동작의 두 각도를 나란히 (같은 피험자, 같은 시행 번호, 같은 y축)
    sub, _, ges = example
    a30, a60 = allt[(sub, "30", ges)][trial], allt[(sub, "60", ges)][trial]
    ylim = np.maximum(np.abs(a30).max(0), np.abs(a60).max(0)) * 1.05
    fig, ax = plt.subplots(5, 2, figsize=(11, 7), sharex=True)
    plot_5ch(ax[:, 0], a30, title=f"피험자 {sub} · {GESTURE_EN[ges]} · 30°", ylim=ylim,
             show_slide_lines=False)
    plot_5ch(ax[:, 1], a60, title=f"피험자 {sub} · {GESTURE_EN[ges]} · 60°", ylim=ylim,
             show_slide_lines=False)
    plt.tight_layout()
    fig.savefig(RESULTS / "angle_side_by_side.png", dpi=120)
    plt.close(fig)

    # (3) 8명 × 30회 평균 포락선: 동작 4 (열) × 채널 5 (행), 30° vs 60°
    fig, ax = plt.subplots(5, 4, figsize=(11, 9), sharex=True)
    env_profile = None
    for j, g in enumerate(GESTURES):
        for ang, col in (("30", COLOR_30), ("60", COLOR_60)):
            e = np.stack([moving_rms(allt[(s, ang, g)])[0] for s in SUBJECTS])  # (8,30,n,5)
            tt = moving_rms(allt[(SUBJECTS[0], ang, g)][:1])[1]
            m = e.mean(axis=(0, 1)) * 1000
            for ch in range(N_CH):
                ax[ch, j].plot(tt, m[:, ch], color=col, lw=1.6, label=f"{ang}°")
        for ch in range(N_CH):
            ax[ch, j].axvspan(*ACT_WIN, color="#9CA3AF", alpha=0.12, lw=0)
            ax[ch, j].grid(alpha=0.2)
            ax[ch, j].tick_params(labelsize=7)
            if j == 0:
                ax[ch, j].set_ylabel(f"{CH_NAMES[ch]}\nRMS (×1000)", fontsize=8)
        ax[0, j].set_title(f"{GESTURE_EN[g]} ({GESTURE_KO[g]})", fontsize=10)
        ax[-1, j].set_xlabel("time (s)", fontsize=8)
    ax[0, 0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle("30° vs 60° — 8명 × 30회 평균 이동 RMS 포락선 (100ms 창), 회색 = 활동 구간",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS / "angle_envelope_grid.png", dpi=120)
    plt.close(fig)

    # (4) 전체 평균 포락선 시간 프로파일 (활동 구간을 정한 근거)
    tot = []
    peak_t = []
    for k, tr in allt.items():
        e, tt = moving_rms(tr)
        s = e.sum(-1)                         # (30, n)
        tot.append(s.mean(0) / s.mean())
        peak_t.extend(tt[s.argmax(1)].tolist())
    env_profile = {"t": tt.tolist(), "mean_norm_env": np.mean(tot, 0).tolist(),
                   "peak_time_median": float(np.median(peak_t)),
                   "peak_time_q10": float(np.quantile(peak_t, 0.1)),
                   "peak_time_q90": float(np.quantile(peak_t, 0.9)),
                   "peak_in_2_3s_ratio": float(np.mean((np.array(peak_t) >= 2) & (np.array(peak_t) < 3)))}

    # (5) 8클래스 × 5채널 활동 구간 RMS 히트맵
    M = np.zeros((len(CLASSES), N_CH))
    for i, c in enumerate(CLASSES):
        ang, g = c.split("_")
        M[i] = np.mean([window_rms(allt[(s, ang, g)], *ACT_WIN).mean(0) for s in SUBJECTS], 0)
    fig, axh = plt.subplots(figsize=(7, 5))
    im = axh.imshow(M * 1000, cmap="Blues", aspect="auto")
    axh.set_xticks(range(N_CH), [f"ch{i+1}\n{n}" for i, n in enumerate(CH_NAMES)], fontsize=8)
    axh.set_yticks(range(len(CLASSES)),
                   [f"{c.split('_')[0]}° {GESTURE_EN[c.split('_')[1]]}" for c in CLASSES], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(N_CH):
            v = M[i, j] * 1000
            axh.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                     color="white" if v > M.max() * 1000 * 0.6 else "#1F2937")
    fig.colorbar(im, ax=axh, fraction=0.04, label="활동 구간 RMS (×1000)")
    axh.set_title(f"8클래스 × 5채널 — 활동 구간({ACT_WIN[0]}~{ACT_WIN[1]}s) RMS, 8명 평균", fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS / "class_channel_rms.png", dpi=120)
    plt.close(fig)
    return env_profile, M


# ----------------------------------------------------------------------
# 단계 4-② / 5. 30° vs 60° 수치 비교
# ----------------------------------------------------------------------
def step5_angle_table(allt):
    rows = []
    for g in GESTURES:
        for ch in range(N_CH):
            r = {}
            for ang in ANGLES:
                act = np.array([window_rms(allt[(s, ang, g)], *ACT_WIN)[:, ch].mean() for s in SUBJECTS])
                rest = np.array([window_rms(allt[(s, ang, g)], *REST_WIN)[:, ch].mean() for s in SUBJECTS])
                r[ang] = (act, rest)
            ratio = r["60"][0].mean() / r["30"][0].mean()
            n_up = int((r["60"][0] > r["30"][0]).sum())
            rows.append({
                "gesture": g, "channel": CH_NAMES[ch],
                "act_rms_30": r["30"][0].mean(), "act_rms_60": r["60"][0].mean(),
                "ratio_60_30": ratio,
                "subjects_60_gt_30": n_up,
                "rest_rms_30": r["30"][1].mean(), "rest_rms_60": r["60"][1].mean(),
                "snr_30": (r["30"][0] / r["30"][1]).mean(),
                "snr_60": (r["60"][0] / r["60"][1]).mean(),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "angle_rms_table.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    return df


def main():
    set_korean_font()
    print("== 단계 1. 환경 점검"); step1_env_check()
    print("\n== 단계 2. 데이터 확보"); files = step2_list_files()
    print("\n== 단계 3. 데이터 구조 파악"); summary = step3_structure(files)
    print("\n== 단계 4. 채널별 신호 그리기")
    allt = load_all_trials(files)
    env_profile, M = step4_plots(allt)
    print(f"포락선 최대 시점 중앙값 {env_profile['peak_time_median']:.2f}s "
          f"(10~90%: {env_profile['peak_time_q10']:.2f}~{env_profile['peak_time_q90']:.2f}s), "
          f"2~3초 안에 최대가 오는 시행 비율 {env_profile['peak_in_2_3s_ratio']*100:.1f}%")
    print("\n== 단계 5. 30° vs 60° 비교 표")
    df = step5_angle_table(allt)
    print(df[["gesture", "channel", "ratio_60_30", "subjects_60_gt_30"]].to_string(index=False))
    out = {"env_profile": env_profile, "class_channel_rms": M.tolist(),
           "classes": CLASSES, "act_win": ACT_WIN, "rest_win": REST_WIN}
    (RESULTS / "signal_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print("\n완료: results/ 폴더 확인")


if __name__ == "__main__":
    main()
