# -*- coding: utf-8 -*-
"""
Step 2 · 2주차 과제(강의자료 17쪽) 제출물 6개 항목을 PDF 하나로 묶는다.
  ① 필터 검증       -> results/week2/filter_check.png
  ② 윈도우 확인     -> results/week2/check_log.txt, segment_windows.png, segment_start.png
  ③ 특징 구현 코드  -> src/week2_lab.py 의 td_features 전체
  ④ 분할 코드       -> src/week2_lab.py 의 시행 단위 분할 부분 (강조 표시)
  ⑤ 데이터셋 요약   -> results/week2/dataset_summary.json, class_feature_means.png
  ⑥ 설정 근거       -> 윈도우 200ms · 오버랩 50%
출력: report/19146809_김지훈_2주차.pdf

먼저 python src/week2_lab.py 로 results/week2/ 를 만든 뒤 실행한다.
실행: python report/generate_week2_report.py
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "report"))
import week2_lab as W  # noqa: E402
from generate_week1_report import (CODE_FONT, INK, MUTED, NAVY, PAGE_SIZE, body_text,  # noqa: E402
                                   code_box, draw_table, image, new_page, section_title,
                                   sub_title)

RESULTS = ROOT / "results" / "week2"
HEADER = "sEMG 동작·각도 동시 분류 · Step 2 2주차 제출물"


def source_lines(fn):
    lines, start = inspect.getsourcelines(fn)
    return [(start + i, l.rstrip("\n")) for i, l in enumerate(lines)]


def numbered_code(fig, y_top, numbered, highlight=(), line_h=0.0158, fontsize=7.9):
    """줄 번호가 붙은 코드 상자. highlight 에 든 줄 번호는 노란 배경으로 표시."""
    h = line_h * len(numbered) + 0.016
    ax = fig.add_axes([0.07, y_top - h, 0.86, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#F4F6F9", edgecolor="#D5DCE6"))
    step = 1 / (len(numbered) + 0.9)
    for i, (no, text) in enumerate(numbered):
        yy = 1 - step * (i + 0.95)
        if no in highlight:
            ax.add_patch(plt.Rectangle((0.004, yy - step / 2), 0.992, step,
                                       facecolor="#FFF1B8", edgecolor="none"))
        ax.text(0.012, yy, f"{no:>3}", fontsize=fontsize, color="#9AA5B4", family=CODE_FONT,
                va="center")
        ax.text(0.065, yy, text, fontsize=fontsize, color=INK, family=CODE_FONT, va="center")
    return y_top - h - 0.018


def generate():
    S = json.loads((RESULTS / "dataset_summary.json").read_text(encoding="utf-8"))
    log = (RESULTS / "check_log.txt").read_text(encoding="utf-8").rstrip().splitlines()
    F = S["filter"]; G = S["segment"]
    Z = pd.DataFrame(S["zc_table"])
    d = np.load(RESULTS / "dataset.npz")
    M = np.array(S["class_feature_means"])

    def gest(c):
        a, g = c.split("_")
        return f"{a}° {W.W1.GESTURE_EN[g]}"

    out_path = ROOT / "report" / "19146809_김지훈_2주차.pdf"
    with PdfPages(out_path) as pdf:
        # ---------------- 표지 ----------------
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor(NAVY)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.08, 0.72, "STEP 2 · 2주차 제출물", fontsize=13, color="#9FB3D1")
        ax.text(0.08, 0.615, "전처리와 특징 추출\n원신호 → 2초 → 19 × 200ms → 25차원",
                fontsize=24, color="white", fontweight="bold", linespacing=1.45)
        ax.text(0.08, 0.50,
                "60Hz 노치 + 20–450Hz 대역통과(filtfilt) → 2초 세그먼트(0.7~2.7s) →\n"
                "200ms·50% 윈도우 → RMS·MAV·WL·ZC·SSC × 5채널 → 시행 단위 8:2 분할 → 표준화",
                fontsize=11, color="#C7D2E5", linespacing=1.6)
        items = ["① 필터 검증", "② 윈도우 확인", "③ 특징 구현 코드", "④ 분할 코드",
                 "⑤ 데이터셋 요약", "⑥ 설정 근거"]
        for i, it in enumerate(items):
            x = 0.08 + (i % 2) * 0.43
            y = 0.40 - (i // 2) * 0.07
            ax.add_patch(plt.Rectangle((x, y - 0.03), 0.40, 0.055, transform=ax.transAxes,
                                       facecolor="#1B3560", edgecolor="none"))
            ax.text(x + 0.02, y - 0.003, it, fontsize=11.5, color="white", va="center")
        ax.text(0.08, 0.135, "학번 19146809 · 컴퓨터공학과 · 김지훈",
                fontsize=11, color="white", fontweight="bold")
        ax.text(0.08, 0.055,
                "논문: Simultaneous Classification of Hand Gestures and Elbow Angles Using sEMG Signals:\n"
                "A Comparison of Machine Learning Performances — Biomedical Engineering Letters (2026)\n"
                "코드: github.com/ddeeqq/learning_pj2 · 데이터셋 파일: results/week2/dataset.npz",
                fontsize=8, color="#7E93B8", linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ① 필터 검증 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ①")
        section_title(ax, 0.90, "① 필터 검증", "filter_check.png — 필터 전후 스펙트럼 비교 (실습 단계 1)")
        image(fig, RESULTS / "filter_check.png", [0.07, 0.355, 0.86, 0.50])
        lo, hi, p60, ps = (np.array(F[k]) for k in ("low_ratio", "high_ratio", "p60_ratio", "pass_ratio"))
        body_text(ax, 0.345,
            "슬라이드 9쪽 preprocess()를 그대로 썼다(노치 Q=30, 4차 Butterworth 20–450Hz, filtfilt, axis=0). "
            "파일 하나(30회 연속 기록)를 통째로 거른 뒤 시행을 잘라, 5초 조각 양 끝의 필터 가장자리 효과를 피했다.\n"
            f"• 20Hz 아래(1–15Hz) 평균 전력이 채널별로 {lo.min():.0f}~{lo.max():.0f}배, 450Hz 위(460–500Hz)는 "
            f"{hi.min():.0f}~{hi.max():.0f}배 줄었다. 대역통과의 양쪽 경계가 제대로 걸렸다.\n"
            f"• 근활성 대역(30–400Hz)의 전력은 {ps.min():.2f}~{ps.max():.2f}배로 거의 그대로다 — 필요한 신호는 "
            "건드리지 않았다.\n"
            f"• 60Hz 지점 전력은 {p60.min():.1f}~{p60.max():.1f}배 더 낮아졌다. 다만 before 곡선에도 60Hz 봉우리가 "
            "없고 오히려 홈이 파여 있다. 저장소의 'bandpass' 열이 이미 노치 + 20–450Hz 대역통과를 거친 신호이기 "
            "때문이다(1주차 확인). 그래서 이 그림은 잡음이 사라지는 장면이 아니라, 같은 필터를 한 번 더 걸어도 "
            "통과 대역은 유지되고 대역 밖만 추가로 눌린다는 확인으로 읽어야 한다.\n"
            "• 원 데이터에는 180·200·400Hz 부근에도 좁은 홈이 있다(저장소 측 처리로 추정). 우리 필터와는 무관하다.",
            fontsize=9.4, line_h=0.0192, width=60)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ② 윈도우 확인 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ②")
        y = section_title(ax, 0.90, "② 윈도우 확인", "윈도우 배열 모양 출력 화면 (실습 단계 1~3)")
        y = code_box(fig, y, ["$ python src/week2_lab.py"] + log[:10], dark=True,
                     line_h=0.0185, fontsize=8.6)
        image(fig, RESULTS / "segment_windows.png", [0.07, y - 0.36, 0.86, 0.35])
        body_text(ax, y - 0.375,
            "• 필터 후 5채널 표준편차가 출력되고, 세그먼트 (2000, 5) → 윈도우 (19, 200, 5)로 슬라이드의 기대값과 "
            "같다. 19 = (2000 − 200) / 100 + 1.\n"
            f"• 세그먼트 시작점은 슬라이드의 실습 가정(1.5초) 대신 {S['start'] / 1000:.1f}초를 썼다. 근거는 다음 쪽.\n"
            "• 아래 막대처럼 이웃 창은 100샘플씩 겹친다. 그래서 같은 시행의 창이 학습·테스트 양쪽에 들어가면 "
            "누수가 된다(④).",
            fontsize=9.4, line_h=0.0192, width=60)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ② 계속: 세그먼트 시작점 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ②")
        y = section_title(ax, 0.90, "② 윈도우 확인 (계속)", "2초 세그먼트 시작점 — 슬라이드 10쪽 '데이터에 맞게 조정'")
        image(fig, RESULTS / "segment_start.png", [0.07, 0.48, 0.86, 0.37])
        y = body_text(ax, 0.47,
            "슬라이드는 1.5초 시작을 '실습 가정이며 논문 설정이 아님'이라고 밝히고, 지난주 그림에서 동작 구간을 "
            "확인해 조정하라고 한다. 1주차에서 이 데이터의 근활성 최대 시점은 mark 기준 중앙값 1.69초"
            "(10~90%: 1.21~2.07초)로, 슬라이드가 가정한 '2~3초 동작'보다 앞섰다.\n"
            "그래서 시작점을 0~3초까지 0.1초 간격으로 옮겨 가며, 2초 구간이 각 시행의 버스트 에너지(마지막 1초 "
            "휴식을 기준선으로 뺀 값)를 몇 % 담는지 1,920개 시행 전체에서 계산했다.", fontsize=9.6, width=60)
        y -= 0.01
        y = draw_table(fig, y, ["시작점", "구간", "에너지 포함 (평균)", "에너지 포함 (하위 10% 시행)"], [
            [f"{W.SLIDE_START / 1000:.1f}s (슬라이드)", "1.5 ~ 3.5s", f"{G['slide_mean'] * 100:.1f}%",
             f"{G['slide_q10'] * 100:.1f}%"],
            [f"{S['start'] / 1000:.1f}s (채택)", "0.7 ~ 2.7s", f"{G['chosen_mean'] * 100:.1f}%",
             f"{G['chosen_q10'] * 100:.1f}%"],
        ], col_widths=[0.22, 0.2, 0.26, 0.32], fontsize=9, height=0.026)
        body_text(ax, y,
            f"• 평균이 가장 큰 시작점이 {G['best_start'] / 1000:.1f}초({G['best_mean'] * 100:.1f}%)였고 이를 채택했다. "
            f"슬라이드 기본값을 그대로 쓰면 평균 {G['slide_mean'] * 100:.0f}%만 담기고, 하위 10% 시행은 "
            f"{G['slide_q10'] * 100:.0f}%로 동작 대부분을 놓친다 — 19개 창 중 상당수가 휴식 구간이 되어 "
            "클래스 정보가 없는 표본이 늘어난다.\n"
            "• 이 선택은 라벨이 아니라 신호 에너지만 보고 정했고, 8클래스 모두에 같은 시작점을 쓰므로 "
            "분류 결과를 미리 보고 고른 값은 아니다.",
            fontsize=9.6, width=60)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ③ 특징 구현 코드 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ③")
        y = section_title(ax, 0.90, "③ 특징 구현 코드", "td_features 함수 전체 — src/week2_lab.py (numpy만 사용)")
        y = numbered_code(fig, y, source_lines(W.td_features), line_h=0.0172, fontsize=8.3)
        y = sub_title(ax, y, "25차원 순서 (리포트·다음 주 해석용)")
        chs = " · ".join(W.CH_NAMES)
        y = draw_table(fig, y, ["열 번호", "특징", "채널 순서", "식"], [
            ["0–4", "RMS", chs, "√(mean x²) — 신호 파워"],
            ["5–9", "MAV", chs, "mean |x|"],
            ["10–14", "WL", chs, "Σ |x[i+1] − x[i]|"],
            ["15–19", "ZC", chs, "부호 바뀜 & |차이| ≥ 0.01"],
            ["20–24", "SSC", chs, "기울기 부호 바뀜 횟수"],
        ], col_widths=[0.11, 0.1, 0.43, 0.36], fontsize=8.2, height=0.021)
        y = sub_title(ax, y, "ZC 임계값 관찰 — th를 0으로 바꿔 보기 (슬라이드 12쪽)")
        rows = [[f"{r.th:g}"] + [f"{getattr(r, 'ZC_' + c):.1f}" for c in W.CH_NAMES]
                + [f"{r.zero_window_ratio * 100:.1f}%"] for r in Z.itertuples()]
        y = draw_table(fig, y, ["th"] + W.CH_NAMES + ["ZC=0 창 비율"], rows,
                       col_widths=[0.1, 0.13, 0.13, 0.13, 0.13, 0.13, 0.25], fontsize=8.2, height=0.021)
        dq = S["abs_diff_quantiles"]
        body_text(ax, y,
            f"표: 36,480개 창의 채널별 평균 ZC. th=0이면 창당 22~29회지만 th=0.01이면 3~13회로 줄고, "
            f"채널-창 중 {Z.iloc[-1].zero_window_ratio * 100:.0f}%는 ZC가 0이 된다. 이 데이터의 인접 샘플 차이는 "
            f"중앙값 {dq[1]:.4f}, 90% 지점 {dq[2]:.4f}라서 0.01은 '잡음 제거'를 넘어 휴식 구간의 교차를 거의 "
            "다 지운다. 그 결과 ZC가 진폭(근수축 강도)과 비슷한 정보를 담게 된다. 이번에는 슬라이드·논문 조건"
            "(0.01)을 그대로 쓰고, 이 값이 신호 단위에 비해 크다는 점을 기록해 둔다.",
            fontsize=9.2, line_h=0.0186, width=62)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ④ 분할 코드 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ④")
        y = section_title(ax, 0.90, "④ 분할 코드", "시행 단위로 먼저 나눈 부분 — 노란 줄이 분할 (src/week2_lab.py)")
        sp = source_lines(W.split_trials_8_2)
        bd = source_lines(W.build)
        s5 = source_lines(W.step5_dataset)[:6]
        hl = {no for no, t in sp if "train_test_split" in t or "np.arange(len(trials))" in t
              or "stratify" in t or "[분할]" in t}
        hl |= {no for no, t in s5 if "split_trials_8_2" in t}
        y = numbered_code(fig, y, sp, highlight=hl, fontsize=7.8, line_h=0.0152)
        y = numbered_code(fig, y, bd, fontsize=7.8, line_h=0.0152)
        y = numbered_code(fig, y, s5, highlight=hl, fontsize=7.8, line_h=0.0152)
        y = body_text(ax, y,
            "• 분할 대상은 윈도우가 아니라 시행 번호 np.arange(1920)이다. 나눈 뒤에야 build()가 각 집합의 "
            "시행 안에서만 세그먼트 → 19개 창 → 특징을 만든다. 한 시행의 19개 창은 항상 같은 쪽에만 있다.\n"
            "• 슬라이드는 파일 하나 = 시행 하나를 가정하고 files를 나누지만, 이 데이터는 파일 하나에 30회가 "
            "들어 있다. 파일 단위로 나누면 일부 '피험자×클래스' 조합이 통째로 테스트로 빠지므로, 시행 단위로 "
            "나누고 stratify를 피험자×클래스(64그룹)로 걸어 그룹마다 정확히 24:6이 되게 했다.",
            fontsize=9.2, line_h=0.0186, width=62)
        draw_table(fig, y - 0.004, ["누수 점검", "결과"], [
            ["학습 ∩ 테스트 시행 수", f"{S['trial_overlap']}  (겹치는 시행 없음)"],
            ["학습 / 테스트 시행", f"{S['n_trials_train']:,} / {S['n_trials_test']:,}"],
            ["표준화 통계", "학습셋에서만 fit, 테스트셋은 transform (단계 6)"],
        ], col_widths=[0.34, 0.66], fontsize=8.6, height=0.022)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ⑤ 데이터셋 요약 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ⑤")
        y = section_title(ax, 0.90, "⑤ 데이터셋 요약", "학습·테스트 샘플 수 · 클래스별 분포 · 표준화 확인 (단계 5·6)")
        rows = []
        for c in W.CLASSES:
            r = S["class_counts"][c]
            rows.append([c, gest(c), f"{r['train_trials']}", f"{r['train_windows']:,}",
                         f"{r['test_trials']}", f"{r['test_windows']:,}"])
        rows.append(["합계", "8클래스", f"{S['n_trials_train']:,}", f"{S['Xtr_shape'][0]:,}",
                     f"{S['n_trials_test']}", f"{S['Xte_shape'][0]:,}"])
        y = draw_table(fig, y, ["라벨", "클래스", "학습 시행", "학습 샘플(창)", "테스트 시행", "테스트 샘플(창)"],
                       rows, col_widths=[0.13, 0.2, 0.15, 0.18, 0.15, 0.19], fontsize=8.3, height=0.0198)
        y = draw_table(fig, y, ["항목", "값"], [
            ["Xtr / Xte 모양", f"{tuple(S['Xtr_shape'])} / {tuple(S['Xte_shape'])} — 학습이 테스트의 4.0배"],
            ["피험자별 시행", "8명 모두 학습 192 · 테스트 48 (같은 사람이 양쪽에 있음 — 아래 참고)"],
            ["학습셋 표준화 후", f"평균 |최댓값| {S['train_mean_absmax']:.1e} · 표준편차 "
                                 f"{S['train_std_min']:.6f}~{S['train_std_max']:.6f}"],
            ["테스트셋 표준화 후", f"평균 |최댓값| {S['test_mean_absmax']:.3f} · 표준편차 "
                                   f"{S['test_std_min']:.3f}~{S['test_std_max']:.3f} (0·1이 아니어도 정상)"],
            ["상수 특징 / NaN", "0개 / 0개 — 25개 모두 표준화 가능"],
            ["dataset.npz", f"Xtr, ytr, Xte, yte + 시행 ID · 피험자 · 특징 이름 · 스케일러 통계 "
                            f"({S['npz_mb']:.1f}MB)"],
        ], col_widths=[0.24, 0.76], fontsize=8.2, height=0.0205, align="center")
        image(fig, RESULTS / "class_feature_means.png", [0.07, y - 0.30, 0.86, 0.29])
        body_text(ax, y - 0.31,
            "• 표준화된 특징의 클래스 평균을 보면 1주차의 관찰이 그대로 남아 있다: Fist는 전완 3채널, Open은 "
            "ED·FDS, Supination은 Biceps가 높고(60°에서 더 강함), Pronation은 전반적으로 낮다.\n"
            "• RMS·MAV·WL 세 블록은 거의 같은 무늬다(모두 진폭 계열). 25차원이지만 서로 강하게 상관된 특징이 "
            "많다는 뜻이고, 다음 주 모델 비교 때 해석에 참고한다.\n"
            "• 이 분할은 같은 피험자의 다른 시행이 양쪽에 들어가는 '피험자 내' 평가다. 처음 보는 사람에 대한 "
            "성능은 따로 피험자 단위 분할로 확인해야 한다.",
            fontsize=9.1, line_h=0.0184, width=63)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ⑥ 설정 근거 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ⑥")
        y = section_title(ax, 0.90, "⑥ 설정 근거", "윈도우 200ms · 오버랩 50%를 고른 이유 (각 두 문장)")
        memo = [
            ("윈도우 200ms",
             "sEMG는 수백 ms 안에서는 통계적 성질이 거의 변하지 않으므로(준정상성) 200ms 조각 하나는 "
             "안정된 신호로 다룰 수 있고, 그 안에 20Hz 성분도 4주기가 들어가 RMS·ZC 같은 통계량이 흔들리지 "
             "않을 만큼 샘플(200개)이 충분하다. 동시에 근전도 기반 제어에서 허용되는 지연(약 300ms) 안에 "
             "들어가므로, 실시간 기기 제어를 목표로 하는 이 문제의 조건과 맞는다."),
            ("오버랩 50% (100ms 이동)",
             "100ms씩 밀면 2초 세그먼트 하나에서 겹침 없이 자를 때(10개)의 약 두 배인 19개 표본이 나와, "
             "시행 수가 고정된 데이터에서 학습 표본을 늘리고 동작이 시작·끝나는 경계도 놓치지 않는다. 또 "
             "실제 사용 시 100ms마다 새 판단을 낼 수 있어 반응이 빨라지며, 겹친 창끼리 비슷하다는 대가는 "
             "시행 단위 분할(④)로 누수가 되지 않게 막았다."),
        ]
        for title, body in memo:
            ax.text(0.07, y, title, fontsize=13, fontweight="bold", color=NAVY)
            y -= 0.032
            y = body_text(ax, y, body, width=56, line_h=0.0225, fontsize=10.3)
            y -= 0.025
        y = sub_title(ax, y, "함께 정한 설정")
        draw_table(fig, y, ["설정", "값", "근거"], [
            ["필터", "노치 60Hz(Q=30) + BPF 20–450Hz 4차", "논문 조건, Q·filtfilt는 실습 설정 (①)"],
            ["세그먼트", "2초, 0.7~2.7s", "버스트 에너지 91% 포함, 슬라이드값 68% (②)"],
            ["ZC 임계값", "0.01", "슬라이드·논문 조건 유지, 신호 대비 크다는 점 기록 (③)"],
            ["분할", "시행 단위 8:2, 피험자×클래스 층화", "누수 방지 + 그룹마다 24:6 (④)"],
            ["표준화", "StandardScaler, 학습셋 fit", "테스트 통계 누출 방지 (⑤)"],
            ["시드", "random_state = 42", "다음 주에도 같은 분할"],
        ], col_widths=[0.16, 0.38, 0.46], fontsize=8.4, height=0.025)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- 부록 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · 부록")
        y = section_title(ax, 0.90, "부록 · 점검표와 재현 방법", "강의자료 15쪽 체크포인트")
        y = draw_table(fig, y, ["체크포인트", "결과"], [
            ["필터 적용 후 5채널 표준편차 출력", "완료 — ② 출력 화면"],
            ["세그먼트가 (2000, 5)로 잘림", "완료"],
            ["윈도우 배열이 (19, 200, 5)", "완료"],
            ["특징 벡터가 (25,)", "완료 — NaN 0개"],
            ["시행 단위로 먼저 분할한 뒤 특징 생성", "완료 — ④, 겹치는 시행 0"],
            ["학습셋 평균 0 · 분산 1 확인", "완료 — ⑤"],
            ["dataset.npz 저장", f"완료 — results/week2/dataset.npz ({S['npz_mb']:.1f}MB)"],
        ], col_widths=[0.5, 0.5], fontsize=9, height=0.025)
        y = sub_title(ax, y, "1주차 대비 달라진 점 · 기록")
        y = body_text(ax, y,
            "• 세그먼트 시작점: 1주차 관찰(활동 1.0~2.5s)을 반영해 0.7s로 조정.\n"
            "• 분할 단위: 파일이 아니라 시행(1,920개). 파일 하나 = 30회라는 1주차 확인 결과에 따름.\n"
            "• 저장 형식: np.savez_compressed — np.load로 그대로 읽힌다(키는 슬라이드와 같은 Xtr·ytr·Xte·yte).",
            fontsize=9.6)
        y -= 0.01
        y = sub_title(ax, y, "재현 방법")
        y = code_box(fig, y, [
            "$ pip install -r requirements.txt",
            "$ git clone https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset \\",
            "      data/sEMG-Hand-Gesture-Elbow-Angle-Dataset",
            "$ python src/week2_lab.py               # 단계 1~6, results/week2/ 생성 (약 30초)",
            "$ python report/generate_week2_report.py  # 이 PDF 생성",
            "",
            ">>> d = np.load('results/week2/dataset.npz')",
            ">>> d['Xtr'].shape, d['Xte'].shape      # (29184, 25) (7296, 25)",
        ], dark=True, line_h=0.02, fontsize=8.3)
        body_text(ax, y,
            "다음 주(3주차): SVM · Random Forest · XGBoost를 dataset.npz로 학습하고, hold-out과 5-fold 교차검증, "
            "8클래스 혼동행렬, 대응표본 t-검정으로 비교한다.", fontsize=9.6, color=MUTED)
        pdf.savefig(fig); plt.close(fig)

    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    generate()
