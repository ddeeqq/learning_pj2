# -*- coding: utf-8 -*-
"""
Step 2 · 1주차 과제(강의자료 16쪽) 제출물 5개 항목을 PDF 하나로 묶는다.
  ① 환경 확인     -> results/week1/env_check.txt
  ② 데이터 요약   -> results/week1/data_summary.json, class_counts.csv
  ③ 신호 그림     -> results/week1/ch5_example.png, class_channel_rms.png
  ④ 비교 관찰     -> results/week1/angle_side_by_side.png, angle_envelope_grid.png, angle_rms_table.csv
  ⑤ 클래스 설계안 -> 5개 항목 한 페이지
출력: report/19146809_김지훈_1주차.pdf

먼저 python src/week1_lab.py 로 results/week1/ 를 만든 뒤 실행한다.
실행: python report/generate_week1_report.py
"""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import week1_lab as L  # noqa: E402

L.set_korean_font()
KFONT = plt.rcParams["font.family"][0] if isinstance(plt.rcParams["font.family"], list) \
    else plt.rcParams["font.family"]
from matplotlib import font_manager  # noqa: E402
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
CODE_FONT = [f for f in ["Consolas", "DejaVu Sans Mono"] if f in _HAVE] + [KFONT]

RESULTS = ROOT / "results" / "week1"
PAGE_SIZE = (8.27, 11.69)  # A4
HEADER = "sEMG 동작·각도 동시 분류 · Step 2 1주차 제출물"
NAVY, BLUE, INK, MUTED = "#12233F", "#4A6FA5", "#1F2937", "#6B7280"


# ----------------------------------------------------------------------
# 페이지 헬퍼 (learning_pj1 리포트와 같은 양식)
# ----------------------------------------------------------------------
def wrap(text, width=50):
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        indent = "   " if para.startswith("• ") else ""
        out.extend(textwrap.wrap(para, width=width, subsequent_indent=indent) or [""])
    return out


def new_page(pdf, header):
    fig = plt.figure(figsize=PAGE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.07, 0.965, header, fontsize=10, color=BLUE, fontweight="bold")
    ax.axhline(0.955, xmin=0.07, xmax=0.93, color=BLUE, linewidth=0.8)
    return fig, ax


def section_title(ax, y, title, subtitle=None):
    ax.text(0.07, y, title, fontsize=19, fontweight="bold", color=NAVY)
    if subtitle:
        ax.text(0.07, y - 0.032, subtitle, fontsize=10.5, color=MUTED)
    return y - (0.07 if subtitle else 0.05)


def body_text(ax, y, text, fontsize=10, width=58, line_h=0.0205, color=INK, x=0.07):
    for line in wrap(text, width=width):
        ax.text(x, y, line, fontsize=fontsize, color=color, va="top")
        y -= line_h
    return y


def sub_title(ax, y, text):
    ax.text(0.07, y, text, fontsize=12.5, fontweight="bold", color=NAVY, va="top")
    return y - 0.03


def draw_table(fig, y_top, col_labels, rows, col_widths=None, fontsize=8.8, height=0.024,
               left=0.07, width=0.86, align="center"):
    n = len(rows) + 1
    tax = fig.add_axes([left, y_top - height * n, width, height * n])
    tax.axis("off")
    tb = tax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc=align,
                   colWidths=col_widths, bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(fontsize)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#D5DCE6")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F3F6FB" if r % 2 == 0 else "white")
    return y_top - height * n - 0.018


def code_box(fig, y_top, lines, line_h=0.0168, fontsize=8.2, numbered=False, dark=False):
    h = line_h * len(lines) + 0.018
    ax = fig.add_axes([0.07, y_top - h, 0.86, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    bg, ed, fg = ("#0F172A", "#0F172A", "#E5E7EB") if dark else ("#F4F6F9", "#D5DCE6", INK)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=bg, edgecolor=ed))
    step = 1 / (len(lines) + 1.0)
    for i, text in enumerate(lines):
        yy = 1 - step * (i + 0.95)
        color = fg
        if dark and text.startswith("#"):
            color = "#94A3B8"
        if dark and text.startswith("$"):
            color = "#86EFAC"
        ax.text(0.02, yy, text, fontsize=fontsize, color=color, family=CODE_FONT, va="center")
    return y_top - h - 0.02


def image(fig, path, rect):
    a = fig.add_axes(rect)
    a.imshow(plt.imread(path))
    a.axis("off")


# ----------------------------------------------------------------------
# 리포트
# ----------------------------------------------------------------------
def generate():
    S = json.loads((RESULTS / "data_summary.json").read_text(encoding="utf-8"))
    T = json.loads((RESULTS / "signal_stats.json").read_text(encoding="utf-8"))
    A = pd.read_csv(RESULTS / "angle_rms_table.csv")
    env_txt = (RESULTS / "env_check.txt").read_text(encoding="utf-8").rstrip().splitlines()
    M = np.array(T["class_channel_rms"]) * 1000
    ep = T["env_profile"]
    act0, act1 = T["act_win"]

    def cls_name(c):
        a, g = c.split("_")
        return f"{a}° {L.GESTURE_EN[g]}"

    def ratio(g, ch):
        r = A[(A.gesture == g) & (A.channel == ch)].iloc[0]
        return r

    out_path = ROOT / "report" / "19146809_김지훈_1주차.pdf"
    with PdfPages(out_path) as pdf:
        # ---------------- 표지 ----------------
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor(NAVY)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.text(0.08, 0.72, "STEP 2 · 1주차 제출물", fontsize=13, color="#9FB3D1")
        ax.text(0.08, 0.615, "문제 정의와 데이터 이해\n손동작 4종 × 팔꿈치 각도 2종 = 8클래스",
                fontsize=24, color="white", fontweight="bold", linespacing=1.45)
        ax.text(0.08, 0.50,
                "환경 점검 → 데이터 확보(논문 원 데이터) → 구조 파악(64파일 · 1,920시행) →\n"
                "5채널 신호 그리기 → 30° vs 60° 비교 → 클래스 설계안",
                fontsize=11, color="#C7D2E5", linespacing=1.6)
        items = ["① 환경 확인", "② 데이터 요약", "③ 신호 그림", "④ 비교 관찰", "⑤ 클래스 설계안",
                 "부록 · 점검표"]
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
                "코드: github.com/ddeeqq/learning_pj2 · "
                "데이터: github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset",
                fontsize=8, color="#7E93B8", linespacing=1.5)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ① 환경 확인 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ①")
        y = section_title(ax, 0.90, "① 환경 확인", "패키지 버전 출력 화면 — 실습 단계 1")
        y = body_text(ax, y,
            "Step 1에서 쓰던 가상환경을 그대로 활성화하고, 이번 주에 새로 필요한 xgboost 하나만 "
            "추가 설치했다. 아래는 강의자료 9쪽의 확인 코드를 그대로 실행한 출력이다.")
        y -= 0.01
        y = code_box(fig, y, [
            "# Step 1 프로젝트 폴더에서",
            "$ source .venv/bin/activate      # Windows: .venv\\Scripts\\activate",
            "$ pip install xgboost",
            "$ python src/week1_lab.py        # 단계 1: 아래 버전 출력 + results/week1/env_check.txt 저장",
        ], dark=True, line_h=0.02, fontsize=8.6)
        y = code_box(fig, y, env_txt, dark=True, line_h=0.022, fontsize=9.5)
        y = body_text(ax, y,
            "• 네 줄(numpy · scipy · sklearn · xgboost)의 버전이 모두 출력되었으므로 성공이다.\n"
            "• 이번 주는 모델을 만들지 않으므로 딥러닝 패키지(torch)는 쓰지 않는다. Step 1 환경은 "
            "지우지 않고 그대로 두었다.\n"
            "• 같은 출력은 results/week1/env_check.txt 에도 저장된다.")
        y -= 0.015
        draw_table(fig, y, ["패키지", "용도"], [
            ["numpy · pandas", "CSV 읽기, 시행 분할, RMS 계산"],
            ["scipy", "신호 처리 (scipy.signal)"],
            ["scikit-learn", "머신러닝 라이브러리"],
            ["xgboost", "이번 주 신규 설치"],
            ["matplotlib", "5채널 파형 · 포락선 · 히트맵 그림"],
        ], col_widths=[0.3, 0.7])
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ② 데이터 요약 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ②")
        y = section_title(ax, 0.90, "② 데이터 요약",
                          "클래스별 시행 수 · 배열 모양 · 채널 수 — 실습 단계 2·3")
        y = body_text(ax, y,
            "데이터는 논문 데이터로 GitHub에 공개된 원 데이터(8명 × 4동작 × 2각도 × 30회, "
            "5채널, 1,000Hz)를 썼다. 방법 B(Ninapro 축소)가 아니므로 8클래스를 그대로 구성할 수 "
            "있다. 받은 폴더는 data/ 아래에 두었다.")
        y -= 0.008
        y = sub_title(ax, y, "파일 구조 — 슬라이드 가정과 다른 점")
        y = body_text(ax, y,
            f"• 파일 이름은 '[피험자]_[각도] [동작].csv' (예: a_30 fist.csv) 이고, 파일 하나에 "
            f"30회 시행이 이어 붙어 있다. 그래서 파일 수는 {S['n_files']}개(8명 × 8클래스)뿐이다.\n"
            f"• 열은 {', '.join(S['raw_columns'][:5])} 다섯 채널과 time, onset, mark이다. mark가 1인 "
            f"지점이 시행 시작이며 간격이 {S['mark_gap_min']:,}~{S['mark_gap_max']:,}샘플(≈5초)로 "
            "일정해서, mark마다 5,000샘플씩 잘라 시행 배열을 만들었다.\n"
            "• 슬라이드의 label_of는 '30_fist_trial01.csv' 형식을 가정하므로, 파일명 규칙에 맞게 "
            "아래처럼 고쳤다(강의자료 15쪽 '클래스 개수가 안 맞음' 해결법).")
        y -= 0.004
        y = code_box(fig, y, [
            "NAME_RE = re.compile(r\"^([a-h])_(30|60) (fist|open|pro|extor)$\")",
            "def label_of(path):                       # 'a_30 fist.csv' -> '30_fist'",
            "    sub, ang, ges = NAME_RE.match(Path(path).stem).groups()",
            "    return f\"{ang}_{ges}\"",
            "def split_trials(x, marks):               # (N,5) -> (30, 5000, 5)",
            "    return np.stack([x[s:s+5000] for s in marks if s+5000 <= len(x)])",
        ], line_h=0.0165, fontsize=7.9)
        rows = []
        for c in L.CLASSES:
            rows.append([c, cls_name(c), "8", "8", f"{S['trial_counts'][c]}", "(5000, 5)"])
        rows.append(["합계", "8클래스", f"{S['n_files']}", "8", f"{S['total_trials']:,}", "—"])
        y = draw_table(fig, y, ["라벨", "클래스", "파일 수", "피험자", "시행 수", "시행 배열"],
                       rows, col_widths=[0.15, 0.22, 0.12, 0.12, 0.14, 0.18], fontsize=8.3,
                       height=0.0198)
        y = draw_table(fig, y, ["항목", "값"], [
            ["파일 원본 배열 모양 (예: a_30 extor.csv)",
             f"({S['raw_file_shape'][0]:,}, {S['raw_file_shape'][1]})  — 약 {S['raw_file_shape'][0] / 1000:.0f}초 연속 기록"],
            ["파일 길이 범위", f"{S['samples_min']:,} ~ {S['samples_max']:,} 샘플 (모두 30회 시행 포함)"],
            ["시행 1개 배열 모양 / 채널 수", f"(5000, 5) — 5초 × 1,000Hz / {S['n_channels']}채널"],
            ["전체 시행 배열", f"({S['total_trials']:,}, 5000, 5)  = 8명 × 8클래스 × 30회"],
            ["클래스 균형", "8클래스 모두 240시행 — 불균형 없음"],
        ], col_widths=[0.42, 0.58], fontsize=8.3, height=0.0198)
        body_text(ax, y,
            "8클래스 모두 240회로 논문의 라벨 세그먼트 수(1,920개)와 일치한다. 저장소 설명상 신호는 이미 "
            "60Hz 노치 + 20–450Hz 대역통과가 적용된 상태다.", fontsize=9.3, line_h=0.019)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ③ 신호 그림 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ③")
        y = section_title(ax, 0.90, "③ 신호 그림",
                          "ch5_example.png — 5채널 파형, 동작 구간 표시 (실습 단계 4)")
        image(fig, RESULTS / "ch5_example.png", [0.06, 0.465, 0.88, 0.40])
        body_text(ax, 0.455,
            "슬라이드 12쪽 코드 그대로 피험자 a의 30° 주먹 쥐기 첫 시행을 그렸다. 빨간 점선은 슬라이드의 "
            "2~3초(동작 1초) 표시이고, 파란 음영은 실제로 근활성이 보인 구간이다.\n"
            "관찰 내용\n"
            f"• 휴식–동작–휴식 3구간은 분명히 보이지만, 동작 구간이 2~3초가 아니라 약 {act0}~{act1}초에 "
            f"있다. 64개 파일 1,920개 시행 전체에서 포락선 최대 시점의 중앙값은 {ep['peak_time_median']:.2f}초"
            f"(10~90%: {ep['peak_time_q10']:.2f}~{ep['peak_time_q90']:.2f}초)였고, 최대가 2~3초 안에 오는 "
            f"시행은 {ep['peak_in_2_3s_ratio'] * 100:.1f}%뿐이었다. mark가 메트로놈 박자 기준이라 실제 "
            "움직임이 표시보다 앞서 시작하는 것으로 보인다.\n"
            "• 주먹 쥐기에서는 전완 3채널(ECR·ED·FDS)이 크게 반응하고, ED가 가장 크다(±0.5). "
            "상완 채널(Biceps·Triceps)은 진폭이 한 자릿수 작다.\n"
            "• ECR·Triceps에는 휴식 구간에도 약 0.6초 간격의 짧은 스파이크가 반복된다. 근수축과 무관하게 "
            "주기적으로 나타나므로 심전도(ECG) 혼입 같은 잡음으로 추정하며, 원인은 확인하지 않았다.",
            fontsize=9.6, line_h=0.0198, width=60)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ③-2 클래스 × 채널 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ③")
        y = section_title(ax, 0.90, "③ 신호 그림 (계속)",
                          "동작별로 어떤 채널이 반응하는가 — 8클래스 × 5채널 활동 구간 RMS")
        image(fig, RESULTS / "class_channel_rms.png", [0.10, 0.50, 0.80, 0.36])
        top = [L.CLASSES[i] for i in M.argmax(0)]
        body_text(ax, 0.485,
            f"각 시행의 활동 구간({act0}~{act1}초) RMS를 구해 8명 × 30회 평균을 냈다(값 × 1000).\n"
            f"• 주먹 쥐기: 전완 3채널이 모두 높다(ED {M[0,1]:.0f} · FDS {M[0,2]:.0f} · ECR {M[0,0]:.0f}). "
            "손가락 굴근과 손목 신근이 함께 수축해 주먹을 고정하는 패턴이다.\n"
            f"• 손 펴기: ED·FDS가 가장 크다(FDS {M[2,2]:.0f}, ED {M[2,1]:.0f}). 굴근 채널(FDS)이 펴기에서 "
            "가장 크게 나온 것은 해부학적 기대와 다르다. 전극 간 혼선(crosstalk)이나 펴기 직후 되돌아오는 "
            "동작의 영향일 수 있는데, 논문도 채널별 기여도는 따로 검증하지 않았으므로 여기서는 기록만 한다.\n"
            f"• 회내: 다섯 채널 모두 고르게 낮다(최대 {M[4:6].max():.0f}). 신호가 약해 다른 동작보다 "
            "구분이 어려울 것으로 예상한다.\n"
            f"• 회외: Biceps가 지배적이다(30° {M[6,3]:.0f} → 60° {M[7,3]:.0f}). 상완이두근이 회외근으로도 "
            "작용한다는 설명과 맞는다.\n"
            f"• 채널별 최대 반응 클래스: " + ", ".join(
                f"ch{i + 1} {L.CH_NAMES[i]} → {cls_name(top[i])}" for i in range(5)) + ".\n"
            "결론적으로 '동작'은 채널 간 상대 패턴(어느 채널이 켜지는가)으로 크게 갈리고, '각도'는 같은 "
            "패턴 안에서 크기가 조금 달라지는 정도다. 다음 페이지에서 이를 확인한다.",
            fontsize=9.6, line_h=0.0198, width=60)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ④ 비교 관찰 (1) ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ④")
        y = section_title(ax, 0.90, "④ 비교 관찰",
                          "같은 동작의 30°와 60° — 한 피험자, 같은 시행 번호, 같은 y축")
        image(fig, RESULTS / "angle_side_by_side.png", [0.05, 0.50, 0.90, 0.36])
        r_b = ratio("fist", "Biceps"); r_t = ratio("fist", "Triceps")
        body_text(ax, 0.485,
            "피험자 a의 주먹 쥐기 첫 시행을 30°(왼쪽)와 60°(오른쪽)로 나란히 그렸다. 비교가 왜곡되지 않도록 "
            "채널마다 두 그림의 y축 범위를 같게 맞췄다.\n"
            "차이점\n"
            "• 전완 채널 중 ED·FDS는 버스트 모양과 크기가 거의 같고, ECR은 이 시행에서 60°가 다소 작다. "
            "손동작 자체의 패턴(어느 채널이 켜지는가)은 각도가 바뀌어도 유지된다.\n"
            "• 상완이두근(Biceps)은 60°에서 버스트가 더 앞쪽(약 1.5초)에서 더 크게 나타나고, 30°에서는 "
            "동작 끝(약 2.4초)에 짧게 튄다. 팔꿈치 자세가 바뀌면 같은 주먹 쥐기라도 상완 근육이 개입하는 "
            "시점과 크기가 달라진다.\n"
            "• 상완삼두근(Triceps)은 60°에서 주기적 스파이크의 진폭이 더 크다. 근수축보다 잡음 성분이 "
            "자세에 따라 달라진 것으로 보이며, 이런 성분이 특징에 섞이면 분류기가 각도 대신 잡음을 "
            "학습할 위험이 있다.\n"
            f"• 8명 전체 평균으로 보면 주먹 쥐기의 활동 구간 RMS는 Biceps {r_b.ratio_60_30:.2f}배, "
            f"Triceps {r_t.ratio_60_30:.2f}배(60°/30°)로 상완 채널만 커졌고, 전완 채널은 0.95~0.99배로 "
            "거의 변하지 않았다.",
            fontsize=9.6, line_h=0.0198, width=60)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ④ 비교 관찰 (2) ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ④")
        y = section_title(ax, 0.90, "④ 비교 관찰 (계속)",
                          "8명 × 30회 평균 포락선과 60°/30° RMS 비율")
        image(fig, RESULTS / "angle_envelope_grid.png", [0.07, 0.445, 0.86, 0.40])
        rows = []
        for g in L.GESTURES:
            row = [f"{L.GESTURE_EN[g]}"]
            for ch in L.CH_NAMES:
                r = ratio(g, ch)
                row.append(f"{r.ratio_60_30:.2f} ({int(r.subjects_60_gt_30)}/8)")
            rows.append(row)
        y = draw_table(fig, 0.435, ["동작 \\ 채널"] + L.CH_NAMES, rows,
                       col_widths=[0.2] + [0.16] * 5, fontsize=8.4, height=0.021)
        r_sb = ratio("extor", "Biceps"); r_sf = ratio("extor", "FDS")
        n_up = int((A.ratio_60_30 > 1).sum())
        body_text(ax, y,
            "표: 활동 구간 RMS의 60°/30° 비율, 괄호는 60°가 더 큰 피험자 수(8명 중).\n"
            f"• 20칸 중 {n_up}칸에서 60°의 근활성이 더 컸다. 각도는 새로운 채널을 켜기보다 "
            "같은 패턴의 크기를 1.0~1.5배 바꾸는 방식으로 신호에 나타난다.\n"
            f"• 가장 큰 차이는 회외의 Biceps({r_sb.ratio_60_30:.2f}배, {int(r_sb.subjects_60_gt_30)}/8명)와 "
            f"FDS({r_sf.ratio_60_30:.2f}배, 8명 전원)다. 회외는 상완이두근이 직접 관여하므로 팔꿈치 각도의 "
            "영향을 가장 크게 받는다.\n"
            "• 반면 주먹 쥐기·손 펴기의 전완 채널은 0.95~1.08배이고 방향도 4~5명씩 엇갈렸다. 각도 정보는 "
            "주로 상완 2채널에 있고, 그마저 사람마다 방향이 다르다 — 개인차가 각도 구분을 흐릴 수 있다는 "
            "뜻이다.\n"
            "• 포락선 모양도 60°에서 버스트가 약간 일찍·뾰족하게 나타나는 경향이 있다(Open의 ED·FDS).",
            fontsize=9.4, line_h=0.0193, width=61)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- ⑤ 클래스 설계안 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · ⑤")
        y = section_title(ax, 0.91, "⑤ 클래스 설계안", "손동작 4종 × 팔꿈치 각도 2종 = 8클래스")
        y = sub_title(ax, y + 0.008, "① 클래스 목록")
        defs = {"fist": "손가락을 모두 굽혀 주먹을 쥠", "open": "손가락을 모두 폄",
                "pro": "손등이 위로 오도록 전완을 안쪽 회전", "extor": "손바닥이 위로 오도록 전완을 바깥 회전"}
        rows = [[f"C{i}", c, f"{L.GESTURE_KO[c.split('_')[1]]} ({L.GESTURE_EN[c.split('_')[1]]})",
                 f"{c.split('_')[0]}°", defs[c.split('_')[1]]] for i, c in enumerate(L.CLASSES)]
        y = draw_table(fig, y, ["번호", "라벨", "손동작", "각도", "정의 (팔꿈치는 플레이트로 고정)"], rows,
                       col_widths=[0.07, 0.13, 0.25, 0.08, 0.47], fontsize=7.9, height=0.0185)
        y = sub_title(ax, y + 0.004, "② 채널 대응")
        y = draw_table(fig, y, ["채널", "측정 근육", "부착 부위", "주 기능", "이번 데이터 최대 반응"], [
            [f"ch{i + 1}", f"{L.CH_KO[i]} ({L.CH_NAMES[i]})", loc, fn, cls_name(top[i])]
            for i, (loc, fn) in enumerate([
                ("전완 후면", "손목 젖히기"), ("전완 후면", "손가락 펴기"), ("전완 전면", "손가락 굽히기"),
                ("상완 전면", "팔꿈치 굽히기·회외"), ("상완 후면", "팔꿈치 펴기")])
        ], col_widths=[0.08, 0.28, 0.14, 0.22, 0.28], fontsize=7.9, height=0.0185)
        y = sub_title(ax, y + 0.004, "③ 설계 근거")
        y = body_text(ax, y,
            "기기를 제어하려면 '무슨 손동작을'과 '팔이 어떤 자세에서'를 함께 알아야 하고, ④에서 본 것처럼 "
            "각도가 같은 손동작의 신호 크기를 바꾸므로(회외 Biceps 1.55배) 두 요인은 서로 독립이 아니다. "
            "그래서 두 축을 곱한 조합 하나를 클래스 하나로 두어, 경량 분류기 하나가 동작과 각도의 상호작용까지 "
            "한 번의 예측으로 학습하게 했다.", fontsize=9.1, line_h=0.0182, width=64)
        y = sub_title(ax, y - 0.004, "④ 대안 검토 — 동작과 각도를 따로 예측")
        y = draw_table(fig, y, ["방식", "장점", "단점"], [
            ["8클래스 결합 (채택)", "상호작용 학습, 모델 1개", "조합마다 데이터 필요, 클래스 수 곱으로 증가"],
            ["분리 (4클래스 + 2클래스)", "클래스당 데이터 많음, 확장 쉬움", "동작별 각도 효과 차이 무시, 오류 2번 누적"],
            ["계층형 (동작 → 각도)", "동작별로 각도 분류기 특화", "모델 5개, 1단계 오류가 그대로 전파"],
        ], col_widths=[0.25, 0.33, 0.42], fontsize=7.9, height=0.0185)
        y = sub_title(ax, y + 0.004, "⑤ 확장 시 문제 — 각도를 3종·4종으로 늘리면")
        body_text(ax, y,
            "• 클래스가 12개·16개로 늘고, 1인 측정 시간(30회 × 5초 기준 순수 동작만)이 20분 → 30분 → 40분으로 "
            "늘어 근피로가 신호를 바꾼다.\n"
            "• 30° 간격에서도 차이가 1.0~1.5배이고 피험자마다 방향이 엇갈렸는데, 15° 간격이면 이웃 각도의 "
            "신호가 더 비슷해져 인접 각도끼리 혼동이 커진다.\n"
            "• 각도는 순서가 있는 연속값인데 분류는 30°↔45° 오류와 30°↔75° 오류를 똑같이 취급한다 — "
            "각도 수가 많아지면 분리 방식이나 각도 회귀가 더 맞을 수 있다.",
            fontsize=9.1, line_h=0.0182, width=64)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- 부록: 점검표 ----------------
        fig, ax = new_page(pdf, f"{HEADER} · 부록")
        y = section_title(ax, 0.90, "부록 · 점검표와 재현 방법", "강의자료 14쪽 체크포인트")
        y = draw_table(fig, y, ["체크포인트", "결과"], [
            ["가상환경에서 xgboost import", "완료 — ① 참조"],
            ["데이터 파일 개수와 클래스별 분포", f"완료 — {S['n_files']}파일, 클래스당 240시행"],
            ["신호 길이와 5채널 구성", "완료 — 시행당 (5000, 5)"],
            ["ch5_example.png 5채널 파형", "완료 — ③ 참조"],
            ["동작 구간의 채널별 파형 관찰", "완료 — ③ 히트맵·관찰"],
            ["클래스 설계안 한 페이지", "완료 — ⑤ 참조"],
        ], col_widths=[0.5, 0.5], fontsize=9, height=0.026)
        y = sub_title(ax, y, "데이터가 논문·슬라이드와 다른 점 (기록)")
        y = body_text(ax, y,
            "• 파일 하나 = 30회 시행 연속 기록. mark 열로 시행을 잘라야 한다(슬라이드는 시행별 파일 가정).\n"
            f"• 근활성 최대 시점이 mark 기준 약 {ep['peak_time_median']:.1f}초로, 슬라이드의 '2~3초 동작' "
            "표시보다 앞선다. 억지로 맞추지 않고 실제 구간을 기록했다.\n"
            "• 손 펴기에서 굴근 채널(FDS)이 가장 크게 나왔다 — 혼선 가능성, 검증하지 않음.\n"
            "• 휴식 구간의 주기적 스파이크(ECR·Triceps) — 잡음으로 추정, 원인은 확인하지 않음.",
            fontsize=9.6)
        y -= 0.01
        y = sub_title(ax, y, "재현 방법")
        y = code_box(fig, y, [
            "$ git clone https://github.com/ddeeqq/learning_pj2 && cd learning_pj2",
            "$ pip install -r requirements.txt",
            "$ git clone https://github.com/TestUser-alt-dev/sEMG-Hand-Gesture-Elbow-Angle-Dataset \\",
            "      data/sEMG-Hand-Gesture-Elbow-Angle-Dataset",
            "$ python src/week1_lab.py           # 단계 1~5, results/week1/ 생성 (약 1분)",
            "$ python report/generate_week1_report.py  # 이 PDF 생성",
        ], dark=True, line_h=0.02, fontsize=8.3)
        pdf.savefig(fig); plt.close(fig)

    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    generate()
