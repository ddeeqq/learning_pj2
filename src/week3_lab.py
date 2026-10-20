# -*- coding: utf-8 -*-
"""
Step 2 · 3주차 실습 — 모델 비교와 통계 검정 (6단계)

  단계 1  세 모델 학습      SVM(RBF) · Random Forest · XGBoost, 같은 데이터(dataset.npz)·같은 분할
  단계 2  hold-out 평가     Accuracy · macro Precision · macro Recall · macro F1 (+ 시행 단위 다수결)
  단계 3  혼동행렬          8클래스, 동작 오류 / 각도 오류 구분       -> results/week3/confusion_*.png
  단계 4  5-fold 교차검증   시행 단위 StratifiedKFold, 폴드마다 scaler 새로 fit
  단계 5  통계 검정         폴드별 정확도로 대응표본 t-검정 (세 쌍)
  단계 6  계산 비용         학습 시간 · 1샘플 추론 시간
  (보조)  LOSO              7명 학습 → 남은 1명 평가 (trial-wise 분할 한계 확인용)

결과는 results/week3/results.json 에 모은다.
실행: python src/week3_lab.py            (전체, 2코어 CPU 기준 약 20분)
      python src/week3_lab.py --no-loso  (LOSO 생략)
      python src/week3_lab.py --holdout-only  (hold-out만 다시 계산)
"""
import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week1_lab as W1  # noqa: E402
import week2_lab as W2  # noqa: E402

ROOT = W1.ROOT
RESULTS = ROOT / "results" / "week3"
RESULTS.mkdir(parents=True, exist_ok=True)
DATASET = ROOT / "results" / "week2" / "dataset.npz"
CACHE = RESULTS / "features_all.npz"        # 1,920개 시행 전체의 창별 특징 (교차검증용, git 제외)
SEED = 42

# 클래스 번호는 동작 순서(동작마다 30°, 60°가 이웃)로 고정한다.
# LabelEncoder 를 쓰면 알파벳순(30_extor, 30_fist, ...)이 되어 같은 동작의 두 각도가 혼동행렬에서 떨어진다.
CLASSES = W1.CLASSES                         # ['30_fist','60_fist','30_open','60_open','30_pro','60_pro','30_extor','60_extor']
SHORT = ["Fist30", "Fist60", "Open30", "Open60", "Pro30", "Pro60", "Sup30", "Sup60"]
ENC = {c: i for i, c in enumerate(CLASSES)}


def make_models():
    """슬라이드 9쪽과 같은 설정. SVM·RF는 기본값, XGBoost만 논문의 예비 튜닝 값."""
    return {
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr"),
        "RF": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
    }


def encode(y):
    return np.array([ENC[v] for v in y])


def trial_vote(pred, y, trial_ids):
    """한 시행의 19개 창 예측을 다수결로 묶어 시행 단위 정확도와 (정답, 예측) 쌍을 돌려준다."""
    yt, yp = [], []
    for t in dict.fromkeys(trial_ids):           # 등장 순서 유지
        i = trial_ids == t
        yt.append(y[i][0])
        yp.append(Counter(pred[i]).most_common(1)[0][0])
    yt, yp = np.array(yt), np.array(yp)
    return float((yt == yp).mean()), yt, yp


def error_types(cm):
    """오답을 (같은 동작·다른 각도) / (다른 동작·같은 각도) / (둘 다 틀림) 으로 나눈다."""
    same_g = diff_g_same_a = both = 0
    for i in range(8):
        for j in range(8):
            if i == j:
                continue
            if i // 2 == j // 2:
                same_g += cm[i, j]
            elif i % 2 == j % 2:
                diff_g_same_a += cm[i, j]
            else:
                both += cm[i, j]
    tot = same_g + diff_g_same_a + both
    return {"angle_only": int(same_g), "gesture_only": int(diff_g_same_a), "both": int(both),
            "total": int(tot)}


# ----------------------------------------------------------------------
# 단계 1~3, 6 — hold-out (2주차 dataset.npz 그대로)
# ----------------------------------------------------------------------
def holdout():
    d = np.load(DATASET)
    Xtr, Xte = d["Xtr"], d["Xte"]
    ytr, yte = encode(d["ytr"]), encode(d["yte"])
    out = {}
    for name, m in make_models().items():
        t0 = time.time(); m.fit(Xtr, ytr); train_t = time.time() - t0
        print(f"[단계 1] {name} 학습 완료 ({train_t:.1f}s)")
        t0 = time.time(); p = m.predict(Xte); batch_t = time.time() - t0
        t0 = time.time()
        for i in range(100):
            m.predict(Xte[i:i + 1])
        infer_ms = (time.time() - t0) * 10       # 1샘플 예측 100회 평균 (ms)

        cm = confusion_matrix(yte, p, labels=range(8))
        pr, rc, f1, _ = precision_recall_fscore_support(yte, p, labels=range(8), zero_division=0)
        tv_acc, tv_y, tv_p = trial_vote(p, yte, d["trial_te"])
        pos = np.arange(len(yte)) % 19                         # 시행 안에서 창 위치 (0~18)
        pos_acc = [float(np.mean(p[pos == k] == yte[pos == k])) for k in range(19)]
        g_acc = float(np.mean(yte // 2 == p // 2))           # 동작만 맞힌 비율
        a_acc = float(np.mean(yte % 2 == p % 2))             # 각도만 맞힌 비율
        out[name] = {
            "accuracy": accuracy_score(yte, p),
            "precision": precision_score(yte, p, average="macro", zero_division=0),
            "recall": recall_score(yte, p, average="macro"),
            "f1": f1_score(yte, p, average="macro"),
            "per_class": {"precision": pr.tolist(), "recall": rc.tolist(), "f1": f1.tolist()},
            "cm": cm.tolist(),
            "error_types": error_types(cm),
            "gesture_acc": g_acc, "angle_acc": a_acc,
            "trial_vote_acc": tv_acc, "window_pos_acc": pos_acc,
            "trial_vote_f1": f1_score(tv_y, tv_p, average="macro"),
            "trial_cm": confusion_matrix(tv_y, tv_p, labels=range(8)).tolist(),
            "train_time_s": train_t, "infer_ms_per_sample": infer_ms,
            "batch_predict_s": batch_t, "n_train": int(len(ytr)), "n_test": int(len(yte)),
        }
        print(f"[단계 2] {name}: acc {out[name]['accuracy']:.4f}  P {out[name]['precision']:.4f}  "
              f"R {out[name]['recall']:.4f}  F1 {out[name]['f1']:.4f}  | 시행 다수결 {tv_acc:.4f}")
        print(f"[단계 6] {name}: 학습 {train_t:.3f}s, 추론 {infer_ms:.3f} ms/sample")
    return out


# ----------------------------------------------------------------------
# 교차검증·LOSO 용 — 1,920개 시행 전체의 창별 특징 (표준화 전)
# ----------------------------------------------------------------------
def all_features():
    if CACHE.exists():
        c = np.load(CACHE)
        return c["X"], c["y"], c["tid"], c["sub"], c["group"]
    files = W1.step2_list_files()
    filt = {}
    for f in files:
        _, q = W2.load_filtered_trials(f)
        filt[W1.parse_name(f)] = q
    trials = W2.trial_table(files)
    X, y, tid, sub = W2.build(trials, filt)   # 2주차와 같은 필터·세그먼트·윈도우·특징
    group = np.array([t.split("#")[0] for t in tid])
    np.savez_compressed(CACHE, X=X, y=y, tid=tid, sub=sub, group=group)
    return X, y, tid, sub, group


def fit_eval(train_idx, test_idx, X, y, tid):
    """한 분할에서 scaler를 학습셋에만 fit 하고 세 모델을 평가한다."""
    sc = StandardScaler()
    Xa, Xb = sc.fit_transform(X[train_idx]), sc.transform(X[test_idx])
    ya, yb = y[train_idx], y[test_idx]
    res = {}
    for name, proto in make_models().items():
        m = clone(proto)
        t0 = time.time(); m.fit(Xa, ya); tt = time.time() - t0
        p = m.predict(Xb)
        res[name] = {"acc": accuracy_score(yb, p), "f1": f1_score(yb, p, average="macro"),
                     "trial_vote_acc": trial_vote(p, yb, tid[test_idx])[0], "train_time_s": tt}
    return res


def cross_validate(X, y, tid, group):
    """단계 4: 시행 단위 5-fold. 피험자×클래스(64그룹) 층화로 폴드마다 그룹당 6시행."""
    trials = np.array(list(dict.fromkeys(tid)))
    t_group = np.array([t.split("#")[0] for t in trials])
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = []
    for k, (tr_t, va_t) in enumerate(skf.split(trials, t_group)):
        tr_idx = np.flatnonzero(np.isin(tid, trials[tr_t]))
        va_idx = np.flatnonzero(np.isin(tid, trials[va_t]))
        assert not set(tid[tr_idx]) & set(tid[va_idx])         # 시행 겹침 없음
        r = fit_eval(tr_idx, va_idx, X, y, tid)
        folds.append(r)
        print(f"[단계 4] fold {k + 1}: " + "  ".join(f"{n} {r[n]['acc'] * 100:.2f}%" for n in r))
    return folds


def loso(X, y, tid, sub):
    folds = {}
    for s in W1.SUBJECTS:
        r = fit_eval(np.flatnonzero(sub != s), np.flatnonzero(sub == s), X, y, tid)
        folds[s] = r
        print(f"[LOSO] 피험자 {s}: " + "  ".join(
            f"{n} {r[n]['acc'] * 100:.1f}% (시행 {r[n]['trial_vote_acc'] * 100:.1f}%)" for n in r))
    return folds


def summarize(folds, names):
    out = {}
    for n in names:
        for key in ("acc", "f1", "trial_vote_acc"):
            v = np.array([f[n][key] for f in folds])
            out.setdefault(n, {})[key] = {"folds": v.tolist(), "mean": float(v.mean()),
                                         "std": float(v.std())}
    return out


def paired_tests(folds, names, key="acc"):
    """단계 5: 폴드별 값을 짝지어 대응표본 t-검정."""
    res = []
    for a, b in combinations(names, 2):
        va = np.array([f[a][key] for f in folds]); vb = np.array([f[b][key] for f in folds])
        t, p = ttest_rel(va, vb)
        res.append({"a": a, "b": b, "t": float(t), "p": float(p),
                    "mean_diff": float((va - vb).mean()),
                    "significant": bool(np.isfinite(p) and p < 0.05)})
        verdict = "검정 불가" if not np.isfinite(p) else ("유의함" if p < 0.05 else "유의하지 않음")
        print(f"[단계 5] {key} {a} vs {b}: t = {t:.3f}, p = {p:.4f}  ({verdict})")
    return res


def corrected_tests(cv, names, key="acc", k=5, test_train_ratio=0.25):
    """Nadeau & Bengio(2003) 보정 t-검정. 교차검증 폴드끼리 학습셋이 75% 겹쳐 생기는
    분산 과소추정을 (1/k + n_test/n_train) 항으로 보정한다 (보조 분석)."""
    from scipy.stats import t as tdist
    res = []
    for a, b in combinations(names, 2):
        d = np.array(cv[a][key]["folds"]) - np.array(cv[b][key]["folds"])
        t = d.mean() / np.sqrt((1 / k + test_train_ratio) * d.var(ddof=1))
        p = float(2 * tdist.sf(abs(t), k - 1))
        res.append({"a": a, "b": b, "t": float(t), "p": p, "significant": p < 0.05})
    return res


# ----------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------
def plot_cm(ax, cm, title, labels=SHORT, norm=True):
    cmn = cm / cm.sum(1, keepdims=True) if norm else cm
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1 if norm else None)
    ax.set_xticks(range(8), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(8), labels, fontsize=8)
    for g in (1.5, 3.5, 5.5):                       # 동작 경계선
        ax.axhline(g, color="white", lw=1.5); ax.axvline(g, color="white", lw=1.5)
    for i in range(8):
        for j in range(8):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.55 else "#1F2937")
    # 최대 오류 칸 표시
    off = cm.copy().astype(float); np.fill_diagonal(off, -1)
    i, j = np.unravel_index(off.argmax(), off.shape)
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#d62728", lw=2))
    ax.set_xlabel("예측 라벨", fontsize=9); ax.set_ylabel("실제 라벨", fontsize=9)
    ax.set_title(title, fontsize=10)


def make_figures(H, names):
    for n in names:
        cm = np.array(H[n]["cm"])
        fig, ax = plt.subplots(figsize=(6, 5.2))
        plot_cm(ax, cm, f"{n} — hold-out 혼동행렬 (창 단위, 클래스당 912)\n"
                        f"acc {H[n]['accuracy'] * 100:.2f}% · 칸 색 = 행 기준 비율 · 빨간 칸 = 최대 오류")
        fig.tight_layout(); fig.savefig(RESULTS / f"confusion_{n.lower()}.png", dpi=120); plt.close(fig)

    # 슬라이드 11쪽의 confusion8.png (RF)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    plot_cm(ax, np.array(H["RF"]["cm"]), "confusion8.png — RF, hold-out (창 단위)")
    fig.tight_layout(); fig.savefig(RESULTS / "confusion8.png", dpi=120); plt.close(fig)

    # 시행 단위 다수결 혼동행렬 (보조)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, n in zip(axes, names):
        cm = np.array(H[n]["trial_cm"])
        plot_cm(ax, cm, f"{n} — 시행 단위 다수결 (클래스당 48시행)\nacc {H[n]['trial_vote_acc'] * 100:.2f}%")
    fig.tight_layout(); fig.savefig(RESULTS / "confusion_trial_vote.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 창 위치별 정확도 (2초 세그먼트 안에서 어디가 어려운가)
    t = (W2.START + np.arange(19) * W2.HOP + W2.WIN / 2) / 1000
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for n, c, mk in zip(names, ["#2a78d6", "#eb6834", "#1baf7a"], ["o", "s", "^"]):
        ax.plot(t, np.array(H[n]["window_pos_acc"]) * 100, marker=mk, ms=4, lw=1.6, color=c, label=n)
    ax.axvspan(1.0, 2.5, color="#9CA3AF", alpha=0.15, lw=0, label="근활성 구간(1주차)")
    ax.set_xlabel("창 중심 시각 (s, mark 기준)", fontsize=9); ax.set_ylabel("정확도 (%)", fontsize=9)
    ax.set_ylim(30, 90); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False, ncol=4, loc="lower center")
    ax.set_title("시행 안 창 위치별 hold-out 정확도 (19개 위치)", fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS / "window_position_acc.png", dpi=120); plt.close(fig)

    # 오류 유형 막대
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = {"angle_only": ("#2a78d6", "같은 동작 · 각도만 틀림"),
            "gesture_only": ("#eb6834", "동작만 틀림 · 각도 맞음"),
            "both": ("#9CA3AF", "동작·각도 모두 틀림")}
    left = np.zeros(len(names))
    for k, (c, lab) in cols.items():
        v = np.array([H[n]["error_types"][k] / H[n]["error_types"]["total"] * 100 for n in names])
        ax.barh(names, v, left=left, color=c, label=lab, edgecolor="white", height=0.55)
        for i, x in enumerate(v):
            ax.text(left[i] + x / 2, i, f"{x:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += v
    ax.set_xlim(0, 100); ax.set_xlabel("오답 중 비율 (%)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("hold-out 오답의 구성", fontsize=10); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(RESULTS / "error_types.png", dpi=120); plt.close(fig)


def main():
    W1.set_korean_font()
    names = ["SVM", "RF", "XGB"]
    print("== hold-out (단계 1·2·3·6)")
    H = holdout()
    make_figures(H, names)
    out = {"holdout": H, "classes": CLASSES, "short": SHORT}

    if "--holdout-only" in sys.argv:                 # hold-out만 다시 계산해 기존 결과에 덮어쓴다
        prev = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
        prev["holdout"] = H
        prev["ttest_corrected_acc"] = corrected_tests(prev["cv"], names, "acc")
        prev["ttest_corrected_trial"] = corrected_tests(prev["cv"], names, "trial_vote_acc")
        (RESULTS / "results.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    print("\n== 5-fold 교차검증 (단계 4·5)")
    X, y_raw, tid, sub, group = all_features()
    y = encode(y_raw)
    folds = cross_validate(X, y, tid, group)
    out["cv"] = summarize(folds, names)
    out["cv_train_time"] = {n: float(np.mean([f[n]["train_time_s"] for f in folds])) for n in names}
    out["ttest_acc"] = paired_tests(folds, names, "acc")
    out["ttest_trial"] = paired_tests(folds, names, "trial_vote_acc")
    out["ttest_corrected_acc"] = corrected_tests(out["cv"], names, "acc")
    out["ttest_corrected_trial"] = corrected_tests(out["cv"], names, "trial_vote_acc")
    for n in names:
        c = out["cv"][n]
        print(f"{n}: {c['acc']['mean'] * 100:.2f} ± {c['acc']['std'] * 100:.2f} %  "
              f"(시행 다수결 {c['trial_vote_acc']['mean'] * 100:.2f} ± {c['trial_vote_acc']['std'] * 100:.2f} %)")

    if "--no-loso" not in sys.argv:
        print("\n== LOSO (보조)")
        lf = loso(X, y, tid, sub)
        out["loso"] = {"per_subject": lf, "summary": summarize(list(lf.values()), names)}

    (RESULTS / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n완료: results/week3/ 폴더 확인")


if __name__ == "__main__":
    main()
