#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA-KIRC Nükleer Entropi — 3. AŞAMA: İstatistik ve Figürler
=============================================================

Girdi : kirc_entropy_pipeline.py çıktı klasörü (case_level_results.csv + cases/*)
        + en güncel Excel (grade bilgisi buradan alınır; Excel'de düzeltme yapılırsa
          segmentasyon tekrarlanmadan istatistik güncellenir)
Çıktı : <results>/statistics/  → statistics_report.xlsx, methods_and_results.txt, fig01…fig13 (PNG)

Proje formundaki plan:
  • Low (G1+G2) vs High (G3+G4): Mann–Whitney U          → H1
  • Entropi ↔ grade (G1<G2<G3<G4): Spearman korelasyonu    → H1/H2
  • Lojistik regresyon, ROC/AUC                             → ayırt edicilik
Eklenenler (bilimsel geçerlilik için):
  • Etki büyüklüğü (rank-biserial r, Hodges–Lehmann farkı) + bootstrap %95 GA
  • Çoklu karşılaştırma düzeltmesi (birincil 4 test: Holm; ikincil: Benjamini–Hochberg)
  • Kruskal–Wallis (4 grade)
  • Çapraz doğrulamalı AUC (iyimserlik yanlılığına karşı)
  • Artımsal değer: entropi, klasik ölçülere (medyan alan, SD) EK bilgi veriyor mu? (LR testi)
  • Karıştırıcı kontrolü: çekirdek sayısı, bulanıklık, TS/BS kesit tipi gruplar arasında dengeli mi?
  • Bin sayısı duyarlılık analizi

Kullanım:
    python kirc_stats.py --results results --excel TCGA_KIRC_200_Case.xlsx
    python kirc_stats.py --demo --out demo_stats        # SİMÜLE veriyle örnek çıktı (sonuç DEĞİL)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261009
N_BOOT = 2000
SUCCESS = {"success", "success_with_warnings"}

PRIMARY = [("area_um2_entropy", "Alan entropisi"),
           ("perimeter_um_entropy", "Çevre entropisi"),
           ("circularity_entropy", "Dairesellik entropisi"),
           ("eccentricity_entropy", "Eksantrisite entropisi")]
# Duyarlılık analizleri (Holm/FDR uygulanmaz; birincil sonucun dayanıklılığını gösterir)
SENSITIVITY = [("area_um2_entropy_n500", "Alan entropisi (n=500 eşitlenmiş)"),
               ("perimeter_um_entropy_n500", "Çevre entropisi (n=500)"),
               ("circularity_entropy_n500", "Dairesellik entropisi (n=500)"),
               ("eccentricity_entropy_n500", "Eksantrisite entropisi (n=500)"),
               ("area_um2_entropy_nospindle", "Alan entropisi (iğsi çekirdekler hariç)"),
               ("perimeter_um_entropy_nospindle", "Çevre entropisi (iğsi hariç)"),
               ("circularity_entropy_nospindle", "Dairesellik entropisi (iğsi hariç)"),
               ("eccentricity_entropy_nospindle", "Eksantrisite entropisi (iğsi hariç)")]
SECONDARY = [("log2_area_entropy", "log₂(alan) entropisi (boyuttan bağımsız)"),
             ("patch_nuc_area_entropy_p90", "En heterojen bölge (tile alan entropisi p90)"),
             ("nuclear_entropy_index_exploratory", "Nükleer entropi indeksi (keşifsel)"),
             ("patch_nuc_area_entropy_std", "Tile alan entropisi SD (uzamsal heterojenlik)"),
             ("area_um2_median", "Medyan nükleer alan (µm²)"),
             ("area_um2_std", "Nükleer alan SD (µm²)"),
             ("area_um2_cv", "Nükleer alan CV"),
             ("circularity_std", "Dairesellik SD"),
             ("eccentricity_std", "Eksantrisite SD"),
             ("nuclei_per_mm2", "Çekirdek yoğunluğu (/mm²)"),
             ("grayscale_entropy", "Gri seviye piksel entropisi"),
             ("hematoxylin_entropy", "Hematoksilen piksel entropisi"),
             ("eosin_entropy", "Eozin piksel entropisi")]
CONFOUNDERS = [("n_nuclei", "Odaklı çekirdek sayısı"),
               ("spindle_fraction", "İğsi çekirdek oranı"),
               ("focus_blurry_tissue_fraction", "Bulanık doku oranı"),
               ("tissue_area_analyzed_mm2", "Analiz edilen doku (mm²)"),
               ("stain_cos_h_to_ref", "H boya vektörü benzerliği")]
LABEL = dict(PRIMARY + SECONDARY + SENSITIVITY + CONFOUNDERS)

# Renkler (dataviz doğrulayıcısından geçti: Low/High CVD ΔE 24.7; grade paleti tüm kontroller PASS)
C_LOW, C_HIGH = "#2a78d6", "#eb6834"
C_GRADE = {"G1": "#5e9de6", "G2": "#1f5fae", "G3": "#eb6834", "G4": "#a8421a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# =============================================================================
# Veri hazırlama
# =============================================================================
def load_analysis_set(results_dir: Path, excel: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Döner: (analiz seti, tüm vakalar, notlar). Analiz birimi = HASTA (1 satır / case_id)."""
    notes = []
    p = results_dir / "case_level_results.csv"
    if not p.exists():
        raise SystemExit(f"{p} bulunamadı. Önce pipeline'ı çalıştırın.")
    allc = pd.read_csv(p)
    # Excel'de seçilmemiş / elle dışlanmış satırlar kohort dışıdır (ör. yedekle değiştirilen 20x vakalar)
    skipped = allc["analysis_status"].isin(["skipped_not_selected", "skipped_manual_exclusion"])
    allc = allc[~skipped].copy()
    COHORT_NOTE["yedek"] = None
    yk = results_dir / "yedek_kaydi.json"
    if yk.exists():
        rec = json.loads(yk.read_text(encoding="utf-8"))
        COHORT_NOTE["yedek"] = rec
        dis = pd.DataFrame(rec.get("dislanan", []))
        if len(dis):
            notes.append(f"{len(dis)} vaka 20x tarandığı için (0.25 µm/px'e yetersiz) analiz edilemedi ve önceden belirlenmiş kurala göre "
                         f"aynı grade'den rastgele seçilen {len(rec.get('yedekler', []))} yedekle değiştirildi "
                         f"(kural: {rec.get('kural', '')}). Reddedilen aday: {len(rec.get('reddedilen_adaylar', []))}.")
    if excel:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kirc_entropy_pipeline import load_case_table
        ex, _ = load_case_table(excel)
        ex = ex[~ex["is_duplicate"]][["case_id_excel", "grade", "grade_group"]].rename(
            columns={"case_id_excel": "case_id", "grade": "grade_excel", "grade_group": "grade_group_excel"})
        allc = allc.merge(ex, on="case_id", how="left")
        mism = allc[allc["grade"].notna() & allc["grade_excel"].notna() & (allc["grade"] != allc["grade_excel"])]
        if len(mism):
            notes.append(f"{len(mism)} vakada grade Excel'de güncellenmiş; Excel değeri kullanıldı: {', '.join(mism['case_id'])}")
        allc["grade"] = allc["grade_excel"].fillna(allc["grade"])
        allc["grade_group"] = allc["grade_group_excel"].fillna(allc["grade_group"])
        allc = allc.drop(columns=["grade_excel", "grade_group_excel"])
    ok = allc[allc["analysis_status"].isin(SUCCESS)].copy()
    ok = ok[ok["grade"].isin(["G1", "G2", "G3", "G4"])]
    exp = {"G1": "Low", "G2": "Low", "G3": "High", "G4": "High"}
    gg = ok["grade_group"].astype(str).str.strip()
    if ok["grade_group"].isna().all() or gg.isin(["", "nan", "None"]).all():
        notes.append("Excel'de Grade Group sütunu boş; Low/High grubu Grade'den türetildi (G1–G2 = Low, G3–G4 = High).")
    else:
        bad = ok[ok["grade"].map(exp) != ok["grade_group"]]
        if len(bad):
            notes.append(f"{len(bad)} vakada Grade Group, Grade ile tutarsız; Grade'den yeniden türetildi.")
    ok["grade_group"] = ok["grade"].map(exp)
    dup = ok["case_id"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(f"Aynı hasta birden fazla satırda: {ok.loc[dup, 'case_id'].unique()} — bağımsızlık ihlali, düzeltin.")
    # Tüm vakalar AYNI yöntem ve parametrelerle mi analiz edildi? (karışık kohort = geçersiz karşılaştırma)
    for col, fix in (("seg_hash", "aynı --segmenter/--config-json ile `batch` çalıştırın"),
                     ("entropy_hash", "`kirc_entropy_pipeline.py entropy --out ...` çalıştırın")):
        if col in ok and ok[col].nunique() > 1:
            seg_counts = ok.groupby(col).size().to_dict()
            extra = ok.groupby(col)["segmenter"].first().to_dict() if "segmenter" in ok else {}
            raise SystemExit(f"Vakalar farklı parametrelerle analiz edilmiş ({col}: {seg_counts}; yöntem: {extra}). "
                             f"Karışık kohortla istatistik yapılmaz → {fix}.")
    if "tss_code" not in ok or ok["tss_code"].isna().all():
        ok["tss_code"] = ok["case_id"].str.split("-").str[1]
    ok["y_high"] = (ok["grade_group"] == "High").astype(int)
    ok["grade_num"] = ok["grade"].str[1].astype(int)
    if "slide_kind" in ok:
        ok["is_top_section"] = (ok["slide_kind"] == "frozen_top").astype(int)
    return ok.reset_index(drop=True), allc, notes


def simulate_cohort(n_per_grade=(13, 87, 50, 50), seed=SEED) -> pd.DataFrame:
    """SİMÜLE kohort — yalnızca kodun ve figürlerin nasıl göründüğünü göstermek için.
    Grade ile ılımlı bir ilişki varsayılmıştır; bu bir SONUÇ veya tahmin DEĞİLDİR."""
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for g, n in zip(("G1", "G2", "G3", "G4"), n_per_grade):
        gi = int(g[1])
        for _ in range(n):
            k += 1
            z = rng.normal()
            area_med = rng.normal(16 + 1.5 * (gi - 1) + 1.2 * z, 3.5)
            rows.append({
                "case_id": f"SIM-{k:04d}", "slide_id": f"SIM-{k:04d}-01A-01-BS1", "grade": g,
                "grade_group": "Low" if gi <= 2 else "High", "analysis_status": "success_with_warnings",
                "slide_kind": rng.choice(["frozen_bottom", "frozen_top"], p=[0.7, 0.3]),
                "area_um2_entropy": 2.7 + 0.18 * (gi - 1) + 0.25 * z + rng.normal(0, 0.15),
                "perimeter_um_entropy": 3.0 + 0.14 * (gi - 1) + 0.22 * z + rng.normal(0, 0.15),
                "circularity_entropy": 3.0 + 0.06 * (gi - 1) + rng.normal(0, 0.20),
                "eccentricity_entropy": 3.45 + 0.02 * (gi - 1) + rng.normal(0, 0.15),
                "area_um2_median": area_med, "area_um2_std": max(area_med * rng.normal(0.55, 0.08), 2),
                "circularity_std": rng.normal(0.11, 0.015), "eccentricity_std": rng.normal(0.16, 0.02),
                "n_nuclei": int(rng.lognormal(11.3, 0.5)), "focus_blurry_tissue_fraction": rng.beta(2, 6),
                "tissue_area_analyzed_mm2": rng.lognormal(3.6, 0.4), "nuclei_per_mm2": rng.normal(3200, 600),
                "grayscale_entropy": rng.normal(6.3, 0.15), "hematoxylin_entropy": rng.normal(4.3 + 0.05 * gi, 0.2),
                "eosin_entropy": rng.normal(4.6, 0.2), "patch_nuc_area_entropy_std": rng.normal(0.26 + 0.02 * gi, 0.05),
                "stain_cos_h_to_ref": rng.normal(0.99, 0.005)})
            r = rows[-1]
            sites = ["BP", "B0", "CZ", "A3", "AK", "B8", "CJ", "DV"]
            pw = np.array([3, 2, 2, 2, 1, 1, 1, 1], float) * (np.array([1, 1.6, 1, 0.7, 1, 1, 1.4, 1]) if gi >= 3 else 1)
            r["tss_code"] = rng.choice(sites, p=pw / pw.sum())   # hafif merkez–grade dengesizliği (gösterim için)
            r["log2_area_entropy"] = 3.6 + 0.05 * (gi - 1) + 0.25 * z + rng.normal(0, 0.15)
            r["patch_nuc_area_entropy_p90"] = r["area_um2_entropy"] + rng.normal(0.25, 0.08)
            r["spindle_fraction"] = rng.beta(2, 18)
            for f in ("area_um2", "perimeter_um", "circularity", "eccentricity"):
                r[f"{f}_entropy_nospindle"] = r[f"{f}_entropy"] + rng.normal(-0.03, 0.04)
            raw = rng.normal([0.32, 0.45, 0.30], [0.06, 0.07, 0.06])
            r.update({f"raw_mean_od_{c}": v for c, v in zip("rgb", raw)})
            r.update({f"norm_mean_od_{c}": v for c, v in zip("rgb", rng.normal([0.36, 0.52, 0.33], [0.012, 0.014, 0.012]))})
            r["stain_class"] = rng.choice(["HE_TYPICAL", "HE_ATYPICAL", "HE_FADED"], p=[0.85, 0.12, 0.03])
            r["seg_hash"], r["entropy_hash"], r["segmenter"] = "sim", "sim", "classical"
            r["area_um2_cv"] = r["area_um2_std"] / r["area_um2_median"]
            for f in ("area_um2", "perimeter_um", "circularity", "eccentricity"):
                r[f"{f}_entropy_n500"] = r[f"{f}_entropy"] - rng.normal(0.04, 0.01)
                for b, d in ((10, -1.55), (20, -0.85), (40, 0.0), (80, 0.9)):
                    r[f"{f}_entropy_b{b}"] = r[f"{f}_entropy"] + d * (1 if f in ("area_um2", "perimeter_um") else 0.8) + rng.normal(0, 0.03)
            r["nuclear_entropy_index_exploratory"] = np.mean([r["area_um2_entropy"] / math.log2(40), r["perimeter_um_entropy"] / math.log2(40),
                                                              r["circularity_entropy"] / math.log2(20), r["eccentricity_entropy"] / math.log2(20)])
    d = pd.DataFrame(rows)
    d["y_high"] = (d["grade_group"] == "High").astype(int)
    d["grade_num"] = d["grade"].str[1].astype(int)
    d["is_top_section"] = (d["slide_kind"] == "frozen_top").astype(int)
    return d


# =============================================================================
# İstatistik fonksiyonları
# =============================================================================
def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def bh_fdr(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    idx = np.argsort(p[ok])
    ranked = p[ok][idx] * m / (np.arange(m) + 1)
    adj = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    tmp = np.empty(m); tmp[idx] = adj
    out[ok] = tmp
    return out


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "—"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def mann_whitney(df, metric, rng):
    from scipy.stats import mannwhitneyu
    hi = df.loc[df.y_high == 1, metric].dropna().to_numpy()
    lo = df.loc[df.y_high == 0, metric].dropna().to_numpy()
    if len(hi) < 3 or len(lo) < 3:
        return None
    u, p = mannwhitneyu(hi, lo, alternative="two-sided")
    r = 2 * u / (len(hi) * len(lo)) - 1  # rank-biserial; + = High daha yüksek
    hl = float(np.median(np.subtract.outer(hi, lo)))
    from scipy.stats import rankdata
    rb, hlb = [], []
    n1, n0 = len(hi), len(lo)
    for _ in range(N_BOOT):
        h_, l_ = rng.choice(hi, n1), rng.choice(lo, n0)
        u_ = rankdata(np.concatenate([h_, l_]))[:n1].sum() - n1 * (n1 + 1) / 2  # = Mann–Whitney U (High)
        rb.append(2 * u_ / (n1 * n0) - 1)
        hlb.append(np.median(np.subtract.outer(h_, l_)))
    return {"n_low": len(lo), "n_high": len(hi),
            "median_low": float(np.median(lo)), "iqr_low": float(np.subtract(*np.percentile(lo, [75, 25]))),
            "median_high": float(np.median(hi)), "iqr_high": float(np.subtract(*np.percentile(hi, [75, 25]))),
            "U_high": float(u), "p": float(p), "rank_biserial_r": float(r),
            "r_ci_low": float(np.percentile(rb, 2.5)), "r_ci_high": float(np.percentile(rb, 97.5)),
            "hodges_lehmann_diff": hl, "hl_ci_low": float(np.percentile(hlb, 2.5)), "hl_ci_high": float(np.percentile(hlb, 97.5)),
            "auc_apparent": float((r + 1) / 2)}


def spearman_grade(df, metric, rng):
    from scipy.stats import spearmanr
    d = df[["grade_num", metric]].dropna()
    if len(d) < 5:
        return None
    rho, p = spearmanr(d["grade_num"], d[metric])
    bs = []
    a = d.to_numpy()
    for _ in range(N_BOOT):
        s = a[rng.integers(0, len(a), len(a))]
        bs.append(spearmanr(s[:, 0], s[:, 1])[0])
    return {"n": len(d), "rho": float(rho), "p": float(p),
            "rho_ci_low": float(np.nanpercentile(bs, 2.5)), "rho_ci_high": float(np.nanpercentile(bs, 97.5))}


def kruskal_grade(df, metric):
    from scipy.stats import kruskal
    groups = [df.loc[df.grade == g, metric].dropna() for g in ("G1", "G2", "G3", "G4")]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    h, p = kruskal(*groups)
    n = sum(len(g) for g in groups)
    eps2 = (h - len(groups) + 1) / (n - len(groups))  # epsilon-kare etki büyüklüğü
    return {"H": float(h), "p": float(p), "epsilon_sq": float(eps2), "k_groups": len(groups)}


def van_elteren(df, metric, strata="tss_code"):
    """Merkeze göre tabakalı Wilcoxon (van Elteren 1960): her merkezde High vs Low sıra toplamı,
    1/(n_s+1) ağırlıkla birleştirilir. Merkez–grade karıştırıcılığına karşı birincil duyarlılık testi."""
    from scipy.stats import norm, rankdata
    T = V = 0.0
    used = 0
    for _, g in df[[metric, "y_high", strata]].dropna().groupby(strata):
        n1, n0 = int((g.y_high == 1).sum()), int((g.y_high == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        n = n1 + n0
        r = rankdata(g[metric].to_numpy())
        w = 1.0 / (n + 1)
        W = r[g.y_high.to_numpy() == 1].sum()
        _, cnt = np.unique(g[metric].to_numpy(), return_counts=True)
        tie = (cnt ** 3 - cnt).sum() / (n * (n - 1)) if n > 1 else 0
        T += w * (W - n1 * (n + 1) / 2)
        V += w ** 2 * n1 * n0 / 12 * ((n + 1) - tie)
        used += 1
    if used == 0 or V <= 0:
        return None
    z = T / math.sqrt(V)
    return {"z": float(z), "p": float(2 * norm.sf(abs(z))), "strata_used": used}


def site_groups(df, min_n=10):
    """≥min_n vakalı merkezler ayrı, diğerleri 'Diğer' (model kararlılığı için)."""
    vc = df["tss_code"].value_counts()
    return df["tss_code"].where(df["tss_code"].isin(vc[vc >= min_n].index), "Diğer")


def logistic(df, cols, label, dummies=None):
    import statsmodels.api as sm
    dummies = list(dummies or [])
    d = df[["y_high", *cols, *dummies]].dropna()
    X = (d[cols] - d[cols].mean()) / d[cols].std(ddof=0)  # OR = 1 SD artış başına
    if dummies:
        X = pd.concat([X, d[dummies].astype(float)], axis=1)
    X = sm.add_constant(X)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(d["y_high"], X).fit(disp=0, maxiter=200)
    except Exception as exc:  # noqa: BLE001  (ör. tam ayrışma)
        return [{"model": label, "variable": c, "error": str(exc)} for c in cols], None
    ci = m.conf_int()
    rows = []
    for c in cols:
        rows.append({"model": label, "variable": c, "label": LABEL.get(c, c), "n": int(m.nobs),
                     "events_high": int(d["y_high"].sum()), "OR_per_SD": float(np.exp(m.params[c])),
                     "OR_ci_low": float(np.exp(ci.loc[c, 0])), "OR_ci_high": float(np.exp(ci.loc[c, 1])),
                     "p": float(m.pvalues[c]), "pseudo_R2_McFadden": float(m.prsquared), "AIC": float(m.aic)})
    return rows, m


def vif_table(df, cols):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = df[cols].dropna()
    X = sm.add_constant((d - d.mean()) / d.std(ddof=0))
    return pd.DataFrame({"variable": cols, "VIF": [variance_inflation_factor(X.values, i + 1) for i in range(len(cols))]})


def cv_auc(df, cols, n_splits=5, n_repeats=20, seed=SEED):
    """Tekrarlı tabakalı k-katlı CV; out-of-fold olasılıklarla AUC (iyimserlikten arındırılmış)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = df[["y_high", *cols]].dropna()
    X, y = d[cols].to_numpy(), d["y_high"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs, oof_all = [], np.zeros((n_repeats, len(y)))
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        model.fit(X[tr], y[tr])
        oof_all[i // n_splits, te] = model.predict_proba(X[te])[:, 1]
    for r in range(n_repeats):
        aucs.append(roc_auc_score(y, oof_all[r]))
    return {"cv_auc_mean": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs, ddof=1)),
            "cv_auc_p2_5": float(np.percentile(aucs, 2.5)), "cv_auc_p97_5": float(np.percentile(aucs, 97.5)),
            "n": int(len(y)), "folds": n_splits, "repeats": n_repeats}, y, oof_all.mean(axis=0)


def auc_bootstrap(y, s, rng):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    auc = roc_auc_score(y, s)
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    bs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])  # tabakalı
        bs.append(roc_auc_score(y[idx], s[idx]))
    return float(auc), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def youden(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    j = np.argmax(tpr - fpr)
    return {"cutoff": float(thr[j]), "sensitivity": float(tpr[j]), "specificity": float(1 - fpr[j])}


# =============================================================================
# Figürler
# =============================================================================
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False})
    return plt


COHORT_NOTE = {"text": "", "yedek": None}   # run() doldurur: tüm veri figürlerinin altına kohort bilgisi


def save(fig, out: Path, name: str, demo: bool):
    if COHORT_NOTE["text"] and not name.startswith(("fig01", "fig02")):
        # Not, eksen etiketleri dahil tüm çizimin ALTINA yerleştirilir (döndürülmüş etiketlerle çakışmasın)
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        y0 = bb.y0 / fig.get_figheight() - 0.012
        x0 = bb.x0 / fig.get_figwidth()
        fig.text(x0, y0, COHORT_NOTE["text"], ha="left", va="top", fontsize=8.5, color=INK2, transform=fig.transFigure)
    if demo:
        fig.text(0.5, 0.5, "SİMÜLE VERİ — GERÇEK SONUÇ DEĞİL", ha="center", va="center", rotation=25,
                 fontsize=34, color="#d0021b", alpha=0.18, fontweight="bold", transform=fig.transFigure, zorder=100)
    p = out / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def _box(ax, data, positions, colors, width=0.55):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.6}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.28); patch.set_edgecolor(c); patch.set_linewidth(1.4)
    rng = np.random.default_rng(1)
    for pos, d, c in zip(positions, data, colors):
        ax.scatter(pos + rng.uniform(-width / 3, width / 3, len(d)), d, s=12, color=c, alpha=0.75,
                   edgecolor="#fcfcfb", linewidth=0.5, zorder=3)


def fig_workflow(out, demo, info: dict | None = None):
    plt = setup_style()
    from matplotlib.patches import FancyBboxPatch
    info = info or {}
    mpp = info.get("mpp") or 0.5
    tile_um = info.get("tile_um") or 256
    tile_px = int(round(tile_um / mpp))
    seg = str(info.get("segmenter") or "classical")
    max_t = info.get("max_tiles")
    seg_box = (("StarDist", "derin öğrenme ile\nçekirdek\nsegmentasyonu") if seg.startswith("stardist")
               else ("Otsu +\nwatershed", "çekirdek\nsegmentasyonu"))
    stages = [
        ("1. AŞAMA — ağır (vaka başına bir kez)", "#e8f0fb", C_LOW, [
            ("Excel + GDC\nID doğrulama", "case / sample / slide,\ntekrar & MD5"),
            ("WSI tile\nokuma", f"OpenSlide, {tile_px} px\n({tile_um:.0f} µm), {mpp:.2f} µm/px"
                                 + (f"\n≤{int(max_t)} rastgele tile" if max_t else "")),
            ("Kalite\nkontrol", "doku maskesi, kalem,\nodak blokları, boya"),
            ("Macenko +\nH kanalı", "renk ayrıştırma,\nnormalizasyon"),
            seg_box,
            ("Morfometri", "alan, çevre (Crofton),\ndairesellik, eksantr.")]),
        ("2. AŞAMA — hafif", "#fdeee6", C_HIGH, [
            ("Shannon\nentropisi", "sabit bin, log₂,\nn=500, duyarlılık")]),
        ("3. AŞAMA — istatistik", "#eeeeea", INK2, [
            ("Mann–Whitney\nSpearman, KW", "etki büyüklüğü,\nHolm / FDR"),
            ("Lojistik,\nROC, CV-AUC", "artımsal değer,\nkarıştırıcılar")])]
    bw, gap, pad, sgap = 1.75, 0.3, 0.2, 0.35
    total = sum(len(b) * (bw + gap) - gap + 2 * pad for _, _, _, b in stages) + sgap * (len(stages) - 1)
    fig, ax = plt.subplots(figsize=(total * 0.95, 5.0))
    ax.axis("off"); ax.grid(False); ax.set_xlim(0, total); ax.set_ylim(0, 5.0)
    x = 0
    for si, (title, bg, ec, boxes) in enumerate(stages):
        w_stage = len(boxes) * (bw + gap) - gap + 2 * pad
        ax.add_patch(FancyBboxPatch((x, 0.9), w_stage, 3.5, boxstyle="round,pad=0,rounding_size=0.15", fc=bg, ec="none"))
        ax.text(x + w_stage / 2, 4.05, title, ha="center", fontsize=10, fontweight="bold", color=INK)
        for i, (h, sub) in enumerate(boxes):
            bx = x + pad + i * (bw + gap)
            ax.add_patch(FancyBboxPatch((bx, 1.3), bw, 2.3, boxstyle="round,pad=0,rounding_size=0.1", fc="white", ec=ec, lw=1.4))
            ax.text(bx + bw / 2, 2.95, h, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
            ax.text(bx + bw / 2, 1.85, sub, ha="center", va="center", fontsize=7.8, color=INK2)
            last = (i == len(boxes) - 1)
            if not (last and si == len(stages) - 1):
                x_end = bx + bw + (gap if not last else pad + sgap + pad)
                ax.annotate("", xy=(x_end, 2.45), xytext=(bx + bw, 2.45), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 1.2})
        x += w_stage + sgap
    ax.text(0.05, 0.45, "Ara çıktılar: nuclei_features.csv.gz (çekirdek başına), tile_metrics.csv, QC görselleri  →  "
            "entropi parametreleri değişirse yalnız 2. aşama çalışır, segmentasyon tekrarlanmaz.", fontsize=9, color=INK2)
    ax.text(0.05, 4.75, "Şekil 1. Analiz iş akışı", fontsize=12, fontweight="bold", color=INK)
    return save(fig, out, "fig01_workflow.png", demo)


EXCL_REASON = [("RESOLUTION_INSUFFICIENT", "20x tarama (yetersiz çözünürlük)"),
               ("STAIN_", "H&E dışı / uygunsuz boyama"), ("NO_OR_LOW_TISSUE", "yetersiz doku"),
               ("OUT_OF_FOCUS", "odak dışı"), ("INSUFFICIENT_NUCLEI", "yetersiz çekirdek"),
               ("MPP_MISSING", "ölçek bilgisi yok"), ("TOO_FEW_TILES", "yetersiz doku (tile)")]


def exclusion_table(allc: pd.DataFrame) -> pd.DataFrame:
    """Analiz dışı kalan vakalar: okunur neden + grade + merkez."""
    rows = []
    for r in allc[~allc["analysis_status"].isin(SUCCESS)].to_dict("records"):
        flags = str(r.get("qc_flags") or "")
        reason = next((t for k, t in EXCL_REASON if k in flags), None)
        if reason is None:
            st = str(r.get("analysis_status"))
            reason = {"download_failed": "indirilemedi", "missing_file": "dosya yok", "failed_error": "işlem hatası"}.get(st, st)
        rows.append({"case_id": r.get("case_id"), "grade": r.get("grade"), "tss_code": r["tss_code"] if isinstance(r.get("tss_code"), str) and r["tss_code"] else str(r.get("case_id", ""))[5:7],
                     "neden": reason, "durum": r.get("analysis_status")})
    return pd.DataFrame(rows, columns=["case_id", "grade", "tss_code", "neden", "durum"])


def fig_flow(allc, ana, out, demo):
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.axis("off"); ax.grid(False); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    total = len(allc)
    ex = exclusion_table(allc)
    no_grade = int((allc["analysis_status"].isin(SUCCESS) & ~allc["grade"].isin(["G1", "G2", "G3", "G4"])).sum())
    gx = allc["grade"].value_counts()

    from matplotlib.patches import FancyBboxPatch

    def box(x, y, w, h, txt, ec=INK2, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)

    yed = COHORT_NOTE.get("yedek") or {}
    dis, yd = pd.DataFrame(yed.get("dislanan", [])), pd.DataFrame(yed.get("yedekler", []))
    box(0.9, 8.3, 5.6, 1.3, f"TCGA-KIRC kohortu: {total} hasta\n(her hasta 1 FFPE tanısal slayt)\n"
                          f"G1 {gx.get('G1', 0)} · G2 {gx.get('G2', 0)} · G3 {gx.get('G3', 0)} · G4 {gx.get('G4', 0)}")
    if len(dis):
        gtxt = lambda d: ", ".join(f"{k} {v}" for k, v in d["grade"].value_counts().sort_index().items())  # noqa: E731
        stxt = ", ".join(f"{k} {v}" for k, v in dis["case_id"].str[5:7].value_counts().items())
        box(6.8, 8.15, 3.15, 1.6, f"Başlangıç seçiminde 20x taranmış: {len(dis)}\n({gtxt(dis)}; merkez {stxt})\n"
                                  f"→ aynı grade'den rastgele yedek: {len(yd)}\n({gtxt(yd) if len(yd) else '-'})", ec=INK2, fs=8.2)
        ax.annotate("", xy=(6.5, 8.95), xytext=(6.75, 8.95), arrowprops={"arrowstyle": "-|>", "color": INK2})
    box(0.9, 4.5, 5.6, 1.3, f"Analiz seti (1 hasta = 1 gözlem)\nn = {len(ana)} / {total}")
    lines = []
    for reason, g in ex.groupby("neden", sort=False):
        gr = ", ".join(f"{k} {v}" for k, v in g["grade"].value_counts().sort_index().items())
        st = ", ".join(f"{k} {v}" for k, v in g["tss_code"].value_counts().items())
        lines.append(f"{reason}\nn = {len(g)} · grade: {gr}\nmerkez: {st}")
    if no_grade:
        lines.append(f"grade bilinmiyor: n = {no_grade}")
    ax.annotate("", xy=(3.7, 5.85), xytext=(3.7, 8.25), arrowprops={"arrowstyle": "-|>", "color": INK2})
    if len(ex) + no_grade:
        ex_txt = f"Analiz dışı: n = {len(ex) + no_grade}\n" + "\n".join(lines)
        box(6.8, 5.6, 3.15, 2.1, ex_txt, ec=C_HIGH, fs=8.2)
        ax.annotate("", xy=(6.75, 6.65), xytext=(3.7, 6.65), arrowprops={"arrowstyle": "-|>", "color": INK2})
    nl, nh = int((ana.y_high == 0).sum()), int((ana.y_high == 1).sum())
    gc = ana["grade"].value_counts()
    box(0.3, 1.2, 4, 1.6, f"Low grade (G1+G2)\nn = {nl}  (G1 {gc.get('G1', 0)}, G2 {gc.get('G2', 0)})", ec=C_LOW)
    box(5.1, 1.2, 4, 1.6, f"High grade (G3+G4)\nn = {nh}  (G3 {gc.get('G3', 0)}, G4 {gc.get('G4', 0)})", ec=C_HIGH)
    ax.annotate("", xy=(2.3, 2.85), xytext=(3.0, 4.45), arrowprops={"arrowstyle": "-|>", "color": INK2})
    ax.annotate("", xy=(7.1, 2.85), xytext=(4.4, 4.45), arrowprops={"arrowstyle": "-|>", "color": INK2})
    ax.set_title("Şekil 2. Kohort akış şeması", loc="left")
    return save(fig, out, "fig02_cohort_flow.png", demo)


def fig_box_group(ana, mw, out, demo):
    plt = setup_style()
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6))
    for ax, (m, lab) in zip(axs, PRIMARY):
        lo, hi = ana.loc[ana.y_high == 0, m].dropna(), ana.loc[ana.y_high == 1, m].dropna()
        _box(ax, [lo, hi], [0, 1], [C_LOW, C_HIGH])
        ax.set_xticks([0, 1], [f"Low\n(n={len(lo)})", f"High\n(n={len(hi)})"])
        r = mw.get(m)
        sub = f"{fmt_p(r['p_holm'])} (Holm) · r={r['rank_biserial_r']:+.2f}" if r else ""
        ax.set_title(f"{lab}\n{sub}", fontsize=10)
        ax.set_ylabel("bit" if ax is axs[0] else "")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Şekil 3. Birincil ölçütler: Low vs High grade (Mann–Whitney U, rank-biserial r)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig03_boxplot_low_high.png", demo)


def fig_box_grade(ana, sp, out, demo):
    plt = setup_style()
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6))
    gs = ["G1", "G2", "G3", "G4"]
    for ax, (m, lab) in zip(axs, PRIMARY):
        data = [ana.loc[ana.grade == g, m].dropna() for g in gs]
        _box(ax, data, [0, 1, 2, 3], [C_GRADE[g] for g in gs])
        ax.set_xticks(range(4), [f"{g}\n(n={len(d)})" for g, d in zip(gs, data)])
        meds = [np.median(d) if len(d) else np.nan for d in data]
        ax.plot(range(4), meds, color=INK2, lw=1, ls="--", zorder=4)
        r = sp.get(m)
        ax.set_title(f"{lab}\nSpearman ρ={r['rho']:+.2f} [{r['rho_ci_low']:+.2f}, {r['rho_ci_high']:+.2f}], {fmt_p(r['p'])}" if r else lab, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Şekil 4. Entropi ve grade (G1→G4) ilişkisi; kesikli çizgi = medyan", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig04_boxplot_by_grade.png", demo)


def fig_hist(ana, out, demo):
    plt = setup_style()
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (m, lab) in zip(axs, PRIMARY):
        v = ana[m].dropna()
        bins = np.linspace(v.min(), v.max(), 22)
        for y, c, n in ((0, C_LOW, "Low"), (1, C_HIGH, "High")):
            ax.hist(ana.loc[ana.y_high == y, m].dropna(), bins=bins, histtype="stepfilled", alpha=0.25, color=c)
            ax.hist(ana.loc[ana.y_high == y, m].dropna(), bins=bins, histtype="step", lw=2, color=c, label=n)
        ax.set_title(lab); ax.set_xlabel("bit")
    axs[0].set_ylabel("vaka sayısı"); axs[0].legend()
    fig.suptitle("Şekil 5. Vaka düzeyi entropi dağılımları", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig05_histograms.png", demo)


def fig_roc(ana, roc_rows, cv_res, out, demo):
    from sklearn.metrics import roc_curve
    plt = setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    styles = [(C_HIGH, "-"), (C_LOW, "-"), (INK2, "--"), (INK2, ":")]
    for (m, lab), (col, s) in zip(PRIMARY, styles):
        d = ana[["y_high", m]].dropna()
        fpr, tpr, _ = roc_curve(d.y_high, d[m])
        r = roc_rows[m]
        axs[0].plot(fpr, tpr, lw=2, ls=s, color=col, label=f"{lab}: AUC {r['auc']:.2f} [{r['auc_ci_low']:.2f}–{r['auc_ci_high']:.2f}]")
    axs[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    axs[0].set(xlabel="1 − özgüllük", ylabel="duyarlılık", title="Tek değişkenli ROC (görünen AUC, bootstrap %95 GA)")
    axs[0].legend(loc="lower right", fontsize=8)
    for name, c, s in (("Klasik (medyan alan + alan SD)", INK2, "--"), ("Entropi (4 ölçüt)", C_LOW, "-"),
                       ("Klasik + entropi", C_HIGH, "-")):
        if name in cv_res:
            res, y, p = cv_res[name]
            fpr, tpr, _ = roc_curve(y, p)
            axs[1].plot(fpr, tpr, lw=2, color=c, ls=s, label=f"{name}: CV-AUC {res['cv_auc_mean']:.2f} ± {res['cv_auc_sd']:.2f}")
    axs[1].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    axs[1].set(xlabel="1 − özgüllük", ylabel="duyarlılık", title="Çok değişkenli lojistik model (5-katlı × 20 CV, out-of-fold)")
    axs[1].legend(loc="lower right", fontsize=8)
    for a in axs:
        a.set_aspect("equal"); a.set_xlim(0, 1); a.set_ylim(0, 1.01)
    fig.suptitle("Şekil 6. ROC eğrileri — High grade ayırt ediciliği", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig06_roc.png", demo)


def fig_forest(uni, multi, out, demo):
    plt = setup_style()
    rows = [("Tek değişkenli", r) for r in uni if "OR_per_SD" in r] + [("Düzeltilmiş", r) for r in multi if "OR_per_SD" in r]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6))
    ys = np.arange(len(rows))[::-1]
    for y, (mdl, r) in zip(ys, rows):
        c = C_LOW if mdl == "Tek değişkenli" else C_HIGH
        ax.plot([r["OR_ci_low"], r["OR_ci_high"]], [y, y], color=c, lw=2)
        ax.scatter(r["OR_per_SD"], y, color=c, s=46, zorder=3, edgecolor="#fcfcfb", lw=1.5)
        ax.text(1.02, y, f"{r['OR_per_SD']:.2f} [{r['OR_ci_low']:.2f}–{r['OR_ci_high']:.2f}]  {fmt_p(r['p'])}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_yticks(ys, [f"{mdl}: {r['label']}" for mdl, r in rows], fontsize=8.5)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    lo_ = min(r["OR_ci_low"] for _, r in rows); hi_ = max(r["OR_ci_high"] for _, r in rows)
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8) if lo_ * 0.9 <= t <= hi_ * 1.1] or [1]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Odds ratio (1 SD artış başına, log ölçek) — High grade")
    ax.grid(axis="y", visible=False)
    ax.set_title("Şekil 7. Lojistik regresyon — her entropi ayrı model; düzeltilmiş = + log10(çekirdek sayısı), "
                 "analiz edilen doku, bulanıklık", loc="left", fontsize=10)
    return save(fig, out, "fig07_forest_logistic.png", demo)


def fig_corr(ana, out, demo):
    plt = setup_style()
    cols = [c for c, _ in PRIMARY + SECONDARY + CONFOUNDERS[:3] if c in ana and ana[c].notna().sum() > 5 and not c.endswith("_n500")]
    cm = ana[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [LABEL.get(c, c) for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), [LABEL.get(c, c) for c in cols], fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman ρ")
    ax.set_title("Şekil 8. Ölçütler arası Spearman korelasyonu (entropi klasik ölçülerden ne kadar bağımsız?)", loc="left", fontsize=10)
    return save(fig, out, "fig08_correlation_heatmap.png", demo)


def fig_bins(sens, out, demo):
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for (feat, lab), s in zip(PRIMARY, ["-", "--", "-.", ":"]):
        d = sens[sens.feature == feat].sort_values("bins")
        if d.empty:
            continue
        ax.errorbar(d["bins"], d["rank_biserial_r"], yerr=[d["rank_biserial_r"] - d["r_ci_low"], d["r_ci_high"] - d["rank_biserial_r"]],
                    lw=2, ls=s, marker="o", ms=6, capsize=3, color=C_HIGH if s == "-" else INK, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks([10, 20, 40, 80], ["10", "20", "40", "80"])
    ax.set_xlabel("histogram bin sayısı"); ax.set_ylabel("rank-biserial r (High − Low)")
    ax.legend(fontsize=8)
    ax.set_title("Şekil 9. Bin sayısı duyarlılığı — sonuç bin seçimine dayanıklı mı?", loc="left")
    return save(fig, out, "fig09_bin_sensitivity.png", demo)


def fig_confounders(ana, conf, out, demo):
    plt = setup_style()
    cols = [(c, l) for c, l in CONFOUNDERS[:3] if c in ana and ana[c].notna().any()]
    fig, axs = plt.subplots(1, len(cols) + 1, figsize=(4 * (len(cols) + 1), 4.2))
    for ax, (c, lab) in zip(axs, cols):
        lo, hi = ana.loc[ana.y_high == 0, c].dropna(), ana.loc[ana.y_high == 1, c].dropna()
        _box(ax, [lo, hi], [0, 1], [C_LOW, C_HIGH])
        ax.set_xticks([0, 1], ["Low", "High"])
        r = conf.get(c)
        ax.set_title(f"{lab}\n{fmt_p(r['p']) if r else ''}", fontsize=9.5)
        if c == "n_nuclei":
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    ax = axs[-1]
    if "slide_kind" in ana:
        ct = pd.crosstab(ana["grade_group"], ana["slide_kind"]).reindex(["Low", "High"])
        bottom = np.zeros(len(ct))
        for k, col in enumerate(ct.columns):
            ax.bar([0, 1], ct[col].values, bottom=bottom, color=[C_LOW, C_HIGH], alpha=0.35 + 0.4 * k, edgecolor="#fcfcfb", lw=2, width=0.6)
            for x, (b, v) in enumerate(zip(bottom, ct[col].values)):
                if v:
                    ax.text(x, b + v / 2, f"{col.replace('frozen_', '')}\n{v}", ha="center", va="center", fontsize=8, color=INK)
            bottom += ct[col].values
        ax.set_xticks([0, 1], ["Low", "High"])
        r = conf.get("slide_kind")
        kinds = list(ct.columns)
        ttl = ("Kesit tipi: tümü FFPE tanısal (DX)" if kinds == ["ffpe_diagnostic"] else "Kesit tipi")
        ax.set_title(f"{ttl}\n{('Fisher ' + fmt_p(r['p'])) if r else ''}", fontsize=9.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Şekil 10. Olası karıştırıcılar — gruplar teknik açıdan dengeli mi?", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig10_confounders.png", demo)


def fig_heatmap_gallery(ana, results_dir, out, demo, cfg_range=(1.5, 4.0)):
    """Temsilî vakaların tile düzeyi entropi ısı haritaları (sabit renk ölçeği)."""
    plt = setup_style()
    picks = []
    for grp in ("Low", "High"):
        d = ana[ana.grade_group == grp].dropna(subset=["area_um2_entropy"]).sort_values("area_um2_entropy")
        if len(d):
            idx = sorted(set([0, len(d) // 2, len(d) - 1]))
            picks += [(grp, d.iloc[i]) for i in idx]
    maps = []
    for grp, r in picks:
        tm = results_dir / "cases" / str(r["slide_id"]) / "tile_metrics.csv"
        if tm.exists():
            t = pd.read_csv(tm)
            t = t[t["selected"].astype(bool)]
            hm = np.full((int(t.grid_y.max()) + 1, int(t.grid_x.max()) + 1), np.nan)
            hm[t.grid_y.astype(int), t.grid_x.astype(int)] = t["nuc_area_entropy"]
            maps.append((grp, r, hm))
    if not maps:
        return None
    rows_g = [g for g in ("Low", "High") if any(m[0] == g for m in maps)]
    ncol = max(sum(m[0] == g for m in maps) for g in rows_g)
    fig, axs = plt.subplots(len(rows_g), ncol, figsize=(5.2 * ncol, 4.6 * len(rows_g)), squeeze=False)
    for a in axs.ravel():
        a.axis("off")
    im = None
    for ri, g in enumerate(rows_g):
        for ci, (grp, r, hm) in enumerate([m for m in maps if m[0] == g]):
            ax = axs[ri, ci]
            im = ax.imshow(hm, cmap="viridis", vmin=cfg_range[0], vmax=cfg_range[1], interpolation="nearest")
            ax.set_title(f"{r['case_id']} · {r['grade']} ({grp})\nalan entropisi {r['area_um2_entropy']:.2f} bit", fontsize=10,
                         color=C_LOW if grp == "Low" else C_HIGH)
    fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.025, label="tile nükleer alan entropisi (bit, sabit ölçek)")
    fig.suptitle("Şekil 11. Entropi ısı haritaları — temsilî vakalar (en düşük / medyan / en yüksek)", x=0.01, ha="left", fontweight="bold")
    return save(fig, out, "fig11_entropy_heatmap_gallery.png", demo)


def fig_pooled_distributions(ana, results_dir, out, demo, bins_cfg=None):
    """Gruplara göre çekirdek özelliği dağılımları. Her vaka EŞİT ağırlıkta: önce vaka içi normalize
    histogram, sonra grup ortalaması (çok çekirdekli vakaların baskın olmasını önler)."""
    plt = setup_style()
    bins_cfg = bins_cfg or {"area_um2": [0, 200, 40], "perimeter_um": [0, 80, 40], "circularity": [0, 1, 20], "eccentricity": [0, 1, 20]}
    acc = {g: {f: [] for f in bins_cfg} for g in ("Low", "High")}
    for _, r in ana.iterrows():
        p = results_dir / "cases" / str(r["slide_id"]) / "nuclei_features.csv.gz"
        if not p.exists():
            continue
        n = pd.read_csv(p, usecols=lambda c: c in (*bins_cfg, "in_focus"))
        if "in_focus" in n:
            n = n[n["in_focus"].astype(bool)]
        for f, (lo, hi, nb) in bins_cfg.items():
            h, _ = np.histogram(np.clip(n[f], lo, hi), bins=np.linspace(lo, hi, int(nb) + 1))
            acc[r["grade_group"]][f].append(h / max(h.sum(), 1))
    if not any(acc[g][f] for g in acc for f in bins_cfg):
        return None
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (f, (lo, hi, nb)) in zip(axs, bins_cfg.items()):
        edges = np.linspace(lo, hi, int(nb) + 1)
        for g, c in (("Low", C_LOW), ("High", C_HIGH)):
            if acc[g][f]:
                m = np.mean(acc[g][f], axis=0)
                ax.stairs(m, edges, color=c, lw=2, label=f"{g} (n={len(acc[g][f])} vaka)")
                ax.stairs(m, edges, color=c, alpha=0.15, fill=True)
        ax.set_title(f); ax.set_xlabel(f)
    axs[0].set_ylabel("ortalama oran"); axs[0].legend(fontsize=8)
    fig.suptitle("Şekil 12. Gruplara göre çekirdek özelliği dağılımları (vakalar eşit ağırlıklı)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig12_pooled_feature_distributions.png", demo)


def fig_incremental(inc, out, demo):
    plt = setup_style()
    if not inc:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(inc.keys())
    means = [inc[n]["cv_auc_mean"] for n in names]
    sds = [inc[n]["cv_auc_sd"] for n in names]
    cols = [INK2, C_LOW, C_HIGH, "#2f8f6f"][: len(names)]
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, means, xerr=sds, color=cols, alpha=0.8, height=0.55, capsize=4, edgecolor="#fcfcfb", lw=2)
    for y, m in zip(ys, means):
        ax.text(m + 0.02, y, f"{m:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, names); ax.set_xlim(0.4, 1.0); ax.axvline(0.5, color=INK2, ls=":", lw=1)
    ax.set_xlabel("çapraz doğrulamalı AUC (ort. ± SD, 20 tekrar)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Şekil 13. Artımsal değer — entropi klasik ölçülere bilgi ekliyor mu?", loc="left")
    return save(fig, out, "fig13_incremental_value.png", demo)


def fig_site(site_tab, site_test, out, demo):
    plt = setup_style()
    t = site_tab.copy()
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(t) + 1.8))
    ys = np.arange(len(t))[::-1]
    left = np.zeros(len(t))
    for g in ("G1", "G2", "G3", "G4"):
        ax.barh(ys, t[g].values, left=left, color=C_GRADE[g], edgecolor="#fcfcfb", lw=1.5, height=0.7, label=g)
        left += t[g].values
    for y, (_, r) in zip(ys, t.iterrows()):
        ax.text(r["n"] + 0.3, y, f"n={int(r['n'])} · High %{r['High_%']:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys, t.index, fontsize=8); ax.set_xlabel("vaka sayısı"); ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    sub = f" — χ² {fmt_p(site_test['p'])}, Cramér V={site_test['cramers_v']:.2f}" if site_test else ""
    ax.set_title(f"Şekil 14. Doku kaynak merkezi (TSS) × grade{sub}", loc="left")
    return save(fig, out, "fig14_site_by_grade.png", demo)


def fig_normalization(ana, norm_eff, out, demo):
    plt = setup_style()
    inten = [r for r in norm_eff if r.get("tip") == "boya şiddeti"]
    angs = [r for r in norm_eff if r.get("tip") == "boya rengi"]
    if not inten and not angs:
        return None
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    if "stain_conc_scale" in ana:
        sc = np.array([json.loads(v) for v in ana["stain_conc_scale"].dropna()], float)
        rel = (1 / sc) / np.nanmedian(1 / sc, axis=0)
        axs[0].hist(rel[:, 0], bins=25, color=INK2, alpha=0.8, label="Hematoksilen — önce")
        axs[0].hist(rel[:, 1], bins=25, color=C_HIGH, alpha=0.5, label="Eozin — önce")
        axs[0].axvline(1, color=C_LOW, lw=2.5, label="normalizasyon sonrası (tüm slaytlar)")
        axs[0].set(xlabel="göreli boya şiddeti (99. persentil, medyan = 1)", ylabel="slayt sayısı",
                   title="Slaytlar arası boya şiddeti farkı")
        axs[0].legend(fontsize=8)
    if inten:
        x = np.arange(len(inten))
        axs[1].bar(x - 0.2, [r["CV_önce"] for r in inten], 0.4, color=INK2, label="önce", edgecolor="#fcfcfb", lw=2)
        axs[1].bar(x + 0.2, [r["CV_sonra"] for r in inten], 0.4, color=C_LOW, label="sonra", edgecolor="#fcfcfb", lw=2)
        for xi, r in zip(x, inten):
            axs[1].text(xi - 0.2, r["CV_önce"], f"{r['CV_önce']:.2f}", ha="center", va="bottom", fontsize=8.5)
            axs[1].text(xi + 0.2, 0, "0", ha="center", va="bottom", fontsize=8.5)
        axs[1].set_xticks(x, [r["ölçüt"].split(" (")[0] for r in inten])
        axs[1].set(ylabel="slaytlar arası CV", title="Boya şiddeti varyasyonu (99. persentil)")
        axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    sub = "; ".join(f"{r['ölçüt']}: önce {r['ort_önce']:.1f}° ± {r['SD_önce']:.1f}° → sonra 0°" for r in angs)
    fig.suptitle("Şekil 15. Renk normalizasyonu — boya şiddeti ve rengi referansa eşitlendi" + (f"\n{sub}" if sub else ""),
                 x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    return save(fig, out, "fig15_stain_normalization.png", demo)


# =============================================================================
# Ana akış
# =============================================================================
def run(ana: pd.DataFrame, allc: pd.DataFrame, results_dir: Path | None, out: Path, demo: bool, notes: list[str]):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _ex = exclusion_table(allc)
    _why = "; ".join(f"{r} {n}" for r, n in _ex["neden"].value_counts().items())
    _y = COHORT_NOTE.get("yedek") or {}
    COHORT_NOTE["text"] = (f"Kohort {len(allc)} hasta; analiz seti {len(ana)} hasta (Low {int((ana.y_high == 0).sum())}, "
                           f"High {int((ana.y_high == 1).sum())})" + (f". Analiz dışı: {_why}" if len(_ex) else "")
                           + (f". 20x taranmış {len(_y.get('dislanan', []))} vaka aynı grade'den rastgele yedekle değiştirildi"
                              if _y.get("dislanan") else "") + ". Ayrıntı: Şekil 2.")
    present = lambda lst: [(c, l) for c, l in lst if c in ana and ana[c].notna().sum() >= 6]  # noqa: E731
    prim, sec, sensl = present(PRIMARY), present(SECONDARY), present(SENSITIVITY)
    if len(ana) < 10 or ana["y_high"].nunique() < 2:
        msg = (f"Analiz setinde {len(ana)} vaka var (Low={int((ana.y_high == 0).sum())}, High={int((ana.y_high == 1).sum())}). "
               "Grup karşılaştırması için her grupta en az 5, anlamlı güç için tüm kohort gerekir. "
               "Batch tamamlanınca tekrar çalıştırın; şimdilik --demo ile çıktı biçimini görebilirsiniz.")
        print(msg)
        (out / "README_istatistik_bekliyor.txt").write_text(msg, encoding="utf-8")
        setup_style(); fig_workflow(out, demo); fig_flow(allc, ana, out, demo)
        return None

    # 1) Tanımlayıcı
    desc_rows = []
    for c, lab in prim + sec + present(CONFOUNDERS):
        for key, sub in [("Tümü", ana)] + [(g, ana[ana.grade_group == g]) for g in ("Low", "High")] + [(g, ana[ana.grade == g]) for g in ("G1", "G2", "G3", "G4")]:
            v = sub[c].dropna()
            desc_rows.append({"metric": c, "label": lab, "group": key, "n": len(v),
                              "median": v.median() if len(v) else np.nan, "q1": v.quantile(.25) if len(v) else np.nan,
                              "q3": v.quantile(.75) if len(v) else np.nan, "mean": v.mean() if len(v) else np.nan,
                              "sd": v.std() if len(v) > 1 else np.nan})
    desc = pd.DataFrame(desc_rows)

    # 2) Mann–Whitney
    mw_rows = []
    for fam, lst in (("birincil", prim), ("ikincil", sec), ("duyarlılık", sensl)):
        for c, lab in lst:
            r = mann_whitney(ana, c, rng)
            if r:
                mw_rows.append({"family": fam, "metric": c, "label": lab, **r})
    mw = pd.DataFrame(mw_rows)
    for fam, fn, col in (("birincil", holm, "p_holm"), ("ikincil", bh_fdr, "p_fdr_bh")):
        m = mw.family == fam
        mw.loc[m, col] = fn(mw.loc[m, "p"].to_numpy())

    # 3) Spearman + Kruskal–Wallis
    sp_rows, kw_rows = [], []
    for fam, lst in (("birincil", prim), ("ikincil", sec)):
        for c, lab in lst:
            r = spearman_grade(ana, c, rng)
            if r:
                sp_rows.append({"family": fam, "metric": c, "label": lab, **r})
            k = kruskal_grade(ana, c)
            if k:
                kw_rows.append({"family": fam, "metric": c, "label": lab, **k})
    sp, kw = pd.DataFrame(sp_rows), pd.DataFrame(kw_rows)
    for df_, col in ((sp, "p_holm"), (kw, "p_holm")):
        m = df_.family == "birincil"
        df_.loc[m, col] = holm(df_.loc[m, "p"].to_numpy())

    # 4) Lojistik regresyon
    pcols = [c for c, _ in prim]
    uni = []
    for c in pcols + [c for c in ("area_um2_median", "area_um2_std") if c in ana]:
        rows, _ = logistic(ana, [c], "tek değişkenli")
        uni += rows
    covs = [c for c in ("n_nuclei", "focus_blurry_tissue_fraction", "is_top_section") if c in ana and ana[c].nunique() > 1]
    ana_m = ana.assign(n_nuclei=np.log10(ana["n_nuclei"])) if "n_nuclei" in ana else ana
    if "tissue_area_analyzed_mm2" in ana and ana["tissue_area_analyzed_mm2"].nunique() > 1:
        covs.append("tissue_area_analyzed_mm2")
    LABEL.update({"is_top_section": "Kesit tipi TS (vs BS)", "n_nuclei": "log10(çekirdek sayısı)"})
    cov_txt = ", ".join(LABEL.get(c, c) for c in covs)
    # (a) Eşzamanlı model (4 entropi birlikte) — yüksek korelasyon nedeniyle katsayılar tek tek yorumlanmamalı (VIF)
    multi, _ = logistic(ana_m, pcols + covs, f"eşzamanlı (4 entropi + {cov_txt})")
    for r in multi:
        r["label"] = LABEL.get(r["variable"], r["variable"])
    vif = vif_table(ana_m, pcols + covs)
    # (b) ANA düzeltilmiş analiz: her entropi AYRI modelde + karıştırıcılar
    adj = []
    for c in pcols:
        rows_, _ = logistic(ana_m, [c] + covs, f"düzeltilmiş (+ {cov_txt})")
        adj += [r for r in rows_ if r["variable"] == c]

    # 5) ROC (görünen) + bootstrap GA + Youden
    roc_rows = {}
    for c, lab in prim + sec:
        d = ana[["y_high", c]].dropna()
        auc, lo, hi = auc_bootstrap(d.y_high, d[c], rng)
        roc_rows[c] = {"metric": c, "label": lab, "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
                       **youden(d.y_high, d[c]), "note": "görünen (apparent); kesim noktası aynı veride seçildi → iyimser"}
    roc = pd.DataFrame(roc_rows.values())

    # 6) CV-AUC + artımsal değer
    cv_res, inc = {}, {}
    classic = [c for c in ("area_um2_median", "area_um2_std") if c in ana]
    spatial = [c for c in ("patch_nuc_area_entropy_std", "patch_nuc_area_entropy_p90") if c in ana and ana[c].notna().sum() > 10]
    models = {"Klasik (medyan alan + alan SD)": classic, "Entropi (4 ölçüt)": pcols, "Klasik + entropi": classic + pcols}
    if spatial:
        models["Klasik + uzamsal heterojenlik (tile entropisi SD, p90)"] = classic + spatial
    for name, cols in models.items():
        if cols:
            res, y, p = cv_auc(ana, cols)
            cv_res[name] = (res, y, p)
            inc[name] = res
    lr_test = {}
    try:
        from scipy.stats import chi2
        _, m0 = logistic(ana, classic, "base")
        _, m1 = logistic(ana, classic + pcols, "full")
        if m0 is not None and m1 is not None:
            lr = 2 * (m1.llf - m0.llf)
            lr_test = {"LR_chi2": float(lr), "df": len(pcols), "p": float(chi2.sf(lr, len(pcols))),
                       "AIC_base": float(m0.aic), "AIC_full": float(m1.aic)}
    except Exception as exc:  # noqa: BLE001
        lr_test = {"error": str(exc)}
    lr_spatial = {}
    if spatial:
        try:
            from scipy.stats import chi2
            d_ = ana.dropna(subset=classic + spatial + ["y_high"])
            _, m0 = logistic(d_, classic, "base")
            _, m1 = logistic(d_, classic + spatial, "full")
            if m0 is not None and m1 is not None:
                lr = 2 * (m1.llf - m0.llf)
                lr_spatial = {"LR_chi2": float(lr), "df": len(spatial), "p": float(chi2.sf(lr, len(spatial))),
                              "AIC_base": float(m0.aic), "AIC_full": float(m1.aic)}
        except Exception as exc:  # noqa: BLE001
            lr_spatial = {"error": str(exc)}

    # 7) Bin duyarlılığı
    sens_rows = []
    for c, lab in prim:
        f = c.replace("_entropy", "")
        for b in (10, 20, 40, 80):
            col = f"{f}_entropy_b{b}"
            if col in ana and ana[col].notna().sum() > 5:
                r = mann_whitney(ana, col, rng)
                sens_rows.append({"feature": c, "label": lab, "bins": b, **{k: r[k] for k in ("rank_biserial_r", "r_ci_low", "r_ci_high", "p")}})
    sens = pd.DataFrame(sens_rows)

    # 8) Karıştırıcılar
    conf = {}
    for c, lab in present(CONFOUNDERS):
        r = mann_whitney(ana, c, rng)
        if r:
            conf[c] = {"label": lab, **r}
    if "slide_kind" in ana and ana["slide_kind"].nunique() > 1:
        from scipy.stats import fisher_exact
        ct = pd.crosstab(ana["grade_group"], ana["slide_kind"])
        if ct.shape == (2, 2):
            conf["slide_kind"] = {"label": "Kesit tipi", "p": float(fisher_exact(ct.values)[1]), "table": ct.to_dict()}

    # 9) MERKEZ (TSS) — grade ile karışıyor mu?
    site_tab, site_test, ve_rows, site_adj = pd.DataFrame(), {}, [], []
    if "tss_code" in ana and ana["tss_code"].nunique() > 1:
        from scipy.stats import chi2_contingency
        site_tab = pd.crosstab(ana["tss_code"], ana["grade"]).reindex(columns=["G1", "G2", "G3", "G4"], fill_value=0)
        site_tab["n"] = site_tab.sum(1)
        site_tab["High_%"] = 100 * (site_tab["G3"] + site_tab["G4"]) / site_tab["n"]
        site_tab = site_tab.sort_values("n", ascending=False)
        g2 = pd.crosstab(site_groups(ana, 5), ana["grade_group"])
        if g2.shape[0] > 1 and g2.shape[1] == 2:
            chi2_, p_, dof_, _ = chi2_contingency(g2.values)
            site_test = {"chi2": float(chi2_), "df": int(dof_), "p": float(p_),
                         "cramers_v": float(math.sqrt(chi2_ / (g2.values.sum() * (min(g2.shape) - 1)))),
                         "note": "≥5 vakalı merkezler ayrı, diğerleri birleştirildi"}
        for c, lab in prim + [x for x in sec if x[0] == "log2_area_entropy"]:
            r = van_elteren(ana, c)
            if r:
                ve_rows.append({"metric": c, "label": lab, **r})
        if ve_rows:
            vdf = pd.DataFrame(ve_rows)
            m = vdf.metric.isin(pcols)
            vdf.loc[m, "p_holm"] = holm(vdf.loc[m, "p"].to_numpy())
            ve_rows = vdf.to_dict("records")
        sg = site_groups(ana, 10)
        dm = pd.get_dummies(sg, prefix="site", drop_first=True, dtype=float)
        if dm.shape[1]:
            ana_s = pd.concat([ana, dm], axis=1)
            for c in pcols:
                rows, _ = logistic(ana_s, [c], "merkeze göre düzeltilmiş", dummies=list(dm.columns))
                site_adj += rows

    # 10) Renk normalizasyonunun etkisi (proje başarı ölçütü: "renk varyasyonlarının azaltılması")
    #   Doğru ölçüt BOYA ŞİDDETİ: her slaytın 99. persentil H ve E konsantrasyonu (normalizasyon öncesi = ref / ölçek).
    #   Normalizasyon bunu referansa eşitler. Slayt ortalama OD'si ise doku BİLEŞİMİNİ (hücresellik, stroma) yansıtır;
    #   biyolojik olduğu için normalizasyonla azalması beklenmez → yalnız bilgi amaçlı raporlanır.
    norm_eff = []
    if "stain_conc_scale" in ana and ana["stain_conc_scale"].notna().any():
        sc = np.array([json.loads(v) if isinstance(v, str) else [np.nan, np.nan] for v in ana["stain_conc_scale"]], float)
        for k, nm in ((0, "Hematoksilen (99. persentil)"), (1, "Eozin (99. persentil)")):
            raw = 1.0 / sc[:, k]
            raw = raw[np.isfinite(raw)]
            if len(raw) > 2:
                norm_eff.append({"ölçüt": nm, "tip": "boya şiddeti", "CV_önce": float(raw.std(ddof=1) / raw.mean()), "CV_sonra": 0.0,
                                 "azalma_%": 100.0, "not": "normalizasyon her slaytı referans şiddete ölçekler"})
    if "stain_cos_h_to_ref" in ana:
        for col, nm in (("stain_cos_h_to_ref", "H boya vektörü açısı (°)"), ("stain_cos_e_to_ref", "E boya vektörü açısı (°)")):
            if col in ana and ana[col].notna().sum() > 2:
                ang = np.degrees(np.arccos(np.clip(ana[col].dropna().to_numpy(float), -1, 1)))
                norm_eff.append({"ölçüt": nm, "tip": "boya rengi", "ort_önce": float(ang.mean()), "SD_önce": float(ang.std(ddof=1)),
                                 "ort_sonra": 0.0, "SD_sonra": 0.0, "azalma_%": 100.0,
                                 "not": "her slaytın kendi H/E vektörü referans vektöre dönüştürülür"})
    if all(f"raw_mean_od_{c}" in ana for c in "rgb"):
        Xb = ana[[f"raw_mean_od_{c}" for c in "rgb"]].dropna().to_numpy()
        Xa = ana[[f"norm_mean_od_{c}" for c in "rgb"]].dropna().to_numpy()
        norm_eff.append({"ölçüt": "Slayt ortalama OD (bilgi amaçlı)", "tip": "doku bileşimi",
                         "azalma_%": float(100 * (1 - np.trace(np.cov(Xa.T)) / np.trace(np.cov(Xb.T)))),
                         "not": "hücresellik/stroma oranını yansıtır (biyolojik); normalizasyonla azalması beklenmez"})
    stain_tab = pd.crosstab(ana["stain_class"], ana["grade_group"]) if "stain_class" in ana else pd.DataFrame()

    # Figürler
    wf_info = {"mpp": float(ana["effective_mpp"].median()) if "effective_mpp" in ana else None,
               "tile_um": float(ana["tile_size_um"].median()) if "tile_size_um" in ana else None,
               "segmenter": ana["segmenter"].mode().iloc[0] if "segmenter" in ana and ana["segmenter"].notna().any() else None,
               "max_tiles": (int(ana["n_tiles_analyzed"].max()) if "n_tiles_analyzed" in ana and "n_tiles_eligible" in ana
                             and (ana["n_tiles_eligible"] > ana["n_tiles_analyzed"]).any() else None)}
    figs = [fig_workflow(out, demo, wf_info), fig_flow(allc, ana, out, demo),
            fig_box_group(ana, {r["metric"]: r for r in mw.to_dict("records")}, out, demo),
            fig_box_grade(ana, {r["metric"]: r for r in sp.to_dict("records")}, out, demo),
            fig_hist(ana, out, demo), fig_roc(ana, roc_rows, cv_res, out, demo),
            fig_forest([r for r in uni if r["variable"] in pcols], adj, out, demo),
            fig_corr(ana, out, demo), fig_bins(sens, out, demo) if len(sens) else None,
            fig_confounders(ana, conf, out, demo), fig_incremental(inc, out, demo),
            fig_site(site_tab, site_test, out, demo) if len(site_tab) else None,
            fig_normalization(ana, norm_eff, out, demo) if norm_eff else None]
    if results_dir:
        figs += [fig_heatmap_gallery(ana, results_dir, out, demo), fig_pooled_distributions(ana, results_dir, out, demo)]

    # Excel raporu
    xl = out / "statistics_report.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as xw:
        info = [("Oluşturulma", datetime.now().isoformat(timespec="seconds")),
                ("VERİ", "SİMÜLE — GERÇEK SONUÇ DEĞİL" if demo else "Gerçek pipeline çıktısı"),
                ("Analiz birimi", "Hasta (case_id); tile ve çekirdekler bağımsız gözlem DEĞİL"),
                ("n (Low / High)", f"{int((ana.y_high == 0).sum())} / {int((ana.y_high == 1).sum())}"),
                ("Birincil hipotez testi", "Mann–Whitney U, iki yönlü, 4 birincil entropi için Holm düzeltmesi"),
                ("Etki büyüklüğü", "rank-biserial r (+: High daha yüksek; |r| 0.1 küçük, 0.3 orta, 0.5 büyük); Hodges–Lehmann farkı"),
                ("GA", f"bootstrap persentil, {N_BOOT} tekrar, seed {SEED}"),
                ("Lojistik", f"OR 1 SD artış başına; düzeltilmiş modeller her entropi için ayrı: + {cov_txt}"),
                ("CV-AUC", "tekrarlı tabakalı 5-katlı CV × 20, L2 (C=1) lojistik, StandardScaler eğitim katında")] + [("Not", n) for n in notes]
        pd.DataFrame(info, columns=["alan", "değer"]).to_excel(xw, sheet_name="00_bilgi", index=False)
        ana.to_excel(xw, sheet_name="01_analysis_set", index=False)
        exclusion_table(allc).to_excel(xw, sheet_name="01b_analiz_disi", index=False)
        desc.to_excel(xw, sheet_name="02_descriptives", index=False)
        mw.to_excel(xw, sheet_name="03_mann_whitney", index=False)
        sp.to_excel(xw, sheet_name="04_spearman_grade", index=False)
        kw.to_excel(xw, sheet_name="05_kruskal_wallis", index=False)
        pd.DataFrame(uni).to_excel(xw, sheet_name="06_logistic_univariable", index=False)
        pd.DataFrame(adj).to_excel(xw, sheet_name="07_logistic_duzeltilmis", index=False)
        pd.DataFrame(multi).to_excel(xw, sheet_name="07b_logistic_eszamanli", index=False)
        vif.to_excel(xw, sheet_name="08_VIF", index=False)
        roc.to_excel(xw, sheet_name="09_roc_auc", index=False)
        pd.DataFrame([{"model": k, **v} for k, v in inc.items()]
                     + ([{"model": "LR testi (klasik → klasik+entropi)", **lr_test}] if lr_test else [])
                     + ([{"model": "LR testi (klasik → klasik+uzamsal)", **lr_spatial}] if lr_spatial else [])).to_excel(
            xw, sheet_name="10_cv_auc_incremental", index=False)
        sens.to_excel(xw, sheet_name="11_bin_sensitivity", index=False)
        pd.DataFrame([{"variable": k, **{kk: vv for kk, vv in v.items() if kk != "table"}} for k, v in conf.items()]).to_excel(xw, sheet_name="12_confounders", index=False)
        ana[[c for c, _ in prim + sec + present(CONFOUNDERS)]].corr(method="spearman").to_excel(xw, sheet_name="13_spearman_matrix")
        if len(site_tab):
            site_tab.to_excel(xw, sheet_name="14_merkez_x_grade")
            pd.DataFrame([site_test] if site_test else []).to_excel(xw, sheet_name="15_merkez_ki_kare", index=False)
            pd.DataFrame(ve_rows).to_excel(xw, sheet_name="16_van_elteren_merkez", index=False)
            pd.DataFrame(site_adj).to_excel(xw, sheet_name="17_logistic_merkez_duz", index=False)
        if norm_eff:
            pd.DataFrame(norm_eff).to_excel(xw, sheet_name="18_renk_normalizasyonu", index=False)
        if len(stain_tab):
            stain_tab.to_excel(xw, sheet_name="19_boya_sinifi_x_grup")

    # Metin özeti (rapora taslak)
    lines = ["SİMÜLE VERİ — BU SAYILAR SONUÇ DEĞİLDİR\n" if demo else "",
             f"Kohort: {len(allc)} hasta. Analiz seti: {len(ana)} hasta (Low {int((ana.y_high == 0).sum())}, High {int((ana.y_high == 1).sum())}).",
             *(f"  Analiz dışı — {r}: {len(g)} (grade: {', '.join(f'{k} {v}' for k, v in g['grade'].value_counts().sort_index().items())}; "
               f"merkez: {', '.join(f'{k} {v}' for k, v in g['tss_code'].value_counts().items())})"
               for r, g in exclusion_table(allc).groupby("neden", sort=False)), ""]
    lines.append("Birincil karşılaştırmalar (Mann–Whitney U, Holm):")
    for r in mw[mw.family == "birincil"].to_dict("records"):
        lines.append(f"  {r['label']}: Low medyan {r['median_low']:.3f} (IQR {r['iqr_low']:.3f}) vs High {r['median_high']:.3f} "
                     f"(IQR {r['iqr_high']:.3f}); r={r['rank_biserial_r']:+.2f} [{r['r_ci_low']:+.2f}, {r['r_ci_high']:+.2f}], "
                     f"HL farkı {r['hodges_lehmann_diff']:+.3f} bit, {fmt_p(r['p_holm'])} (Holm)")
    lines.append("\nGrade ile Spearman korelasyonu (G1..G4):")
    for r in sp[sp.family == "birincil"].to_dict("records"):
        lines.append(f"  {r['label']}: ρ={r['rho']:+.2f} [{r['rho_ci_low']:+.2f}, {r['rho_ci_high']:+.2f}], {fmt_p(r['p_holm'])} (Holm)")
    lines.append("\nAyırt edicilik (CV-AUC, ort. ± SD):")
    for k, v in inc.items():
        lines.append(f"  {k}: {v['cv_auc_mean']:.3f} ± {v['cv_auc_sd']:.3f}")
    if "LR_chi2" in lr_test:
        lines.append(f"  Entropinin klasik ölçülere artımsal katkısı: LR χ²={lr_test['LR_chi2']:.2f} (df={lr_test['df']}), {fmt_p(lr_test['p'])}")
    if "LR_chi2" in lr_spatial:
        lines.append(f"  Uzamsal heterojenliğin (tile entropisi SD + p90) klasik ölçülere katkısı: LR χ²={lr_spatial['LR_chi2']:.2f} "
                     f"(df={lr_spatial['df']}), {fmt_p(lr_spatial['p'])}")
    lines.append("\nKarıştırıcılara göre düzeltilmiş lojistik regresyon (her entropi ayrı model; OR / 1 SD):")
    for r in adj:
        if "OR_per_SD" in r:
            lines.append(f"  {r['label']}: OR {r['OR_per_SD']:.2f} [{r['OR_ci_low']:.2f}–{r['OR_ci_high']:.2f}], {fmt_p(r['p'])}  ({r['model']})")
    sens_mw = mw[mw.family == "duyarlılık"]
    if len(sens_mw):
        lines.append("\nDuyarlılık analizleri (Mann–Whitney, düzeltmesiz):")
        for r in sens_mw.to_dict("records"):
            lines.append(f"  {r['label']}: r={r['rank_biserial_r']:+.2f}, {fmt_p(r['p'])}")
    if site_test:
        lines.append(f"\nMerkez × grade grubu: χ²={site_test['chi2']:.1f} (df={site_test['df']}), {fmt_p(site_test['p'])}, "
                     f"Cramér V={site_test['cramers_v']:.2f}" + ("  → MERKEZ KARIŞTIRICI OLABİLİR; tabakalı sonuçları öne çıkarın" if site_test['p'] < 0.05 else ""))
    for r in ve_rows:
        lines.append(f"  Merkeze göre tabakalı (van Elteren) {r['label']}: z={r['z']:+.2f}, {fmt_p(r.get('p_holm', r['p']))}")
    for r in site_adj:
        if "OR_per_SD" in r:
            lines.append(f"  Merkeze göre düzeltilmiş OR/SD {r['label']}: {r['OR_per_SD']:.2f} [{r['OR_ci_low']:.2f}–{r['OR_ci_high']:.2f}], {fmt_p(r['p'])}")
    if norm_eff:
        lines.append("\nRenk normalizasyonu:")
        for r in norm_eff:
            if r.get("tip") == "boya şiddeti":
                lines.append(f"  {r['ölçüt']}: slaytlar arası CV {r['CV_önce']:.2f} → 0 (%100 azalma)")
            elif r.get("tip") == "boya rengi":
                lines.append(f"  {r['ölçüt']}: referansa sapma {r['ort_önce']:.1f}° ± {r['SD_önce']:.1f}° → 0°")
            else:
                lines.append(f"  {r['ölçüt']}: varyans değişimi %{r['azalma_%']:.0f} — doku bileşimini yansıtır, başarı ölçütü değildir")
    lines.append("\nKarıştırıcı kontrolü (Low vs High):")
    for k, v in conf.items():
        lines.append(f"  {v['label']}: {fmt_p(v['p'])}")
    lines += ["", "Yorum uyarıları:",
              "  • Görünen AUC ve Youden kesim noktası aynı veride seçildiği için iyimserdir; CV-AUC raporlanmalıdır.",
              "  • Karıştırıcılardan biri gruplar arasında farklıysa (p<0.05), çok değişkenli model sonucu öne çıkarılmalıdır.",
              "  • Entropi klasik SD/CV ile çok yüksek korelasyonluysa (Şekil 8, |ρ|>0.8), 'yeni bilgi' iddiası artımsal değer testine dayanmalıdır.",
              "  • Bulgular segmentasyonun Dice/F1 doğrulamasına koşulludur."] + [f"  • {n}" for n in notes]
    (out / "methods_and_results.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nÇıktılar: {out}\n  " + "\n  ".join(str(f.name) for f in figs if f) + f"\n  {xl.name}")
    return mw


def main(argv=None):
    ap = argparse.ArgumentParser(description="KIRC entropi — istatistik ve figürler (3. aşama)")
    ap.add_argument("--results", type=Path, help="kirc_entropy_pipeline.py çıktı klasörü")
    ap.add_argument("--excel", type=Path, help="En güncel Case_Selection Excel'i (grade kaynağı)")
    ap.add_argument("--out", type=Path, help="Çıktı klasörü (varsayılan: <results>/statistics)")
    ap.add_argument("--demo", action="store_true", help="SİMÜLE kohortla örnek çıktı üret (sonuç değildir)")
    a = ap.parse_args(argv)
    if a.demo:
        ana = simulate_cohort()
        out = a.out or Path("statistics_DEMO_simulated")
        run(ana, ana.copy(), a.results, out, demo=True, notes=["Tüm veriler simüle edilmiştir; yalnızca çıktı biçimini gösterir."])
        return
    if not a.results:
        raise SystemExit("--results gerekli (veya --demo)")
    ana, allc, notes = load_analysis_set(a.results, a.excel)
    run(ana, allc, a.results, a.out or a.results / "statistics", demo=False, notes=notes)


if __name__ == "__main__":
    main()
