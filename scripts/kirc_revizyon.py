#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Danışman revizyonları (makale hazırlığı) — proje.py komutlarıyla çalıştırılır.

R1  uzamsal-ozgulluk : Uzamsal kazanç ENTROPİYE mi özgü? Tile düzeyi entropi yerine tile düzeyi
                       çekirdek alanı SD'si (ve CV'si) ile artımsal analiz tekrarlanır.

Tüm analizler kayıtlı çekirdek tablolarından (09_results/pipeline/cases/<slayt>/nuclei_features.csv.gz)
yeniden hesaplanır; slayt indirilmez. Analiz seti, ana istatistikteki 200 hastadır.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kirc_entropy_pipeline as kp  # noqa: E402
import kirc_stats as ks  # noqa: E402

CLASSIC = ["area_um2_median", "area_um2_std"]          # rapordaki "klasik" model (Şekil 11)

# -----------------------------------------------------------------------------------------------
# Ortak: çekirdek tablolarını vaka vaka okuma
# -----------------------------------------------------------------------------------------------

def odakli_cekirdekler(path: Path, cfg: kp.Config) -> pd.DataFrame:
    """Ana analizdeki seçimin aynısı: odaklı çekirdekler (ve varsa sınıf filtresi)."""
    nuc = pd.read_csv(path)
    if "in_focus" in nuc and cfg.focus_exclude:
        nuc = nuc[nuc["in_focus"].astype(bool)]
    if cfg.nucleus_classes_include and "nucleus_class" in nuc:
        nuc = nuc[nuc["nucleus_class"].isin(cfg.nucleus_classes_include)]
    return nuc


def vaka_dongusu(ana: pd.DataFrame, results: Path, cfg: kp.Config, fonk, etiket: str) -> pd.DataFrame:
    rows, eksik = [], []
    for k, r in enumerate(ana.itertuples(), 1):
        p = results / "cases" / str(r.slide_id) / "nuclei_features.csv.gz"
        if not p.exists():
            eksik.append(r.case_id)
            continue
        rec = {"case_id": r.case_id}
        nuc = odakli_cekirdekler(p, cfg)
        mpp0 = getattr(r, "image_resolution_or_mpp", None)
        nuc.attrs["mpp_l0"] = float(mpp0) if mpp0 is not None and np.isfinite(mpp0) else None
        rec.update(fonk(nuc, cfg))
        rows.append(rec)
        if k % 25 == 0 or k == len(ana):
            print(f"  {etiket}: {k}/{len(ana)} vaka")
    if eksik:
        print(f"  UYARI: {len(eksik)} vakanın çekirdek tablosu yok: {', '.join(eksik[:10])}{' …' if len(eksik) > 10 else ''}")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------------------------
# R1 — Uzamsal kazanç entropiye mi özgü?
# -----------------------------------------------------------------------------------------------
# Tile düzeyi ölçütler (her tile için, ≥ min_nuclei_tile odaklı çekirdek): alan entropisi (ana analizle aynı
# sabit bin), alan SD'si, alan CV'si (ölçekten bağımsız). Vaka düzeyinde her birinin tile'lar arası SD'si
# (uzamsal heterojenlik) ve 90. persentili (en heterojen bölge) alınır.
TILE_METRICS = {"ent": "alan entropisi", "sd": "alan SD'si", "cv": "alan CV'si"}


def uzamsal_olcutler(nuc: pd.DataFrame, cfg: kp.Config) -> dict:
    lo, hi, nb = cfg.feature_bins["area_um2"]
    out = {}
    if not len(nuc) or "tile_id" not in nuc:
        return out
    g = nuc.groupby("tile_id")["area_um2"]
    t = pd.DataFrame({"n": g.size(), "sd": g.std(ddof=1), "mean": g.mean()})
    t["cv"] = t["sd"] / t["mean"]
    t["ent"] = g.apply(lambda v: kp.histogram_entropy(v, lo, hi, int(nb), cfg.log_base)["entropy"])
    t = t[t["n"] >= cfg.min_nuclei_tile]
    out["r1_n_tile"] = int(len(t))
    if len(t) < 5:
        return out
    q = cfg.hotspot_percentile
    for m in TILE_METRICS:
        out[f"tile_{m}_sd"] = float(t[m].std(ddof=1))
        out[f"tile_{m}_p{int(q)}"] = float(np.percentile(t[m], q))
        out[f"tile_{m}_mean"] = float(t[m].mean())
    return out


R1_LABEL = {"tile_ent_sd": "Tile alan entropisi SD", "tile_ent_p90": "Tile alan entropisi p90",
            "tile_sd_sd": "Tile alan SD'si SD", "tile_sd_p90": "Tile alan SD'si p90",
            "tile_cv_sd": "Tile alan CV'si SD", "tile_cv_p90": "Tile alan CV'si p90",
            "tile_ent_mean": "Tile alan entropisi ort.", "tile_sd_mean": "Tile alan SD'si ort.",
            "tile_cv_mean": "Tile alan CV'si ort.", **ks.LABEL}


def _lr(d, base, full):
    from scipy.stats import chi2
    _, m0 = ks.logistic(d, base, "b")
    _, m1 = ks.logistic(d, full, "f")
    if m0 is None or m1 is None:
        return {"LR_chi2": np.nan, "df": len(full) - len(base), "p": np.nan, "AIC_base": np.nan, "AIC_full": np.nan}
    lr = 2 * (m1.llf - m0.llf)
    df_ = len(full) - len(base)
    return {"LR_chi2": float(lr), "df": df_, "p": float(chi2.sf(lr, df_)), "AIC_base": float(m0.aic), "AIC_full": float(m1.aic)}


def _oof_repeats(d, cols, n_splits=5, n_repeats=20, seed=ks.SEED):
    """Her tekrar için out-of-fold AUC'lar (aynı bölünmeler → modeller eşleştirilmiş karşılaştırılır)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = d[cols].to_numpy(float), d["y_high"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = np.zeros((n_repeats, len(y)))
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X[tr], y[tr])
        oof[i // n_splits, te] = m.predict_proba(X[te])[:, 1]
    return np.array([roc_auc_score(y, oof[r]) for r in range(n_repeats)]), oof.mean(axis=0), y


def _delta_auc_boot(y, s1, s0, rng, n=2000):
    from sklearn.metrics import roc_auc_score
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s0[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def r1_analiz(d: pd.DataFrame, rng=None) -> dict:
    """d: analiz seti + uzamsal ölçütler (y_high, grade_num, CLASSIC, tile_* sütunları)."""
    rng = rng or np.random.default_rng(ks.SEED)
    from scipy.stats import spearmanr
    ENT, SD, CV = ["tile_ent_sd", "tile_ent_p90"], ["tile_sd_sd", "tile_sd_p90"], ["tile_cv_sd", "tile_cv_p90"]
    allc = CLASSIC + ENT + SD + CV
    d = d.dropna(subset=["y_high", *allc]).reset_index(drop=True)

    # 1) Tek değişkenli
    uni = []
    for c in [*ENT, *SD, *CV, "tile_ent_mean", "tile_sd_mean", "tile_cv_mean"]:
        mw = ks.mann_whitney(d, c, rng)
        sp = ks.spearman_grade(d, c, rng)
        uni.append({"ölçüt": R1_LABEL[c], "değişken": c, "n": int(d[c].notna().sum()),
                    "r (High−Low)": mw["rank_biserial_r"], "r_GA": f"[{mw['r_ci_low']:+.2f}, {mw['r_ci_high']:+.2f}]",
                    "p_MWU": mw["p"], "rho_grade": sp["rho"], "p_spearman": sp["p"], "AUC_görünen": mw["auc_apparent"]})
    uni = pd.DataFrame(uni)

    # 2) Modeller (aynı bölünmeler, eşleştirilmiş)
    models = {"M0 Klasik (medyan alan + alan SD)": CLASSIC,
              "M1 Klasik + tile ENTROPİ (SD, p90)": CLASSIC + ENT,
              "M2 Klasik + tile alan SD'si (SD, p90)": CLASSIC + SD,
              "M3 Klasik + tile alan CV'si (SD, p90)": CLASSIC + CV,
              "M4 Klasik + tile entropi + tile alan SD'si": CLASSIC + ENT + SD}
    cv, oofs, y = {}, {}, None
    for name, cols in models.items():
        aucs, oof, y = _oof_repeats(d, cols)
        cv[name], oofs[name] = aucs, oof
    k0 = list(models)[0]
    mod = []
    for name, cols in models.items():
        a = cv[name]
        row = {"model": name, "değişken_sayısı": len(cols), "CV_AUC_ort": a.mean(), "CV_AUC_SD": a.std(ddof=1),
               "ΔCV_AUC_vs_M0": (a - cv[k0]).mean(),
               "tekrar_yuzde_M0_iyi": 100 * float((a > cv[k0]).mean()) if name != k0 else np.nan}
        if name != k0:
            lo_, hi_ = _delta_auc_boot(y, oofs[name], oofs[k0], rng)
            row.update({"ΔAUC_boot_GA_low": lo_, "ΔAUC_boot_GA_high": hi_})
        _, m = ks.logistic(d, cols, name)
        row["AIC"] = float(m.aic) if m is not None else np.nan
        mod.append(row)
    mod = pd.DataFrame(mod)

    # 3) İç içe olabilirlik oranı testleri
    tests = [("Entropi → klasik (M0→M1)", CLASSIC, CLASSIC + ENT),
             ("Alan SD'si → klasik (M0→M2)", CLASSIC, CLASSIC + SD),
             ("Alan CV'si → klasik (M0→M3)", CLASSIC, CLASSIC + CV),
             ("ENTROPİ, tile alan SD'sinin ÜSTÜNE (M2→M4)", CLASSIC + SD, CLASSIC + ENT + SD),
             ("Tile alan SD'si, ENTROPİNİN ÜSTÜNE (M1→M4)", CLASSIC + ENT, CLASSIC + ENT + SD)]
    lr = pd.DataFrame([{"test": t, **_lr(d, b, f)} for t, b, f in tests])

    # 4) Entropi ve SD tabanlı uzamsal ölçütler ne kadar örtüşüyor?
    cor = []
    for a_, b_ in (("tile_ent_sd", "tile_sd_sd"), ("tile_ent_p90", "tile_sd_p90"), ("tile_ent_sd", "tile_cv_sd"),
                   ("tile_ent_p90", "tile_cv_p90"), ("tile_ent_sd", "area_um2_std"), ("tile_sd_sd", "area_um2_std")):
        rho, p = spearmanr(d[a_], d[b_])
        cor.append({"ölçüt 1": R1_LABEL[a_], "ölçüt 2": R1_LABEL[b_], "Spearman ρ": rho, "p": p})
    cor = pd.DataFrame(cor)

    # 5) Önceden belirlenmiş karar kuralı
    p_e_over_s = lr.loc[3, "p"]
    p_s_over_e = lr.loc[4, "p"]
    p_sd = lr.loc[1, "p"]
    if p_e_over_s < 0.05 and p_s_over_e >= 0.05:
        karar = ("ENTROPİYE ÖZGÜ: tile entropisi, tile alan SD'sinin üstüne bilgi ekliyor; tersi geçerli değil. "
                 "Makalede 'uzamsal entropi heterojenliği' çerçevesi korunur (en güçlü nokta).")
    elif p_sd < 0.05 and p_e_over_s >= 0.05:
        karar = ("ENTROPİYE ÖZGÜ DEĞİL: tile alan SD'si benzer kazanç sağlıyor ve entropi bunun üstüne ek bilgi taşımıyor. "
                 "Makalede bulgu 'çekirdek boyutunun uzamsal heterojenliği' olarak yeniden çerçevelenir.")
    elif p_e_over_s < 0.05 and p_s_over_e < 0.05:
        karar = "TAMAMLAYICI: her iki ölçüt de diğerinin üstüne bilgi ekliyor; ikisi birlikte raporlanır."
    else:
        karar = ("AYIRT EDİLEMİYOR: ölçütler büyük ölçüde örtüşüyor, hiçbiri diğerinin üstüne anlamlı bilgi eklemiyor. "
                 "Bulgu 'çekirdek boyutunun uzamsal heterojenliği' olarak, entropi ve SD birlikte sunularak raporlanır.")
    return {"d": d, "uni": uni, "mod": mod, "lr": lr, "cor": cor, "karar": karar, "cv": cv, "oofs": oofs, "y": y}


def r1_sekil(res: dict, path: Path):
    plt = ks.setup_style()
    from sklearn.metrics import roc_curve
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    mod = res["mod"]
    cols = ["#6f6e6a", ks.C_LOW, "#2f8f6f", "#b58b2a", ks.C_HIGH]
    ax = axs[0]
    yy = np.arange(len(mod))[::-1]
    ax.barh(yy, mod["CV_AUC_ort"], xerr=mod["CV_AUC_SD"], color=cols, alpha=0.85, height=0.6, capsize=4)
    for yi, v in zip(yy, mod["CV_AUC_ort"]):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(yy, [m.split(" ", 1)[1] for m in mod["model"]], fontsize=9.5)
    ax.set_xlim(0.5, max(0.85, float(mod["CV_AUC_ort"].max()) + 0.08))
    ax.set_xlabel("çapraz doğrulamalı AUC (5 katlı × 20)")
    ax.set_title("a  Uzamsal ölçüt türüne göre artımsal değer")
    ax = axs[1]
    d = res["d"]
    for g, c in ((0, ks.C_LOW), (1, ks.C_HIGH)):
        s = d[d.y_high == g]
        ax.scatter(s["tile_sd_sd"], s["tile_ent_sd"], s=16, color=c, alpha=0.7, label="High" if g else "Low")
    r0 = res["cor"].iloc[0]
    ax.set_xlabel("tile alan SD'sinin tile'lar arası SD'si (µm²)")
    ax.set_ylabel("tile alan entropisinin tile'lar arası SD'si (bit)")
    ax.set_title(f"b  İki uzamsal ölçüt (Spearman ρ = {r0['Spearman ρ']:.2f})")
    ax.legend()
    ax = axs[2]
    for (name, oof), c in zip(res["oofs"].items(), cols):
        fpr, tpr, _ = roc_curve(res["y"], oof)
        ax.plot(fpr, tpr, color=c, lw=1.8, label=name.split(" ", 1)[0])
    ax.plot([0, 1], [0, 1], ":", color=ks.INK2)
    ax.set_xlabel("1 − özgüllük"); ax.set_ylabel("duyarlılık")
    ax.set_title("c  Out-of-fold ROC (20 tekrarın ortalaması)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def r1_rapor(res: dict) -> str:
    L = ["", "R1 — UZAMSAL KAZANÇ ENTROPİYE Mİ ÖZGÜ?", f"Analiz edilen hasta: {len(res['d'])}", "",
         "Tek değişkenli (High vs Low):"]
    for _, r in res["uni"].iterrows():
        L.append(f"  {r['ölçüt']:<28} r={r['r (High−Low)']:+.2f} {r['r_GA']}  {ks.fmt_p(r['p_MWU'])} · "
                 f"ρ_grade={r['rho_grade']:+.2f} · AUC {r['AUC_görünen']:.3f}")
    L += ["", "Modeller (CV-AUC, aynı bölünmeler):"]
    for _, r in res["mod"].iterrows():
        extra = ""
        if pd.notna(r.get("ΔAUC_boot_GA_low", np.nan)):
            extra = (f"  Δ={r['ΔCV_AUC_vs_M0']:+.3f} [GA {r['ΔAUC_boot_GA_low']:+.3f}, {r['ΔAUC_boot_GA_high']:+.3f}], "
                     f"tekrarların %{r['tekrar_yuzde_M0_iyi']:.0f}'inde M0'dan iyi")
        L.append(f"  {r['model']:<44} {r['CV_AUC_ort']:.3f} ± {r['CV_AUC_SD']:.3f}  AIC {r['AIC']:.1f}{extra}")
    L += ["", "Olabilirlik oranı testleri:"]
    for _, r in res["lr"].iterrows():
        L.append(f"  {r['test']:<46} χ²={r['LR_chi2']:.2f} (df={r['df']}), {ks.fmt_p(r['p'])}")
    L += ["", "Örtüşme (Spearman):"]
    for _, r in res["cor"].iterrows():
        L.append(f"  {r['ölçüt 1']} ↔ {r['ölçüt 2']}: ρ={r['Spearman ρ']:.2f}")
    L += ["", "KARAR: " + res["karar"]]
    return "\n".join(L)


def r1_kaydet(res: dict, xlsx: Path, png: Path, csv: Path):
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"açıklama": [
            "R1: Uzamsal kazanç entropiye mi özgü? Tile düzeyi alan entropisi yerine tile düzeyi alan SD'si / CV'si.",
            "Tile: ≥30 odaklı çekirdekli 1024 px (258 µm) tile; vaka ölçütü: tile değerlerinin tile'lar arası SD'si ve p90'ı.",
            "Klasik model: medyan alan + alan SD (raporla aynı). CV: 5 katlı × 20 tekrar, tüm modellerde aynı bölünmeler.",
            "ΔAUC GA: ortalama out-of-fold olasılıklar üzerinde hastaların 2000 bootstrap'ı.",
            "KARAR: " + res["karar"]]}).to_excel(xw, sheet_name="00_bilgi", index=False)
        res["uni"].to_excel(xw, sheet_name="01_tek_degiskenli", index=False)
        res["mod"].to_excel(xw, sheet_name="02_modeller_cv_auc", index=False)
        res["lr"].to_excel(xw, sheet_name="03_LR_testleri", index=False)
        res["cor"].to_excel(xw, sheet_name="04_ortusme", index=False)
    keep = ["case_id", "grade", "y_high", "r1_n_tile"] + [c for c in res["d"].columns if c.startswith("tile_")]
    res["d"][[c for c in keep if c in res["d"]]].to_csv(csv, index=False)
    r1_sekil(res, png)


def r1_calistir(ana: pd.DataFrame, results: Path, cfg: kp.Config, out_xlsx: Path, out_png: Path, out_csv: Path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sp = vaka_dongusu(ana, results, cfg, uzamsal_olcutler, "R1 tile ölçütleri")
    d = ana.merge(sp, on="case_id", how="inner")
    # Tutarlılık: yeniden hesaplanan tile entropisi SD'si = ana analizdeki değer mi?
    if "patch_nuc_area_entropy_std" in d:
        diff = (d["tile_ent_sd"] - d["patch_nuc_area_entropy_std"]).abs().max()
        print(f"  Kontrol: ana analizdeki tile entropisi SD'si ile en büyük fark = {diff:.2e} "
              f"({'AYNI ✓' if diff < 1e-6 else 'FARKLI — bildirin'})")
    res = r1_analiz(d)
    r1_kaydet(res, out_xlsx, out_png, out_csv)
    txt = r1_rapor(res)
    print(txt)
    return res, txt


# -----------------------------------------------------------------------------------------------
# R4 — Yalnız tümör çekirdekleri (lenfosit benzeri ve iğsi çekirdekler dışlanınca)
# -----------------------------------------------------------------------------------------------
# Doğrulanmış bir hücre sınıflandırıcısı DEĞİL, morfolojik kurallar (önceden belirlendi, sonuçlara bakılmadan):
#   küçük   : alan ≤ 30 µm²  (lenfosit çekirdeği ~6 µm çap; segmentasyon alanları gerçek alandan küçük)
#   yuvarlak: eksantrisite ≤ 0.65
#   koyu    : hematoksilen ortalaması ≥ slaytın 70. persentili (lenfositler belirgin hiperkromatik)
#   kümeli  : 12 µm içinde ≥3 başka küçük çekirdek (lenfositler sık kümelenir; berrak hücreli tümör
#             hücreleri geniş sitoplazma nedeniyle seyrek dizilir) — düşük dereceli tümörün küçük-koyu
#             çekirdeklerinin yanlışlıkla dışlanmasını azaltmak için
# Varyantlar:
#   A  lenfosit kümesi dışlandı      : küçük & yuvarlak & koyu & kümeli
#   B  küçük-yuvarlak-koyu dışlandı  : küçük & yuvarlak & koyu (kümelenme şartı yok; en agresif, üst sınır)
#   C  tümörle zenginleştirilmiş     : A + iğsi çekirdekler (endotel/fibroblast; ana analizdeki kural)
LYM_AREA_MAX, LYM_ECC_MAX, LYM_H_PCT, LYM_R_UM, LYM_K = 30.0, 0.65, 70, 12.0, 3
R4_VARIANTS = {"A": "Lenfosit benzeri kümeler dışlandı", "B": "Küçük-yuvarlak-koyu tümü dışlandı",
               "C": "Tümörle zenginleştirilmiş (A + iğsi)"}


def lenfosit_isaretle(nuc: pd.DataFrame, cfg: kp.Config) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    mpp = float(cfg.target_mpp or 0.25)
    small = nuc["area_um2"] <= LYM_AREA_MAX
    rnd = nuc["eccentricity"] <= LYM_ECC_MAX
    if "h_mean" in nuc and nuc["h_mean"].notna().any():
        dark = nuc["h_mean"] >= np.nanpercentile(nuc["h_mean"], LYM_H_PCT)
    else:
        dark = pd.Series(True, index=nuc.index)
    cand = small & rnd & dark
    clus = pd.Series(False, index=nuc.index)
    # Kayıtlı tablolarda merkezler seviye-0 piksel (x_l0, y_l0); eski/sentetik tablolarda tile içi analiz pikseli
    if {"x_l0", "y_l0"} <= set(nuc.columns):
        xcol, ycol, scale = "x_l0", "y_l0", float(nuc.attrs.get("mpp_l0") or mpp)
    elif {"centroid_x_px", "centroid_y_px"} <= set(nuc.columns):
        xcol, ycol, scale = "centroid_x_px", "centroid_y_px", mpp
    else:
        raise SystemExit("Çekirdek tablosunda merkez koordinatları yok (x_l0/y_l0) — kümelenme şartı uygulanamaz.")
    if "tile_id" in nuc:
        for _, g in nuc[small].groupby("tile_id"):
            if len(g) <= LYM_K:
                continue
            xy = g[[xcol, ycol]].to_numpy(float) * scale
            cnt = np.array([len(v) - 1 for v in cKDTree(xy).query_ball_point(xy, LYM_R_UM)])
            clus.loc[g.index] = cnt >= LYM_K
    spindle = (nuc["eccentricity"] >= cfg.spindle_min_eccentricity) & (nuc["minor_axis_um"] <= cfg.spindle_max_minor_axis_um) \
        if "minor_axis_um" in nuc else pd.Series(False, index=nuc.index)
    out = pd.DataFrame({"A": cand & clus, "B": cand, "C": (cand & clus) | spindle}, index=nuc.index)
    out.attrs["clus_frac"] = float(clus.mean())
    return out


def tumor_cekirdek_olcutleri(nuc: pd.DataFrame, cfg: kp.Config) -> dict:
    out = {}
    if not len(nuc):
        return out
    fl = lenfosit_isaretle(nuc, cfg)
    out["r4_has_h_mean"] = int("h_mean" in nuc)
    out["r4_kumeli_kucuk_oran"] = float(fl.attrs.get("clus_frac", np.nan))
    for v in R4_VARIANTS:
        sub = nuc[~fl[v]]
        out[f"r4_{v}_dislanan_oran"] = float(fl[v].mean())
        out[f"r4_{v}_n_nuclei"] = int(len(sub))
        if len(sub) < cfg.min_nuclei_case:
            continue
        e = kp._entropy_block(sub, cfg, suffix=f"_{v}", full=False)
        out.update({f"{f}_entropy_{v}": e.get(f"{f}_entropy_{v}") for f in kp.MORPH_FEATURES})
        out[f"area_um2_median_{v}"] = float(sub["area_um2"].median())
        out[f"area_um2_std_{v}"] = float(sub["area_um2"].std(ddof=1))
        sp = uzamsal_olcutler(sub, cfg)
        out[f"tile_ent_sd_{v}"], out[f"tile_ent_p90_{v}"] = sp.get("tile_ent_sd"), sp.get("tile_ent_p90")
    return out


def r4_analiz(d: pd.DataFrame, rng=None) -> dict:
    rng = rng or np.random.default_rng(ks.SEED)
    from scipy.stats import chi2
    feats = [(f"{f}_entropy", ks.LABEL.get(f"{f}_entropy", f)) for f in kp.MORPH_FEATURES]
    covs = [c for c in ("focus_blurry_tissue_fraction", "tissue_area_analyzed_mm2") if c in d and d[c].nunique() > 1]

    # 1) Dışlanan oran — gruplar / grade'ler arasında
    frac = []
    for v, lab in R4_VARIANTS.items():
        c = f"r4_{v}_dislanan_oran"
        mw = ks.mann_whitney(d, c, rng)
        sp = ks.spearman_grade(d, c, rng)
        row = {"varyant": f"{v}: {lab}", "Low medyan %": 100 * d.loc[d.y_high == 0, c].median(),
               "High medyan %": 100 * d.loc[d.y_high == 1, c].median(), "p_MWU": mw["p"], "rho_grade": sp["rho"]}
        for g in ("G1", "G2", "G3", "G4"):
            row[f"{g} medyan %"] = 100 * d.loc[d.grade == g, c].median()
        frac.append(row)
    frac = pd.DataFrame(frac)

    # 2) Entropi bulguları her varyantta
    rows = []
    for v, vlab in [("", "ANA ANALİZ (tüm odaklı çekirdekler)")] + [(k, f"{k}: {l}") for k, l in R4_VARIANTS.items()]:
        sfx = f"_{v}" if v else ""
        nn = f"r4_{v}_n_nuclei" if v else "n_nuclei"
        dm = d.assign(_log_n=np.log10(d[nn].clip(lower=1)))
        for base, lab in feats:
            c = base + sfx
            if c not in d or d[c].notna().sum() < 20:
                continue
            mw = ks.mann_whitney(d, c, rng)
            sp = ks.spearman_grade(d, c, rng)
            r_, _ = ks.logistic(dm, [c, "_log_n", *covs], "adj")
            orr = next((x for x in r_ if x.get("variable") == c), {})
            rows.append({"analiz": vlab, "ölçüt": lab, "n": int(d[c].notna().sum()),
                         "r (High−Low)": mw["rank_biserial_r"], "r_GA_low": mw["r_ci_low"], "r_GA_high": mw["r_ci_high"],
                         "p_MWU": mw["p"], "rho_grade": sp["rho"], "p_spearman": sp["p"],
                         "OR_düzeltilmiş/SD": orr.get("OR_per_SD"), "OR_GA_low": orr.get("OR_ci_low"),
                         "OR_GA_high": orr.get("OR_ci_high"), "p_OR": orr.get("p")})
    ent = pd.DataFrame(rows)

    # 3) Uzamsal heterojenlik bulgusu (R1/rapor) her varyantta: klasik → klasik + tile entropisi SD/p90
    sp_rows = []
    for v, vlab in [("", "ANA ANALİZ")] + [(k, f"{k}: {l}") for k, l in R4_VARIANTS.items()]:
        sfx = f"_{v}" if v else ""
        cl = [f"area_um2_median{sfx}", f"area_um2_std{sfx}"]
        spc = [f"tile_ent_sd{sfx}", f"tile_ent_p90{sfx}"] if v else ["patch_nuc_area_entropy_std", "patch_nuc_area_entropy_p90"]
        if not all(c in d for c in cl + spc):
            continue
        dd = d.dropna(subset=cl + spc + ["y_high"])
        _, m0 = ks.logistic(dd, cl, "b")
        _, m1 = ks.logistic(dd, cl + spc, "f")
        lr = 2 * (m1.llf - m0.llf)
        a0, _, _ = _oof_repeats(dd, cl)
        a1, _, _ = _oof_repeats(dd, cl + spc)
        sp_rows.append({"analiz": vlab, "n": len(dd), "CV_AUC_klasik": a0.mean(), "CV_AUC_klasik+uzamsal": a1.mean(),
                        "ΔCV_AUC": (a1 - a0).mean(), "LR_chi2": float(lr), "p_LR": float(chi2.sf(lr, 2))})
    spt = pd.DataFrame(sp_rows)
    return {"d": d, "frac": frac, "ent": ent, "sp": spt}


def r4_sekil(res: dict, path: Path):
    plt = ks.setup_style()
    d, ent = res["d"], res["ent"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axs[0]
    grades = ["G1", "G2", "G3", "G4"]
    cols = {"A": ks.C_LOW, "B": "#6f6e6a", "C": "#2f8f6f"}
    for i, v in enumerate(R4_VARIANTS):
        c = f"r4_{v}_dislanan_oran"
        data = [100 * d.loc[d.grade == g, c].dropna().to_numpy() for g in grades]
        pos = np.arange(4) + (i - 1) * 0.26
        bp = ax.boxplot(data, positions=pos, widths=0.22, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=cols[v], alpha=0.55, edgecolor=cols[v])
        for m in bp["medians"]:
            m.set(color=ks.INK)
        ax.plot([], [], color=cols[v], lw=8, alpha=0.55, label=f"{v}: {R4_VARIANTS[v]}")
    ax.set_xticks(range(4), grades)
    ax.set_ylabel("dışlanan çekirdek oranı (%)")
    ax.set_title("a  Dışlanan çekirdek oranı (grade'e göre)")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
    ax = axs[1]
    labs = [l for l in ent["analiz"].unique()]
    feats = list(ent["ölçüt"].unique())
    fc = [ks.C_HIGH, "#b58b2a", ks.C_LOW, "#6f6e6a"]
    y = 0
    yt, yl = [], []
    for j, f in enumerate(feats):
        for k, a in enumerate(labs):
            r = ent[(ent["ölçüt"] == f) & (ent["analiz"] == a)]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(r["r (High−Low)"], y, xerr=[[r["r (High−Low)"] - r["r_GA_low"]], [r["r_GA_high"] - r["r (High−Low)"]]],
                        fmt="o", color=fc[j], ms=5 if k else 7, mfc="white" if k else fc[j], capsize=2)
            yt.append(y); yl.append(f"{f} · {a.split(':')[0] if k else 'Ana'}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=ks.INK2, lw=1, ls="--")
    ax.set_yticks(yt, yl, fontsize=8)
    ax.set_xlabel("rank-biserial r (High − Low) ve %95 GA")
    ax.set_title("b  Entropi bulguları: ana analiz ve dışlama varyantları")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def r4_rapor(res: dict) -> str:
    L = ["", "R4 — YALNIZ TÜMÖR ÇEKİRDEKLERİ (morfolojik kural; doğrulanmış sınıflandırıcı değil)",
         f"Hasta: {len(res['d'])} · h_mean mevcut: {int(res['d']['r4_has_h_mean'].min())}", "",
         "Dışlanan çekirdek oranı (medyan %):"]
    for _, r in res["frac"].iterrows():
        L.append(f"  {r['varyant']:<44} Low {r['Low medyan %']:.1f} · High {r['High medyan %']:.1f} ({ks.fmt_p(r['p_MWU'])}) · "
                 f"G1 {r['G1 medyan %']:.1f} G2 {r['G2 medyan %']:.1f} G3 {r['G3 medyan %']:.1f} G4 {r['G4 medyan %']:.1f}")
    L += ["", "Entropi bulguları (r, Spearman ρ, düzeltilmiş OR/SD):"]
    for _, r in res["ent"].iterrows():
        orr = "" if pd.isna(r["OR_düzeltilmiş/SD"]) else \
            f" · OR {r['OR_düzeltilmiş/SD']:.2f} [{r['OR_GA_low']:.2f}–{r['OR_GA_high']:.2f}] {ks.fmt_p(r['p_OR'])}"
        L.append(f"  {r['analiz'][:40]:<40} {r['ölçüt']:<22} r={r['r (High−Low)']:+.2f} {ks.fmt_p(r['p_MWU'])} · "
                 f"ρ={r['rho_grade']:+.2f}{orr}")
    L += ["", "Uzamsal heterojenlik (klasik → klasik + tile entropisi SD, p90):"]
    for _, r in res["sp"].iterrows():
        L.append(f"  {r['analiz'][:40]:<40} CV-AUC {r['CV_AUC_klasik']:.3f} → {r['CV_AUC_klasik+uzamsal']:.3f} "
                 f"(Δ {r['ΔCV_AUC']:+.3f}) · LR χ²={r['LR_chi2']:.2f}, {ks.fmt_p(r['p_LR'])}")
    return "\n".join(L)


def r4_calistir(ana: pd.DataFrame, results: Path, cfg: kp.Config, out_xlsx: Path, out_png: Path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    tc = vaka_dongusu(ana, results, cfg, tumor_cekirdek_olcutleri, "R4 tümör çekirdekleri")
    d = ana.merge(tc, on="case_id", how="inner")
    if d["r4_has_h_mean"].min() == 0:
        print("  UYARI: bazı vakalarda hematoksilen yoğunluğu (h_mean) yok → 'koyu' şartı uygulanamadı.")
    print(f"  Kontrol: kümeli küçük çekirdek oranı medyan %{100 * d['r4_kumeli_kucuk_oran'].median():.1f}; "
          f"A varyantında çekirdek dışlanan vaka: {(d['r4_A_dislanan_oran'] > 0).sum()}/{len(d)}")
    if (d["r4_A_dislanan_oran"] > 0).sum() == 0:
        print("  UYARI: A varyantı hiçbir çekirdeği dışlamadı — Claude'a bildirin.")
    res = r4_analiz(d)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"açıklama": [
            "R4: Yalnız tümör çekirdekleri — lenfosit benzeri (küçük, yuvarlak, koyu, kümeli) ve iğsi çekirdekler dışlanarak.",
            f"Kurallar: alan ≤ {LYM_AREA_MAX} µm², eksantrisite ≤ {LYM_ECC_MAX}, H yoğunluğu ≥ slaytın {LYM_H_PCT}. persentili, "
            f"{LYM_R_UM} µm içinde ≥{LYM_K} küçük komşu (kümelenme). İğsi: eksantrisite ≥ 0.95 ve kısa eksen ≤ 2.5 µm.",
            "Doğrulanmış hücre sınıflandırıcısı değildir; duyarlılık analizi olarak raporlanır.",
            "Düzeltilmiş OR: entropi + log10(kalan çekirdek sayısı) + bulanık doku oranı + analiz edilen doku alanı."]}).to_excel(
            xw, sheet_name="00_bilgi", index=False)
        res["frac"].to_excel(xw, sheet_name="01_dislanan_oran", index=False)
        res["ent"].to_excel(xw, sheet_name="02_entropi_bulgulari", index=False)
        res["sp"].to_excel(xw, sheet_name="03_uzamsal", index=False)
        keep = ["case_id", "grade"] + [c for c in d.columns if c.startswith("r4_") or any(
            c.endswith(f"_{v}") for v in R4_VARIANTS)]
        d[keep].to_excel(xw, sheet_name="04_vaka_bazinda", index=False)
    r4_sekil(res, out_png)
    txt = r4_rapor(res)
    print(txt)
    return res, txt


# -----------------------------------------------------------------------------------------------
# R3 — Prognoz: TCGA-KIRC genel (OS) ve hastalığa özgü (DSS) sağkalım
# -----------------------------------------------------------------------------------------------
# Kaynak: TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR; Liu ve ark., Cell 2018) — TCGA için önerilen,
# küratörlü sağkalım sonlanımları (OS, DSS, PFI). GDC'den tek dosya olarak indirilir.
CDR_URL = "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
CDR_NAME = "TCGA-CDR-SupplementalTableS1.xlsx"
R3_EXPOSURES = [("area_um2_entropy", "Alan entropisi"), ("perimeter_um_entropy", "Çevre entropisi"),
                ("patch_nuc_area_entropy_std", "Tile alan entropisi SD (uzamsal)"),
                ("patch_nuc_area_entropy_p90", "Tile alan entropisi p90"),
                ("area_um2_std", "Nükleer alan SD (klasik)"), ("area_um2_median", "Medyan nükleer alan (klasik)")]
R3_ENDPOINTS = [("OS", "Genel sağkalım"), ("DSS", "Hastalığa özgü sağkalım"), ("PFI", "Progresyonsuz aralık")]


def cdr_indir(dst_dir: Path) -> Path:
    import urllib.request
    dst = dst_dir / CDR_NAME
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"  TCGA-CDR indiriliyor (~3 MB): {CDR_URL}")
    try:
        with urllib.request.urlopen(CDR_URL, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"İndirilemedi ({exc}). Tarayıcıda https://gdc.cancer.gov/about-data/publications/pancanatlas sayfasından "
                         f"'TCGA-CDR-SupplementalTableS1.xlsx' dosyasını indirip şununla çalıştırın:\n"
                         f"  python scripts/proje.py sagkalim --dosya ~/Downloads/{CDR_NAME}")
    return dst


def cdr_oku(path: Path) -> pd.DataFrame:
    x = pd.ExcelFile(path)
    sh = "TCGA-CDR" if "TCGA-CDR" in x.sheet_names else x.sheet_names[0]
    c = x.parse(sh)
    need = {"bcr_patient_barcode", "type", "OS", "OS.time", "DSS", "DSS.time"}
    if not need <= set(c.columns):
        raise SystemExit(f"{path.name} beklenen TCGA-CDR dosyası değil (eksik sütunlar: {need - set(c.columns)}).")
    c = c[c["type"] == "KIRC"].copy()
    c["case_id"] = c["bcr_patient_barcode"].astype(str).str[:12]
    st = c["ajcc_pathologic_tumor_stage"].astype(str).str.upper()
    num = np.select([st.str.contains("IV"), st.str.contains("III"), st.str.contains("II"), st.str.contains("STAGE I")],
                    [4, 3, 2, 1], default=np.nan)
    c["stage_num"] = num
    c["stage_late"] = np.where(np.isnan(num), np.nan, (num >= 3).astype(float))
    c["age"] = pd.to_numeric(c["age_at_initial_pathologic_diagnosis"], errors="coerce")
    for e, _ in R3_ENDPOINTS:
        c[e] = pd.to_numeric(c.get(e), errors="coerce")
        c[f"{e}.time"] = pd.to_numeric(c.get(f"{e}.time"), errors="coerce")
    return c[["case_id", "stage_num", "stage_late", "age", "gender", "histological_grade"] +
             [x for e, _ in R3_ENDPOINTS for x in (e, f"{e}.time")]]


def _cox(d, cols, dur, ev):
    from lifelines import CoxPHFitter
    dd = d[[dur, ev, *cols]].dropna()
    dd = dd[dd[dur] > 0]
    X = dd.copy()
    for c in cols:
        if X[c].nunique() > 2:   # sürekli değişkenler 1 SD başına (yaş: 10 yıl başına)
            X[c] = (X[c] - X[c].mean()) / (10.0 if c == "age" else X[c].std(ddof=0))
    m = CoxPHFitter().fit(X, duration_col=dur, event_col=ev)
    return m, X


def _cindex_boot(d, base, full, dur, ev, rng, n=200):
    from lifelines.utils import concordance_index
    dd = d[[dur, ev, *full]].dropna()
    dd = dd[dd[dur] > 0].reset_index(drop=True)
    def c_of(sub, cols):
        m, X = _cox(sub, cols, dur, ev)
        return concordance_index(X[dur], -m.predict_partial_hazard(X), X[ev])
    c0, c1 = c_of(dd, base), c_of(dd, full)
    diffs = []
    for _ in range(n):   # iyimserliği azaltmak için: bootstrap örneğinde kur, out-of-bag'de değerlendir
        i = rng.integers(0, len(dd), len(dd))
        oob = np.setdiff1d(np.arange(len(dd)), i)
        b, o = dd.iloc[i], dd.iloc[oob]
        if b[ev].sum() < 10 or o[ev].sum() < 5:
            continue
        try:
            def oob_c(cols):
                m, X = _cox(b, cols, dur, ev)
                Xo = o[[dur, ev, *cols]].copy()
                for c in cols:
                    if b[c].nunique() > 2:
                        Xo[c] = (Xo[c] - b[c].mean()) / (10.0 if c == "age" else b[c].std(ddof=0))
                return concordance_index(Xo[dur], -m.predict_partial_hazard(Xo), Xo[ev])
            diffs.append(oob_c(full) - oob_c(base))
        except Exception:  # noqa: BLE001
            continue
    diffs = np.array(diffs)
    return {"C_base": float(c0), "C_full": float(c1), "ΔC_görünen": float(c1 - c0),
            "ΔC_OOB_ort": float(diffs.mean()) if len(diffs) else np.nan,
            "ΔC_OOB_GA_low": float(np.percentile(diffs, 2.5)) if len(diffs) else np.nan,
            "ΔC_OOB_GA_high": float(np.percentile(diffs, 97.5)) if len(diffs) else np.nan}


def r3_analiz(d: pd.DataFrame, rng=None) -> dict:
    rng = rng or np.random.default_rng(ks.SEED)
    from lifelines.statistics import logrank_test, proportional_hazard_test
    d = d.assign(grade_high=d["y_high"].astype(float))
    base = ["stage_late", "grade_high", "age"]
    desc, cox_rows, cidx, km = [], [], [], []
    for e, elab in R3_ENDPOINTS:
        dur, ev = f"{e}.time", e
        sub = d.dropna(subset=[dur, ev])
        desc.append({"sonlanım": elab, "n": len(sub), "olay": int(sub[ev].sum()),
                     "medyan_izlem_ay": float(sub[dur].median() / 30.44)})
        for x, xl in R3_EXPOSURES:
            if x not in d:
                continue
            for model, cols in (("tek değişkenli", [x]), ("düzeltilmiş (evre III–IV, grade High, yaş)", [x, *base])):
                try:
                    m, X = _cox(d, cols, dur, ev)
                    s = m.summary.loc[x]
                    ph = proportional_hazard_test(m, X, time_transform="rank").summary.loc[x, "p"]
                    cox_rows.append({"sonlanım": e, "ölçüt": xl, "değişken": x, "model": model, "n": int(len(X)),
                                     "olay": int(X[ev].sum()), "HR_per_SD": float(s["exp(coef)"]),
                                     "HR_GA_low": float(s["exp(coef) lower 95%"]), "HR_GA_high": float(s["exp(coef) upper 95%"]),
                                     "p": float(s["p"]), "PH_varsayımı_p": float(ph)})
                except Exception as exc:  # noqa: BLE001
                    cox_rows.append({"sonlanım": e, "ölçüt": xl, "değişken": x, "model": model, "hata": str(exc)})
            if e in ("OS", "DSS"):
                print(f"  {e} · {xl}: C-indeksi bootstrap…")
                try:
                    cidx.append({"sonlanım": e, "ölçüt": xl, **_cindex_boot(d, base, [*base, x], dur, ev, rng)})
                except Exception as exc:  # noqa: BLE001
                    cidx.append({"sonlanım": e, "ölçüt": xl, "hata": str(exc)})
                s2 = d.dropna(subset=[dur, ev, x])
                hi = s2[x] >= s2[x].median()
                lr = logrank_test(s2.loc[hi, dur], s2.loc[~hi, dur], s2.loc[hi, ev], s2.loc[~hi, ev])
                km.append({"sonlanım": e, "ölçüt": xl, "değişken": x, "logrank_p": float(lr.p_value),
                           "olay_yüksek": int(s2.loc[hi, ev].sum()), "olay_düşük": int(s2.loc[~hi, ev].sum())})
    # temel modelin kendisi (referans)
    basem = []
    for e, _ in R3_ENDPOINTS[:2]:
        m, X = _cox(d, base, f"{e}.time", e)
        for c in base:
            s = m.summary.loc[c]
            basem.append({"sonlanım": e, "değişken": c, "HR": float(s["exp(coef)"]), "HR_GA_low": float(s["exp(coef) lower 95%"]),
                          "HR_GA_high": float(s["exp(coef) upper 95%"]), "p": float(s["p"]), "n": int(len(X)), "olay": int(X[e].sum())})
    return {"d": d, "desc": pd.DataFrame(desc), "cox": pd.DataFrame(cox_rows), "cidx": pd.DataFrame(cidx),
            "km": pd.DataFrame(km), "base": pd.DataFrame(basem)}


def r3_sekil(res: dict, path: Path):
    plt = ks.setup_style()
    from lifelines import KaplanMeierFitter
    d = res["d"]
    pairs = [("OS", "area_um2_entropy", "Alan entropisi"), ("OS", "patch_nuc_area_entropy_std", "Tile entropisi SD"),
             ("DSS", "area_um2_entropy", "Alan entropisi"), ("DSS", "patch_nuc_area_entropy_std", "Tile entropisi SD")]
    fig, axs = plt.subplots(1, 5, figsize=(24, 4.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]})
    for ax, (e, x, xl) in zip(axs[:4], pairs):
        s = d.dropna(subset=[f"{e}.time", e, x])
        hi = s[x] >= s[x].median()
        for mask, lab, c in ((~hi, "düşük (< medyan)", ks.C_LOW), (hi, "yüksek (≥ medyan)", ks.C_HIGH)):
            KaplanMeierFitter().fit(s.loc[mask, f"{e}.time"] / 30.44, s.loc[mask, e], label=lab).plot_survival_function(
                ax=ax, color=c, ci_alpha=0.12)
        p = res["km"].query("sonlanım == @e and değişken == @x")["logrank_p"].iloc[0]
        ax.set_title(f"{e} · {xl}\nlog-rank {ks.fmt_p(p)}", fontsize=10)
        ax.set_xlabel("ay"); ax.set_ylabel("sağkalım olasılığı"); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, loc="lower left")
    ax = axs[4]
    cx = res["cox"].dropna(subset=["HR_per_SD"])
    cx = cx[cx["model"].str.startswith("düzeltilmiş") & cx["sonlanım"].isin(["OS", "DSS"])]
    y, yt, yl = 0, [], []
    for e, col in (("OS", ks.C_HIGH), ("DSS", ks.C_LOW)):
        for _, r in cx[cx["sonlanım"] == e].iterrows():
            ax.errorbar(r["HR_per_SD"], y, xerr=[[r["HR_per_SD"] - r["HR_GA_low"]], [r["HR_GA_high"] - r["HR_per_SD"]]],
                        fmt="o", color=col, capsize=3)
            yt.append(y); yl.append(f"{e} · {r['ölçüt']}")
            y -= 1
        y -= 0.6
    ax.axvline(1, color=ks.INK2, ls="--", lw=1)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    lo_ = min(0.5, float(cx["HR_GA_low"].min()) * 0.9) if len(cx) else 0.5
    hi_ = max(2.0, float(cx["HR_GA_high"].max()) * 1.1) if len(cx) else 2.0
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6) if lo_ <= t <= hi_]
    ax.set_xlim(lo_, hi_)
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(yt, yl, fontsize=8.5)
    ax.set_xlabel("HR (1 SD başına), evre, grade ve yaşa göre düzeltilmiş")
    ax.set_title("Düzeltilmiş Cox modelleri")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def r3_rapor(res: dict) -> str:
    L = ["", "R3 — PROGNOZ (TCGA-CDR sonlanımları; Cox, HR 1 SD başına)", ""]
    for _, r in res["desc"].iterrows():
        L.append(f"  {r['sonlanım']:<26} n={r['n']}, olay={r['olay']}, medyan izlem {r['medyan_izlem_ay']:.0f} ay")
    L += ["", "Temel model (evre III–IV, grade High, yaş/10 yıl):"]
    for _, r in res["base"].iterrows():
        L.append(f"  {r['sonlanım']:<4} {r['değişken']:<11} HR {r['HR']:.2f} [{r['HR_GA_low']:.2f}–{r['HR_GA_high']:.2f}] {ks.fmt_p(r['p'])}")
    L += ["", "Cox modelleri:"]
    for _, r in res["cox"].iterrows():
        if pd.isna(r.get("HR_per_SD", np.nan)):
            L.append(f"  {r['sonlanım']:<4} {r['ölçüt']:<34} {r['model'][:13]:<13} HATA: {r.get('hata')}")
            continue
        L.append(f"  {r['sonlanım']:<4} {r['ölçüt']:<34} {r['model'][:13]:<13} HR {r['HR_per_SD']:.2f} "
                 f"[{r['HR_GA_low']:.2f}–{r['HR_GA_high']:.2f}] {ks.fmt_p(r['p'])} (olay {r['olay']}; PH p={r['PH_varsayımı_p']:.2f})")
    L += ["", "C-indeksi (temel → temel + ölçüt; ΔC out-of-bag bootstrap):"]
    for _, r in res["cidx"].iterrows():
        if "C_base" not in r or pd.isna(r.get("C_base")):
            continue
        L.append(f"  {r['sonlanım']:<4} {r['ölçüt']:<34} {r['C_base']:.3f} → {r['C_full']:.3f} · ΔC_OOB {r['ΔC_OOB_ort']:+.3f} "
                 f"[{r['ΔC_OOB_GA_low']:+.3f}, {r['ΔC_OOB_GA_high']:+.3f}]")
    L += ["", "Kaplan–Meier (medyanla ikiye bölme, log-rank):"]
    for _, r in res["km"].iterrows():
        L.append(f"  {r['sonlanım']:<4} {r['ölçüt']:<34} {ks.fmt_p(r['logrank_p'])} (olay yüksek {r['olay_yüksek']} / düşük {r['olay_düşük']})")
    return "\n".join(L)


def r3_calistir(ana: pd.DataFrame, cdr_path: Path, out_xlsx: Path, out_png: Path):
    warnings.filterwarnings("ignore")
    try:
        import lifelines  # noqa: F401
    except ImportError:
        raise SystemExit("lifelines paketi gerekli:  pip install lifelines")
    cdr = cdr_oku(cdr_path)
    d = ana.merge(cdr, on="case_id", how="left")
    miss = d["OS.time"].isna().sum()
    print(f"  TCGA-CDR eşleşmesi: {len(d) - miss}/{len(d)} hasta (eksik {miss})")
    gmis = d.dropna(subset=["histological_grade"])
    gm = (gmis["histological_grade"].astype(str).str.upper() != gmis["grade"].astype(str).str.upper()).sum()
    print(f"  Kontrol: Excel grade ile TCGA-CDR grade uyuşmayan hasta: {gm}")
    res = r3_analiz(d)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"açıklama": [
            "R3: TCGA-KIRC sağkalım. Kaynak: TCGA-CDR (Liu ve ark., Cell 2018;173:400–416), GDC'den indirildi.",
            "Sonlanımlar: OS (genel), DSS (hastalığa özgü), PFI (progresyonsuz aralık, ikincil).",
            "Cox: sürekli değişkenler 1 SD başına; yaş 10 yıl başına. Düzeltme: evre III–IV (vs I–II), grade High (vs Low), yaş.",
            "PH_varsayımı_p: Schoenfeld artıkları (rank zaman dönüşümü); p<0.05 orantılı tehlike varsayımının ihlalini düşündürür.",
            "ΔC_OOB: bootstrap örneğinde kurulan modelin out-of-bag hastalarda C-indeksi farkı (iyimserlik düzeltmeli)."]}).to_excel(
            xw, sheet_name="00_bilgi", index=False)
        res["desc"].to_excel(xw, sheet_name="01_sonlanimlar", index=False)
        res["base"].to_excel(xw, sheet_name="02_temel_model", index=False)
        res["cox"].to_excel(xw, sheet_name="03_cox", index=False)
        res["cidx"].to_excel(xw, sheet_name="04_c_indeksi", index=False)
        res["km"].to_excel(xw, sheet_name="05_kaplan_meier", index=False)
    r3_sekil(res, out_png)
    txt = r3_rapor(res)
    print(txt)
    return res, txt


# -----------------------------------------------------------------------------------------------
# R2 — Dış doğrulama: CPTAC-CCRCC (NCI Imaging Data Commons; DICOM-WSI, herkese açık)
# -----------------------------------------------------------------------------------------------
# Görüntüler: IDC 'cptac_ccrcc' koleksiyonu (idc-index ile, giriş gerekmez).
# Grade ve sağkalım: GDC CPTAC-3 projesi (diagnoses.tumor_grade).
# Envanter (Kasım 2026): IDC'deki 782 CPTAC-CCRCC slaytının TAMAMI 20x (0.49 µm/px). TCGA'daki 0.25 µm/px kuralı
# uygulanamaz ve pipeline yapay büyütme yapmaz. Bu nedenle sonuçlar görülmeden önce belirlenen kural:
#   • CPTAC, pipeline'ın 0.5 µm/px modunda analiz edilir (tile fiziksel boyutu aynı: 256 µm; StarDist 2x ölçekli tahmin,
#     bin kenarları µm² cinsinden aynı). TCGA katsayıları CPTAC'a taşınmaz (çözünürlük farkı);
#     doğrulama = aynı ilişkilerin bağımsız kohortta, kohort içinde tekrarlanması.
#   • Her hastadan 1 tümör H&E slaytı (en büyük piksel matrisi), grade G1–G4, berrak hücreli RCC.
GDC_CASES = "https://api.gdc.cancer.gov/cases"
CPTAC_MPP = 0.5


def _gdc_cptac_klinik() -> pd.DataFrame:
    import urllib.request
    fields = ["submitter_id", "diagnoses.tumor_grade", "diagnoses.primary_diagnosis", "diagnoses.morphology", "diagnoses.ajcc_pathologic_stage",
              "diagnoses.age_at_diagnosis", "diagnoses.days_to_last_follow_up", "diagnoses.tissue_or_organ_of_origin",
              "demographic.vital_status", "demographic.days_to_death", "disease_type", "primary_site"]
    filt = {"op": "and", "content": [{"op": "in", "content": {"field": "project.project_id", "value": ["CPTAC-3"]}},
                                     {"op": "in", "content": {"field": "primary_site", "value": ["Kidney"]}}]}
    body = json.dumps({"filters": filt, "fields": ",".join(fields), "format": "JSON", "size": 5000}).encode()
    req = urllib.request.Request(GDC_CASES, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        hits = json.loads(r.read())["data"]["hits"]
    rows = []
    for h in hits:
        dx = [x for x in (h.get("diagnoses") or []) if x.get("tumor_grade") not in (None, "Not Reported", "Unknown", "unknown")]
        d0 = (dx or h.get("diagnoses") or [{}])[0]
        dm = h.get("demographic") or {}
        rows.append({"PatientID": h.get("submitter_id"), "tumor_grade": d0.get("tumor_grade"),
                     "primary_diagnosis": d0.get("primary_diagnosis"), "morphology": d0.get("morphology"),
                     "stage": d0.get("ajcc_pathologic_stage"),
                     "age_days": d0.get("age_at_diagnosis"), "days_to_last_follow_up": d0.get("days_to_last_follow_up"),
                     "vital_status": dm.get("vital_status"), "days_to_death": dm.get("days_to_death"),
                     "disease_type": h.get("disease_type")})
    return pd.DataFrame(rows)


def _idc_cptac_slaytlar() -> pd.DataFrame:
    try:
        from idc_index import IDCClient
    except ImportError:
        raise SystemExit("idc-index paketi gerekli:  pip install idc-index")
    c = IDCClient()
    c.fetch_index("sm_index")
    df = c.sql_query("SELECT i.PatientID, i.series_size_MB, s.* FROM index i "
                     "JOIN sm_index s ON i.SeriesInstanceUID = s.SeriesInstanceUID "
                     "WHERE i.collection_id = 'cptac_ccrcc'")
    for col in df.columns:   # dizi sütunlarını metne çevir
        if df[col].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any():
            df[col] = df[col].map(lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple, np.ndarray)) else v)
    return df


def _sutun(df, *adaylar):
    for a in adaylar:
        for c in df.columns:
            if c.lower() == a.lower():
                return c
    for a in adaylar:
        for c in df.columns:
            if a.lower() in c.lower():
                return c
    return None


def cptac_envanter(out_xlsx: Path) -> pd.DataFrame:
    print("  IDC'den CPTAC-CCRCC slayt listesi alınıyor…")
    sl = _idc_cptac_slaytlar()
    print(f"  {len(sl)} slayt serisi, {sl['PatientID'].nunique()} hasta")
    print("  GDC'den CPTAC-3 böbrek klinik verisi alınıyor…")
    kl = _gdc_cptac_klinik()
    ps = _sutun(sl, "min_PixelSpacing_2sf", "PixelSpacing")
    mag = _sutun(sl, "ObjectiveLensPower")
    tis = _sutun(sl, "primaryAnatomicStructureModifier_CodeMeaning")
    stn = _sutun(sl, "staining_usingSubstance_CodeMeaning")
    emb = _sutun(sl, "embeddingMedium_CodeMeaning")
    cols_ = _sutun(sl, "max_TotalPixelMatrixColumns", "TotalPixelMatrixColumns")
    rows_ = _sutun(sl, "max_TotalPixelMatrixRows", "TotalPixelMatrixRows")
    d = sl.copy()
    v = pd.to_numeric(d[ps], errors="coerce") if ps else pd.Series(np.nan, index=d.index)
    d["mpp_um"] = np.where(v < 0.01, v * 1000, v)   # DICOM PixelSpacing mm cinsindendir
    d["is_40x"] = d["mpp_um"] <= 0.275
    d["tissue_type"] = d[tis].astype(str) if tis else "?"
    d["is_tumor"] = d["tissue_type"].str.contains("Neoplasm|Tumor|tumor", regex=True)
    d["stain"] = d[stn].astype(str) if stn else "?"
    d["is_he"] = d["stain"].str.contains("hematoxylin", case=False) & d["stain"].str.contains("eosin", case=False) if stn else True
    d["embedding"] = d[emb].astype(str) if emb else "?"
    d["objective"] = d[mag] if mag else np.nan
    d["pixels"] = (pd.to_numeric(d[cols_], errors="coerce") * pd.to_numeric(d[rows_], errors="coerce")) if cols_ and rows_ else d["series_size_MB"]
    d = d.merge(kl, on="PatientID", how="left")
    gmap = {"G1": "G1", "G2": "G2", "G3": "G3", "G4": "G4"}
    d["grade"] = d["tumor_grade"].map(lambda g: gmap.get(str(g).strip().upper()) if pd.notna(g) else None)
    # Histoloji: GDC CPTAC-3'te tanı çoğunlukla "Renal cell carcinoma, NOS" olarak kodlu. Koleksiyon (CPTAC-CCRCC) berrak hücreli
    # RCC kohortudur; bu nedenle yalnız AÇIKÇA başka bir alt tip olarak kodlananlar dışlanır (papiller, kromofob, toplayıcı kanal…).
    morph = d["morphology"].astype(str) if "morphology" in d else pd.Series("", index=d.index)
    dx = d["primary_diagnosis"].astype(str)
    non_cc = morph.str.match(r"^(8260|8317|8319|8270|8290|8480|8959|8963|8050)") | dx.str.contains(
        "papillary|chromophobe|collecting|oncocytoma|sarcoma|unclassified|translocation", case=False)
    d["is_ccrcc"] = ~non_cc
    elig = d[d["is_tumor"] & d["is_he"] & d["grade"].notna() & d["is_ccrcc"]]
    sec = elig.sort_values("pixels", ascending=False).drop_duplicates("PatientID")
    sec = sec.assign(grade_group=np.where(sec["grade"].isin(["G1", "G2"]), "Low", "High"))
    ozet = {
        "slayt serisi (toplam)": len(d), "hasta (toplam)": d["PatientID"].nunique(),
        "40x slayt": int(d["is_40x"].sum()),
        "tümör + H&E + grade + ccRCC slayt": len(elig), "seçilen hasta (1 slayt/hasta)": sec["PatientID"].nunique(),
        "seçilen: G1/G2/G3/G4": "/".join(str(int((sec["grade"] == g).sum())) for g in ("G1", "G2", "G3", "G4")),
        "seçilen: Low/High": f"{int((sec.grade_group == 'Low').sum())}/{int((sec.grade_group == 'High').sum())}",
        "seçilenlerin toplam boyutu (GB)": round(float(pd.to_numeric(sec["series_size_MB"], errors="coerce").sum()) / 1024, 1),
        "grade bilgisi olmayan tümör hastası": int(d[d["is_tumor"] & d["grade"].isna()]["PatientID"].nunique()),
        "açıkça ccRCC dışı kodlu tümör hastası": int(d[d["is_tumor"] & ~d["is_ccrcc"]]["PatientID"].nunique()),
        "morfoloji kodları (hasta)": "; ".join(f"{k}: {v}" for k, v in d.drop_duplicates("PatientID")["morphology"].astype(str)
                                               .value_counts().items()) if "morphology" in d else "—",
    }
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"ölçüt": list(ozet), "değer": list(ozet.values())}).to_excel(xw, sheet_name="00_ozet", index=False)
        for nm, s in (("01_buyutme", d.groupby(["objective", "mpp_um"], dropna=False).size()),
                      ("02_doku_tipi", d.groupby("tissue_type", dropna=False).size()),
                      ("03_gomme_boya", d.groupby(["embedding", "stain"], dropna=False).size()),
                      ("04_grade_tani", d.drop_duplicates("PatientID").groupby(["primary_diagnosis", "tumor_grade"], dropna=False).size())):
            s.rename("n").reset_index().to_excel(xw, sheet_name=nm, index=False)
        sec.to_excel(xw, sheet_name="10_secilen", index=False)
        d.to_excel(xw, sheet_name="20_tum_slaytlar", index=False)
    L = ["", "R2 — CPTAC-CCRCC ENVANTERİ"] + [f"  {k}: {v}" for k, v in ozet.items()]
    L += ["", "  Büyütme / piksel aralığı:"] + [f"    {k}: {n}" for k, n in d.groupby(["objective", "mpp_um"], dropna=False).size().items()]
    L += ["", "  Doku tipi:"] + [f"    {k}: {n}" for k, n in d.groupby("tissue_type", dropna=False).size().items()]
    L += ["", "  Gömme ortamı / boya:"] + [f"    {k}: {n}" for k, n in d.groupby(["embedding", "stain"], dropna=False).size().items()]
    L += ["", "  Tanı / grade (hasta):"] + [f"    {k}: {n}" for k, n in d.drop_duplicates("PatientID").groupby(
        ["primary_diagnosis", "tumor_grade"], dropna=False).size().items()]
    n_ = sec["PatientID"].nunique()
    nl, nh = int((sec.grade_group == "Low").sum()), int((sec.grade_group == "High").sum())
    karar = (f"KARAR: {n_} hasta (Low {nl} / High {nh}) 0.5 µm/px modunda analiz edilecek."
             if min(nl, nh) >= 20 else f"KARAR BEKLİYOR: gruplardan biri < 20 hasta (Low {nl} / High {nh}). Claude'a gönderin.")
    L += ["", karar]
    txt = "\n".join(L)
    print(txt)
    return sec, txt


def _idc_indir(uid: str, dl: Path) -> Path:
    from idc_index import IDCClient
    c = IDCClient()
    try:
        c.download_from_selection(downloadDir=str(dl), seriesInstanceUID=[uid], dirTemplate="%SeriesInstanceUID",
                                  show_progress_bar=False)
    except TypeError:
        c.download_from_selection(downloadDir=str(dl), seriesInstanceUID=[uid])
    dcm = sorted(dl.rglob("*.dcm"))
    if not dcm:
        raise RuntimeError("indirilen .dcm dosyası yok")
    return dcm[0]


def cptac_analiz(sec_xlsx: Path, out_dir: Path, tmp_dir: Path, cfg: kp.Config, limit: int | None = None):
    """Slaytları sırayla analiz eder; bir sonraki slayt arka planda önceden indirilir (en fazla 2 slayt diskte)."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor
    sec = pd.read_excel(sec_xlsx, sheet_name="10_secilen")
    if abs(cfg.target_mpp - CPTAC_MPP) > 1e-9:
        raise SystemExit(f"CPTAC {CPTAC_MPP} µm/px'te analiz edilmeli (cfg.target_mpp={cfg.target_mpp}).")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    todo = sec if limit is None else sec.head(limit)
    jobs = []
    for r in todo.itertuples():
        sid = f"{r.PatientID}_{str(r.SeriesInstanceUID)[-8:]}"
        if not (out_dir / "cases" / sid / "case_result.json").exists():
            jobs.append((r, sid))
    print(f"  Analiz edilecek: {len(jobs)} hasta (zaten bitmiş: {len(todo) - len(jobs)})")
    if not jobs:
        return
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_idc_indir, jobs[0][0].SeriesInstanceUID, tmp_dir / jobs[0][1])
        for k, (r, sid) in enumerate(jobs):
            dl = tmp_dir / sid
            nxt = None
            try:
                path = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [{k + 1}/{len(jobs)}] {r.PatientID}: İNDİRME HATASI: {exc}")
                path = None
            if k + 1 < len(jobs):   # bir sonrakini şimdiden indirmeye başla
                r2, sid2 = jobs[k + 1]
                nxt = ex.submit(_idc_indir, r2.SeriesInstanceUID, tmp_dir / sid2)
            if path is not None:
                try:
                    res = kp.analyze_case(r.PatientID, path, out_dir, cfg=cfg,
                                          meta={"grade": r.grade, "grade_group": r.grade_group, "slide_uuid": sid,
                                                "slide_barcode": sid})
                    kalan = (time.time() - t0) / (k + 1) * (len(jobs) - k - 1) / 3600
                    print(f"  [{k + 1}/{len(jobs)}] {r.PatientID} ({r.grade}) → {res.get('analysis_status')} · çekirdek "
                          f"{res.get('n_nuclei')} · alan entropisi {res.get('area_um2_entropy')} · kalan ~{kalan:.1f} sa")
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{k + 1}/{len(jobs)}] {r.PatientID}: ANALİZ HATASI: {exc}")
            shutil.rmtree(dl, ignore_errors=True)
            fut = nxt


def cptac_istatistik(out_dir: Path, cfg: kp.Config, out_xlsx: Path, r1_xlsx: Path, r1_png: Path, r1_csv: Path) -> str:
    """CPTAC'ta (kohort içinde) birincil ilişkiler + uzamsal artımsal değer + uzamsal özgüllük (R1)."""
    warnings.filterwarnings("ignore")
    p = out_dir / "case_level_results.csv"
    if not p.exists():
        raise SystemExit(f"{p} yok. Önce: python scripts/proje.py cptac-analiz")
    a = pd.read_csv(p)
    n_all = len(a)
    a = a[a["analysis_status"].isin(kp.SUCCESS_STATUSES) & a["grade"].isin(["G1", "G2", "G3", "G4"])].copy()
    a = a.drop_duplicates("case_id")
    a["y_high"] = a["grade"].isin(["G3", "G4"]).astype(int)
    a["grade_num"] = a["grade"].str[1].astype(int)
    rng = np.random.default_rng(ks.SEED)
    L = ["", "R2 — CPTAC-CCRCC DIŞ DOĞRULAMA (0.5 µm/px; kohort içi)",
         f"Analiz edilen hasta: {len(a)} / {n_all} (Low {int((a.y_high == 0).sum())}, High {int((a.y_high == 1).sum())}); "
         f"G1/G2/G3/G4: {'/'.join(str(int((a.grade == g).sum())) for g in ('G1', 'G2', 'G3', 'G4'))}", ""]
    rows = []
    for c, lab in ks.PRIMARY + [("log2_area_entropy", "log2(alan) entropisi"), ("area_um2_std", "Nükleer alan SD (µm²)"),
                                ("area_um2_median", "Medyan nükleer alan (µm²)")]:
        if c not in a:
            continue
        mw, sp = ks.mann_whitney(a, c, rng), ks.spearman_grade(a, c, rng)
        rows.append({"ölçüt": lab, "değişken": c, "Low medyan": mw["median_low"], "High medyan": mw["median_high"],
                     "r (High−Low)": mw["rank_biserial_r"], "r_GA": f"[{mw['r_ci_low']:+.2f}, {mw['r_ci_high']:+.2f}]",
                     "p_MWU": mw["p"], "rho_grade": sp["rho"], "p_spearman": sp["p"], "AUC_görünen": mw["auc_apparent"]})
    prim = pd.DataFrame(rows)
    prim["p_MWU_holm (4 birincil)"] = np.nan
    m4 = prim["değişken"].isin([c for c, _ in ks.PRIMARY])
    prim.loc[m4, "p_MWU_holm (4 birincil)"] = ks.holm(prim.loc[m4, "p_MWU"].to_numpy())
    L.append("Birincil ve karşılaştırma ölçütleri (High vs Low):")
    for _, r in prim.iterrows():
        L.append(f"  {r['ölçüt']:<28} Low {r['Low medyan']:.3f} · High {r['High medyan']:.3f} · r={r['r (High−Low)']:+.2f} "
                 f"{r['r_GA']} {ks.fmt_p(r['p_MWU'])} · ρ={r['rho_grade']:+.2f} · AUC {r['AUC_görünen']:.3f}")
    # Önceden belirlenen duyarlılık: dondurulmuş kesit / odak artefaktı şüphesi düşük vakalar
    q = a[(a["focus_blurry_tissue_fraction"] <= 0.10) & (a["n_nuclei"] >= 20000)]
    L += ["", f"Duyarlılık (bulanık doku ≤ %10 ve ≥ 20.000 çekirdek; n={len(q)}, Low {int((q.y_high == 0).sum())} / "
              f"High {int((q.y_high == 1).sum())}):"]
    sens = []
    for c, lab in ks.PRIMARY:
        mw = ks.mann_whitney(q, c, rng)
        if mw is None:
            L.append(f"  {lab:<28} yetersiz vaka"); continue
        sens.append({"ölçüt": lab, "n": len(q), "r (High−Low)": mw["rank_biserial_r"], "p_MWU": mw["p"]})
        L.append(f"  {lab:<28} r={mw['rank_biserial_r']:+.2f} {ks.fmt_p(mw['p'])}")
    sens = pd.DataFrame(sens)
    # Düzeltilmiş lojistik (TCGA ile aynı kovaryatlar)
    covs = [c for c in ("focus_blurry_tissue_fraction", "tissue_area_analyzed_mm2") if c in a and a[c].nunique() > 1]
    am = a.assign(_log_n=np.log10(a["n_nuclei"].clip(lower=1)))
    adj = []
    for c, lab in ks.PRIMARY:
        r_, _ = ks.logistic(am, [c, "_log_n", *covs], "adj")
        o = next((x for x in r_ if x.get("variable") == c), {})
        adj.append({"ölçüt": lab, "OR_per_SD": o.get("OR_per_SD"), "GA_low": o.get("OR_ci_low"), "GA_high": o.get("OR_ci_high"), "p": o.get("p")})
    adj = pd.DataFrame(adj)
    L += ["", "Düzeltilmiş lojistik (+ log10 çekirdek sayısı, bulanık doku, doku alanı):"]
    for _, r in adj.iterrows():
        if pd.notna(r["OR_per_SD"]):
            L.append(f"  {r['ölçüt']:<28} OR {r['OR_per_SD']:.2f} [{r['GA_low']:.2f}–{r['GA_high']:.2f}] {ks.fmt_p(r['p'])}")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"açıklama": [
            "R2: CPTAC-CCRCC (IDC cptac_ccrcc; grade: GDC CPTAC-3). Tüm slaytlar 20x → 0.5 µm/px modunda analiz.",
            "Tile fiziksel boyutu TCGA ile aynı (256 µm); StarDist 2x ölçekli tahmin; bin kenarları aynı (µm²).",
            "Mutlak değerler çözünürlük nedeniyle TCGA ile doğrudan karşılaştırılamaz; ilişkiler kohort içinde test edildi."]}).to_excel(
            xw, sheet_name="00_bilgi", index=False)
        prim.to_excel(xw, sheet_name="01_birincil", index=False)
        adj.to_excel(xw, sheet_name="02_duzeltilmis", index=False)
        sens.to_excel(xw, sheet_name="03_duyarlilik", index=False)
        a.to_excel(xw, sheet_name="10_vaka", index=False)
    # Uzamsal artımsal değer + özgüllük (R1 analizinin aynısı, CPTAC çekirdek tablolarından)
    _, txt1 = r1_calistir(a, out_dir, cfg, r1_xlsx, r1_png, r1_csv)
    return "\n".join(L) + "\n" + txt1
