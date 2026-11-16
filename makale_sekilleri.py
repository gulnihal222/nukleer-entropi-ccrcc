"""
Histopathology (Wiley) için makale şekilleri: Şekil 1-6.

Dergi kuralları:
  - genişlik 170,5 mm (iki sütun), en fazla 251,5 mm yükseklik
  - grafikler 800 dpi, fotoğraf içeren şekil (Şekil 2) 500 dpi
  - TIFF (LZW sıkıştırmalı, RGB) + ayrıca vektör PDF
  - panel sayısı 1, 2, 4 veya 6

Bu betik yeni analiz yapmaz; mevcut çıktı dosyalarınızı okuyup yalnızca çizer.
Şekil 3-6'daki her sayı (r, rho, HR, AUC) sizin sonuç dosyalarınızdan hesaplanır
ve çalışırken ekrana yazılır; makaledeki değerlerle karşılaştırın.

Kullanım (proje klasöründe):
  python scripts/makale_sekilleri.py
  python scripts/makale_sekilleri.py --sadece 3 6
  python scripts/makale_sekilleri.py --tcga yol/case_level_results.csv --cdr yol/TCGA-CDR-SupplementalTableS1.xlsx

Şekil 2'nin fotoğraf panelleri için --karolar seçeneğine 8 PNG verin (4 alan x [ham, segmentasyonlu]),
sırası: açık hücreli, yüksek grade, yoğun hücreli, nekroz. Verilmezse bu paneller boş çerçeve olarak çizilir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 2209

MM = 1 / 25.4
W_FULL = 170.5 * MM          # iki sütun
H_MAX = 251.5 * MM
DPI_GRAFIK = 800
DPI_KARMA = 500

BLUE, ORANGE, GREY, INK = "#3a76af", "#e1812c", "#bdbdbd", "#222222"
GRADE_COL = {"G1": "#c6dbef", "G2": "#3a76af", "G3": "#f5b97a", "G4": "#c8452b"}
ENTROPIES = [("area_um2_entropy", "Area entropy"), ("perimeter_um_entropy", "Perimeter entropy"),
             ("circularity_entropy", "Circularity entropy"), ("eccentricity_entropy", "Eccentricity entropy")]

# MoNuSeg böbrek görüntülerinde segmentasyon doğrulaması (Tablo S1; kirc_validate.py çıktısı)
MONUSEG = {"Classical (Otsu + watershed)": [0.44, 0.66, 0.36, 0.56],
           "StarDist, default thresholds": [0.70, 0.77, 0.56, 0.31],
           "StarDist, tuned thresholds (used)": [0.78, 0.82, 0.59, 0.25]}
KAROLAR = [("Clear-cell area", "TCGA-A3-3307, G3"), ("High-grade area", "TCGA-B0-4698, G4"),
           ("Densely cellular area", "TCGA-BP-4345, G3"), ("Necrotic debris", "TCGA-B0-4698, G4")]


# --------------------------------------------------------------------------- stil
def stil():
    aile = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": aile, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    print(f"Yazı tipi: {aile}")


def harf(ax, h, x=-0.16, y=1.04):
    ax.text(x, y, h, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def kaydet(fig, cikti: Path, ad: str, dpi: int):
    cikti.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    assert abs(w - W_FULL) < 1e-3 and h <= H_MAX + 1e-6, f"{ad}: boyut {w / MM:.1f} x {h / MM:.1f} mm"
    fig.savefig(cikti / f"{ad}.tif", dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    from PIL import Image  # dergi RGB istiyor; matplotlib RGBA (saydamlık kanallı) yazar
    with Image.open(cikti / f"{ad}.tif") as im:
        rgb = im.convert("RGB")
    rgb.save(cikti / f"{ad}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
    fig.savefig(cikti / f"{ad}.pdf")
    plt.close(fig)
    print(f"  -> {ad}.tif ({dpi} dpi, {w / MM:.1f} x {h / MM:.1f} mm) ve {ad}.pdf")


# --------------------------------------------------------------------------- veri
def oku(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p, sheet_name=0)


def analiz_seti(p: Path, beklenen: int) -> pd.DataFrame:
    d = oku(p)
    d = d[d["quality_control_status"].isin(["PASS", "WARN"])].copy()
    d["y"] = d["grade"].isin(["G3", "G4"]).astype(int)   # yüksek grade = G3-G4
    d["g"] = d["grade"].str[1].astype(int)
    if len(d) != beklenen:
        print(f"  UYARI: {p.name} içinde {len(d)} hasta var, beklenen {beklenen}")
    return d


def rbc(x, y):
    """Rank-biserial r (pozitif: yüksek grade'de daha yüksek)."""
    hi, lo = x[y == 1], x[y == 0]
    u = mannwhitneyu(hi, lo).statistic
    return 2 * u / (len(hi) * len(lo)) - 1


def rbc_ci(x, y, n=2000):
    rng = np.random.default_rng(SEED)
    hi, lo = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        h = x[rng.choice(hi, len(hi))]
        l = x[rng.choice(lo, len(lo))]
        u = mannwhitneyu(h, l).statistic
        vals.append(2 * u / (len(h) * len(l)) - 1)
    return np.percentile(vals, [2.5, 97.5])


def kayitli_ci(yol: Path | None, sayfa: str) -> dict:
    """Sonuç dosyanızdaki r ve %95 GA değerlerini okur (Tablo 2 ile birebir aynı olsun diye).
    Bulunamazsa boş döner; o zaman GA yeniden bootstrap ile hesaplanır ve küçük farklar olabilir."""
    if not yol or not Path(yol).exists():
        return {}
    try:
        t = pd.read_excel(yol, sheet_name=sayfa)
    except ValueError:
        return {}
    anahtar = next((c for c in t.columns if t[c].astype(str).isin(["area_um2_entropy"]).any()), None)
    if anahtar is None:
        return {}
    out = {}
    for _, r in t.iterrows():
        if "r_ci_low" in t:
            out[r[anahtar]] = (r["rank_biserial_r"], r["r_ci_low"], r["r_ci_high"])
        elif "r_GA" in t:
            lo, hi = [float(v) for v in str(r["r_GA"]).strip("[]").split(",")]
            out[r[anahtar]] = (r["r (High\u2212Low)"], lo, hi)
    return out


def grade_kutu(ax, d, col, ylab=None):
    rng = np.random.default_rng(SEED)
    vals = [d.loc[d.grade == g, col].dropna().values for g in ["G1", "G2", "G3", "G4"]]
    bp = ax.boxplot(vals, positions=range(4), widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.6))
    for patch, g in zip(bp["boxes"], GRADE_COL):
        patch.set_facecolor(GRADE_COL[g]); patch.set_alpha(0.55)
    for i, (v, g) in enumerate(zip(vals, GRADE_COL)):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=2.2, color=GRADE_COL[g], lw=0, zorder=3)
    ax.plot(range(4), [np.median(v) for v in vals], "--", color="#444444", lw=0.6, zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(["G1", "G2", "G3", "G4"])
    if ylab:
        ax.set_ylabel(ylab)


# --------------------------------------------------------------------------- Şekil 1
def sekil1(cikti, **_):
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(W_FULL, 92 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170.5); ax.set_ylim(8, 100); ax.axis("off")

    def kutu(x, y, w, h, baslik, metin, kenar="#555555", dolgu="white", bs=7.5, ms=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=dolgu, ec=kenar, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, baslik, ha="center", va="top", fontsize=bs, fontweight="bold")
        ax.text(x + w / 2, y + h - 2.2 - bs * 0.55, metin, ha="center", va="top", fontsize=ms, linespacing=1.3)

    kutu(4, 70, 76, 27, "TCGA-KIRC (development cohort)",
         "200 patients, one FFPE diagnostic slide each\nLow grade 100 (G1 13, G2 87)\nHigh grade 100 (G3 50, G4 50)\n"
         "40× scans, analysed at 0.25 µm/pixel\n24.7 million in-focus nuclei", BLUE, "#eef3f9")
    kutu(90.5, 70, 76, 27, "CPTAC-CCRCC (external cohort)",
         "221 patients, one tumour slide each (IDC, DICOM)\nLow grade 133 (G1 19, G2 114)\nHigh grade 88 (G3 67, G4 21)\n"
         "20× scans, analysed at 0.5 µm/pixel\n26.4 million in-focus nuclei", ORANGE, "#fdf2e9")
    adim = [("Tissue mask\nand QC", "pen marks, focus,\nstain screen"),
            ("Stain\nnormalisation", "Macenko,\nhaematoxylin\nchannel"),
            ("Nuclear\nsegmentation", "StarDist,\nvalidated on\nMoNuSeg"),
            ("Morphometry", "area, perimeter,\ncircularity,\neccentricity"),
            ("Shannon\nentropy", "fixed bins;\nslide and tile\nlevel")]
    w, gap, x0, y0, h = 29.5, 4.25, 4, 37, 23
    for i, (b, m) in enumerate(adim):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc="white", ec="#555555", lw=0.8))
        ax.text(x + w / 2, y0 + h - 2, b, ha="center", va="top", fontsize=7, fontweight="bold", linespacing=1.0)
        ax.text(x + w / 2, y0 + 8.5, m, ha="center", va="center", fontsize=6.2, linespacing=1.15)
        if i < 4:
            ax.annotate("", (x + w + gap - 0.4, y0 + h / 2), (x + w + 0.4, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
    mid = x0 + 2 * (w + gap) + w / 2
    for xs in (42, 128.5):
        ax.annotate("", (mid, y0 + h + 0.5), (xs, 70), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
    ax.annotate("", (mid, 32), (mid, y0), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
    ax.add_patch(FancyBboxPatch((4, 10), 162.5, 22, boxstyle="round,pad=0,rounding_size=1.5", fc="#f4f4f4", ec="#888888", lw=0.8))
    ax.text(85.25, 29.5, "Analyses", ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.text(9, 24, "Association with grade (Mann-Whitney U, Spearman,\nadjusted and centre-stratified models)\n"
            "Incremental value over classical morphometry\nSpatial heterogeneity: entropy versus SD-based tile metrics",
            va="top", fontsize=6.5, linespacing=1.35)
    ax.text(90, 24, "Tumour-enriched nuclei (lymphocyte-like and\nspindle nuclei removed)\n"
            "Survival: OS, DSS and PFI (TCGA-CDR)\nExternal replication in CPTAC-CCRCC",
            va="top", fontsize=6.5, linespacing=1.35)
    kaydet(fig, cikti, "Figure1", DPI_GRAFIK)


# --------------------------------------------------------------------------- Şekil 2 (6 panel)
def sekil2(cikti, karolar=None, **_):
    from PIL import Image
    fig = plt.figure(figsize=(W_FULL, 162 * MM))
    gs = fig.add_gridspec(1, 4, left=0.07, right=0.99, top=0.91, bottom=0.66, wspace=0.05)
    gk = fig.add_gridspec(2, 4, left=0.015, right=0.99, top=0.585, bottom=0.005, hspace=0.13, wspace=0.025)
    renk = [GREY, "#8fb4dc", BLUE]
    ax_a = fig.add_subplot(gs[0, 0:3]); ax_b = fig.add_subplot(gs[0, 3])
    x = np.arange(3); bw = 0.26
    for j, (lab, v) in enumerate(MONUSEG.items()):
        ax_a.bar(x + (j - 1) * bw, v[:3], bw, color=renk[j], ec="#555555", lw=0.4, label=lab)
        ax_b.bar((j - 1) * bw, v[3], bw, color=renk[j], ec="#555555", lw=0.4)
        for xi, vi in zip(x + (j - 1) * bw, v[:3]):
            ax_a.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=5.5)
        ax_b.text((j - 1) * bw, v[3] + 0.015, f"{v[3]:.2f}", ha="center", fontsize=5.5)
    ax_a.set_xticks(x); ax_a.set_xticklabels(["F1", "Pixel Dice", "PQ"]); ax_a.set_ylim(0, 1); ax_a.set_ylabel("value")
    ax_b.set_xticks([0]); ax_b.set_xticklabels(["Entropy error"]); ax_b.set_ylabel("bits"); ax_b.set_ylim(0, 0.7)
    ax_b.yaxis.set_label_coords(-0.32, 0.5)
    ax_a.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=6, handlelength=1.2, columnspacing=1.2)
    harf(ax_a, "a", -0.08, 1.12); harf(ax_b, "b", -0.42, 1.12)
    pb = ax_b.get_position(); ax_b.set_position([pb.x0 + 0.06, pb.y0, pb.width - 0.06, pb.height])
    yer = [(0, 0), (0, 2), (1, 0), (1, 2)]
    for k, ((bas, vaka), (r, c)) in enumerate(zip(KAROLAR, yer)):
        for s in range(2):
            ax = fig.add_subplot(gk[r, c + s]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#999999")
            yol = karolar[2 * k + s] if karolar and len(karolar) == 8 else None
            if yol:
                ax.imshow(np.asarray(Image.open(yol).convert("RGB")))
            else:
                ax.text(0.5, 0.5, "tile image\nnot supplied", ha="center", va="center", fontsize=6, color="#999999",
                        transform=ax.transAxes)
            if s == 0:
                ax.set_title(f"      {bas} ({vaka})", loc="left", fontsize=7, pad=3)
                harf(ax, "cdef"[k], 0.0, 1.075)
    if not karolar:
        print("  UYARI: --karolar verilmedi; c-f panelleri boş. 8 PNG verin (ham, segmentasyonlu x 4).")
    kaydet(fig, cikti, "Figure2", DPI_KARMA)


# --------------------------------------------------------------------------- Şekil 3 (4 panel)
def sekil3(cikti, tcga=None, **_):
    fig, axs = plt.subplots(1, 4, figsize=(W_FULL, 58 * MM))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.12, wspace=0.42)
    for i, (ax, (col, lab)) in enumerate(zip(axs, ENTROPIES)):
        grade_kutu(ax, tcga, col, "Shannon entropy (bits)" if i == 0 else None)
        r = rbc(tcga[col].values, tcga.y.values); rho = spearmanr(tcga.g, tcga[col]).statistic
        ax.set_title(f"{lab}\nr = {r:.2f}, ρ = {rho:.2f}".replace("-", "−"))
        harf(ax, "abcd"[i], -0.30, 1.0)
        print(f"  {lab}: r = {r:.3f}, rho = {rho:.3f}")
    kaydet(fig, cikti, "Figure3", DPI_GRAFIK)


# --------------------------------------------------------------------------- Şekil 4 (2 panel)
def sekil4(cikti, tcga=None, cptac=None, ist=None, r2ozet=None, **_):
    olc = [("area_um2_entropy", "Area entropy"), ("perimeter_um_entropy", "Perimeter entropy"),
           ("circularity_entropy", "Circularity entropy"), ("eccentricity_entropy", "Eccentricity entropy"),
           ("log2_area_entropy", "log$_2$(area) entropy"), ("area_um2_std", "Nuclear area SD"),
           ("area_um2_median", "Median nuclear area")]
    fig = plt.figure(figsize=(W_FULL, 72 * MM))
    ax_a = fig.add_axes([0.19, 0.17, 0.30, 0.70]); ax_b = fig.add_axes([0.60, 0.17, 0.39, 0.70])
    kayit = {"TCGA-KIRC": kayitli_ci(ist, "03_mann_whitney"), "CPTAC-CCRCC": kayitli_ci(r2ozet, "01_birincil")}
    for k, (d, ad, renk, mk, dy) in enumerate([(tcga, "TCGA-KIRC", BLUE, "o", -0.15), (cptac, "CPTAC-CCRCC", ORANGE, "s", 0.15)]):
        for i, (c, lab) in enumerate(olc):
            if c in kayit[ad]:
                r, lo, hi = kayit[ad][c]
            else:
                print(f"  not: {ad} {c} için kayıtlı GA bulunamadı, bootstrap ile hesaplanıyor")
                r = rbc(d[c].values, d.y.values); lo, hi = rbc_ci(d[c].values, d.y.values)
            yy = i + (0.6 if i >= 4 else 0) + dy
            ax_a.errorbar(r, yy, xerr=[[r - lo], [hi - r]], fmt=mk, ms=3, color=renk, capsize=1.5, lw=0.7,
                          label=f"{ad} (n = {len(d)})" if i == 0 else None)
            print(f"  {ad:12s} {lab:22s} r = {r:.2f} ({lo:.2f} to {hi:.2f})")
    yt = [i + (0.6 if i >= 4 else 0) for i in range(len(olc))]
    ax_a.set_yticks(yt); ax_a.set_yticklabels([l for _, l in olc]); ax_a.invert_yaxis()
    ax_a.axvline(0, color="#777777", ls="--", lw=0.6); ax_a.axhline(3.8, color="#dddddd", lw=0.6)
    ax_a.set_xlabel("rank-biserial r, high vs low grade (95% CI)")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, handletextpad=0.3)
    harf(ax_a, "a", -0.62, 1.02)
    # b: CPTAC alan ve çevre entropisi, grade'e göre, aynı eksende
    rng = np.random.default_rng(SEED)
    for j, (c, lab, renk) in enumerate([("area_um2_entropy", "Area entropy", BLUE), ("perimeter_um_entropy", "Perimeter entropy", ORANGE)]):
        vals = [cptac.loc[cptac.grade == g, c].values for g in ["G1", "G2", "G3", "G4"]]
        pos = np.arange(4) + (j - 0.5) * 0.36
        bp = ax_b.boxplot(vals, positions=pos, widths=0.3, showfliers=False, patch_artist=True,
                          medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                          boxprops=dict(lw=0.6))
        for p in bp["boxes"]:
            p.set_facecolor(renk); p.set_alpha(0.45)
        for pp, v in zip(pos, vals):
            ax_b.scatter(pp + rng.uniform(-0.09, 0.09, len(v)), v, s=1.6, color=renk, lw=0, zorder=3)
        r = rbc(cptac[c].values, cptac.y.values); rho = spearmanr(cptac.g, cptac[c]).statistic
        ax_b.plot([], [], "s", color=renk, alpha=0.6, label=f"{lab}: r = {r:.2f}, ρ = {rho:.2f}")
    ax_b.set_xticks(range(4)); ax_b.set_xticklabels(["G1", "G2", "G3", "G4"])
    ax_b.set_ylabel("Shannon entropy (bits), CPTAC-CCRCC")
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.3)
    harf(ax_b, "b", -0.17, 1.02)
    kaydet(fig, cikti, "Figure4", DPI_GRAFIK)


# --------------------------------------------------------------------------- Şekil 5 (4 panel)
def sekil5(cikti, tcga=None, r1=None, r2=None, **_):
    fig = plt.figure(figsize=(W_FULL, 130 * MM))
    ax = fig.add_axes([0.08, 0.60, 0.30, 0.32])
    for yv, lab, renk in [(0, "Low grade", BLUE), (1, "High grade", ORANGE)]:
        s = tcga[tcga.y == yv]
        ax.scatter(s.area_um2_std, s.area_um2_entropy, s=3, color=renk, alpha=0.75, lw=0, label=lab)
    rho = spearmanr(tcga.area_um2_std, tcga.area_um2_entropy).statistic
    ax.set_xlabel("SD of nuclear area (µm²)"); ax.set_ylabel("Area entropy (bits)")
    ax.set_title(f"TCGA-KIRC, Spearman ρ = {rho:.2f}"); ax.legend(loc="lower right", markerscale=2)
    harf(ax, "a", -0.22, 1.04); print(f"  rho(area entropy, area SD) = {rho:.3f}")
    ax = fig.add_axes([0.66, 0.60, 0.32, 0.32])
    pix = [("grayscale_entropy", "Grey-level pixel entropy", GREY), ("hematoxylin_entropy", "Haematoxylin pixel entropy", GREY),
           ("eosin_entropy", "Eosin pixel entropy", GREY), ("area_um2_entropy", "Nuclear area entropy", BLUE),
           ("perimeter_um_entropy", "Nuclear perimeter entropy", BLUE)]
    for i, (c, lab, renk) in enumerate(pix):
        auc = (rbc(tcga[c].values, tcga.y.values) + 1) / 2
        ax.barh(i, auc - 0.45, left=0.45, color=renk, height=0.6)
        ax.text(auc + 0.005, i, f"{auc:.2f}", va="center", fontsize=6); print(f"  AUC {lab}: {auc:.3f}")
    ax.set_yticks(range(5)); ax.set_yticklabels([p[1] for p in pix]); ax.invert_yaxis()
    ax.set_xlim(0.45, 0.8); ax.axvline(0.5, color="#777777", ls=":", lw=0.6)
    ax.set_xlabel("apparent AUC, high vs low grade"); ax.set_title("TCGA-KIRC")
    harf(ax, "b", -0.75, 1.04)
    etiket = ["Classical", "+ tile\nentropy", "+ tile\narea SD", "+ tile\narea CV", "+ tile entropy\n+ tile area SD"]
    for k, (tab, ad, renk) in enumerate([(r1, "TCGA-KIRC", BLUE), (r2, "CPTAC-CCRCC", ORANGE)]):
        ax = fig.add_axes([0.08 + 0.50 * k, 0.13, 0.40, 0.32])
        m, s = tab["CV_AUC_ort"].values, tab["CV_AUC_SD"].values
        ax.bar(range(5), m, yerr=s, color=renk, width=0.65, capsize=1.8, error_kw=dict(lw=0.6))
        for i, (mi, si) in enumerate(zip(m, s)):
            ax.text(i, mi + si + 0.001, f"{mi:.3f}", ha="center", fontsize=5.5)
        ax.set_ylim(0.68, 0.76); ax.set_xticks(range(5)); ax.set_xticklabels(etiket, fontsize=6)
        ax.set_ylabel("cross-validated AUC (mean ± SD)" if k == 0 else ""); ax.set_title(ad)
        harf(ax, "cd"[k], -0.20 if k == 0 else -0.10, 1.04); print(f"  CV-AUC {ad}: " + ", ".join(f"{v:.3f}" for v in m))
    fig.text(0.5, 0.01, "Classical model: median nuclear area + area SD; tile measures entered as SD and 90th percentile. 5-fold cross-validation, 20 repeats.", ha="center", fontsize=6, color="#555555")
    kaydet(fig, cikti, "Figure5", DPI_GRAFIK)


# --------------------------------------------------------------------------- Şekil 6 (4 panel)
def sekil6(cikti, tcga=None, cdr=None, r3=None, **_):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    c = pd.read_excel(cdr, sheet_name="TCGA-CDR")
    c = c[c["type"] == "KIRC"].copy(); c["case_id"] = c["bcr_patient_barcode"].astype(str).str[:12]
    for e in ("OS", "DSS"):
        c[e] = pd.to_numeric(c[e], errors="coerce"); c[f"{e}.time"] = pd.to_numeric(c[f"{e}.time"], errors="coerce")
    d = tcga.merge(c[["case_id", "OS", "OS.time", "DSS", "DSS.time"]], on="case_id", how="left")
    fig = plt.figure(figsize=(W_FULL, 150 * MM))
    yerler = {"a": [0.08, 0.62, 0.38, 0.30], "b": [0.58, 0.62, 0.38, 0.30],
              "c": [0.25, 0.11, 0.27, 0.39], "d": [0.72, 0.11, 0.26, 0.39]}
    for k, (e, ad) in enumerate([("OS", "Overall survival"), ("DSS", "Disease-specific survival")]):
        ax = fig.add_axes(yerler["ab"[k]]); s = d.dropna(subset=[e, f"{e}.time"])
        hi = s.area_um2_entropy >= s.area_um2_entropy.median()
        for mask, lab, renk in [(~hi, "below median", BLUE), (hi, "at or above median", ORANGE)]:
            KaplanMeierFitter().fit(s.loc[mask, f"{e}.time"] / 30.44, s.loc[mask, e], label=lab).plot_survival_function(
                ax=ax, color=renk, lw=0.9, ci_alpha=0.12)
        p = logrank_test(s.loc[hi, f"{e}.time"], s.loc[~hi, f"{e}.time"], s.loc[hi, e], s.loc[~hi, e]).p_value
        ps = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        ax.set_title(f"{ad}\nlog-rank P {ps}"); ax.set_xlabel("months"); ax.set_ylim(0, 1.02)
        ax.set_ylabel("survival probability" if k == 0 else "")
        ax.legend(title="Area entropy", loc="lower left", title_fontsize=6.5)
        harf(ax, "ab"[k], -0.16, 1.04); print(f"  {e}: n = {len(s)}, log-rank P = {p:.4f}")
    cox = pd.read_excel(r3, sheet_name="03_cox")
    cox = cox[cox["model"].str.startswith("düzelt")]
    ad_map = {"area_um2_entropy": "Area entropy", "perimeter_um_entropy": "Perimeter entropy",
              "patch_nuc_area_entropy_std": "Tile entropy SD", "patch_nuc_area_entropy_p90": "Tile entropy p90",
              "area_um2_std": "Nuclear area SD", "area_um2_median": "Median nuclear area"}
    paneller = [("c", [("OS", "#333333"), ("DSS", BLUE)], "OS and DSS"),
                ("d", [("PFI", ORANGE)], "PFI (secondary endpoint)")]
    for k, (h, sons, bas) in enumerate(paneller):
        ax = fig.add_axes(yerler[h]); y = 0; yt, yl = [], []
        for e, renk in sons:
            t = cox[cox["sonlanım"] == e]
            for _, r in t.iterrows():
                ax.errorbar(r.HR_per_SD, y, xerr=[[r.HR_per_SD - r.HR_GA_low], [r.HR_GA_high - r.HR_per_SD]],
                            fmt="o", ms=2.6, color=renk, capsize=1.5, lw=0.7)
                yt.append(y); yl.append(f"{e} · {ad_map.get(r['değişken'], r['değişken'])}"); y += 1
            y += 0.6
        ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6); ax.invert_yaxis()
        ax.set_xscale("log"); ax.set_xticks([0.75, 1, 1.5, 2]); ax.set_xticklabels(["0.75", "1", "1.5", "2"])
        ax.minorticks_off(); ax.axvline(1, color="#777777", ls="--", lw=0.6)
        ax.set_xlabel("HR per SD (95% CI)"); ax.set_title(bas); harf(ax, h, -0.85 if k == 0 else -0.80, 1.03)
    fig.text(0.5, 0.01, "Cox models adjusted for stage III-IV, high grade and age", ha="center", fontsize=6, color="#555555")
    kaydet(fig, cikti, "Figure6", DPI_GRAFIK)


# --------------------------------------------------------------------------- ana
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tcga", type=Path, default=ROOT / "09_results" / "pipeline" / "case_level_results.csv")
    ap.add_argument("--cptac", type=Path, default=ROOT / "09_results" / "cptac" / "case_level_results.csv")
    ap.add_argument("--cdr", type=Path, default=ROOT / "02_raw_data" / "TCGA-CDR-SupplementalTableS1.xlsx")
    ap.add_argument("--r1", type=Path, default=ROOT / "07_statistics" / "revizyon" / "R1_uzamsal_ozgulluk.xlsx")
    ap.add_argument("--r2", type=Path, default=ROOT / "07_statistics" / "revizyon" / "R2_cptac_uzamsal.xlsx")
    ap.add_argument("--r3", type=Path, default=ROOT / "07_statistics" / "revizyon" / "R3_sagkalim.xlsx")
    ap.add_argument("--ist", type=Path, default=ROOT / "07_statistics" / "statistics_report.xlsx")
    ap.add_argument("--r2ozet", type=Path, default=ROOT / "07_statistics" / "revizyon" / "R2_cptac.xlsx")
    ap.add_argument("--karolar", type=Path, nargs="*", help="Şekil 2 için 8 PNG")
    ap.add_argument("--cikti", type=Path, default=ROOT / "08_figures" / "makale")
    ap.add_argument("--sadece", type=int, nargs="*", help="yalnız bu şekiller, ör. --sadece 3 6")
    a = ap.parse_args()
    stil()
    hangi = a.sadece or [1, 2, 3, 4, 5, 6]
    veri = {"cikti": a.cikti, "karolar": a.karolar, "cdr": a.cdr, "r3": a.r3, "ist": a.ist, "r2ozet": a.r2ozet}
    if set(hangi) & {3, 4, 5, 6}:
        veri["tcga"] = analiz_seti(a.tcga, 200)
    if 4 in hangi:
        veri["cptac"] = analiz_seti(a.cptac, 221)
    if 5 in hangi:
        veri["r1"] = pd.read_excel(a.r1, sheet_name="02_modeller_cv_auc")
        veri["r2"] = pd.read_excel(a.r2, sheet_name="02_modeller_cv_auc")
    for n in hangi:
        print(f"\nŞekil {n}")
        globals()[f"sekil{n}"](**veri)
    print(f"\nTamam. Dosyalar: {a.cikti}")


if __name__ == "__main__":
    main()
