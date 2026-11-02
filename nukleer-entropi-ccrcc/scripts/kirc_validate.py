#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentasyon doğrulaması (proje takvimi 4-5. ay: "Dice/F1 ≥ 0.85", "çekirdek tespitinde doğruluk ≥ %80")
=======================================================================================================

İş akışı:
  1) export-tiles : analiz edilmiş vakalardan, grade'e göre dengeli, ODAKLI tile'ları PNG olarak dışa aktarır.
  2) QuPath       : her PNG'de çekirdekleri elle çizin (isteğe bağlı sınıf: Tumor / Lymphocyte / Endothelial / Other)
                    → File ▸ Export objects as GeoJSON → aynı klasöre <png_adı>.geojson
  3) evaluate     : her segmentasyon yöntemi (classical, stardist) için nesne düzeyi F1, PQ, piksel Dice,
                    çekirdek özelliklerinde yanlılık ve — en önemlisi — TILE ENTROPİSİNDEKİ sapma.
  4) tune         : klasik yöntemin parametreleri; vakalar ikiye bölünür (ayar / test) → aşırı uyum önlenir.

Kullanım:
  python kirc_validate.py export-tiles --results results --out annot --n-cases 10 --tiles-per-case 2
  python kirc_validate.py evaluate --annot annot --results results --segmenters classical stardist
  python kirc_validate.py tune --annot annot --results results
  python kirc_validate.py selftest
Kural: parametreler YALNIZ bu doğrulama setine bakılarak seçilir, grade sonuçlarına bakılarak değil.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kirc_entropy_pipeline as kp  # noqa: E402

IOU_MATCH = 0.5
TARGET_F1 = 0.80     # proje: çekirdek tespitinde doğruluk ≥ %80
TARGET_DICE = 0.85   # proje: Dice/F1 ≥ 0.85
GUIDE = """# QuPath ile çekirdek etiketleme (ground truth)

1. QuPath ▸ Create project ▸ bu klasördeki PNG'leri ekleyin (Image type: Brightfield H&E).
2. Her görüntüde TÜM çekirdekleri çizin: Brush (B) veya Wand (W). Tile kenarına değen çekirdekleri de çizin.
   • Çakışan çekirdekleri ayrı nesne olarak çizin. • Emin olmadığınız soluk yapıları çizmeyin, not alın.
3. (İsteğe bağlı ama önerilir) sınıf verin: Tumor, Lymphocyte, Endothelial, Other
   (Annotations ▸ sınıf listesine ekleyin; nesneyi seçip sınıfı atayın).
4. File ▸ Export objects as GeoJSON ▸ "Export as FeatureCollection" ▸ dosya adı: <png adı>.geojson (aynı klasöre).
5. Süre tahmini: 512×512 piksel (128 µm) tile ≈ 60–100 çekirdek ≈ 20–40 dk. 20 tile ≈ 8–12 saat.
6. Mümkünse 5 tile'ı danışmanınız/patolog da çizsin → gözlemciler arası uyum (aynı komutla iki klasör karşılaştırılabilir).
"""


# -----------------------------------------------------------------------------
# Yardımcılar
# -----------------------------------------------------------------------------
def load_case_ctx(results: Path | None, slide_id: str, cfg: kp.Config, rgb: np.ndarray, mpp: float) -> dict:
    """Vakanın kayıtlı boya parametreleri (pipeline ile birebir aynı ön işleme). Yoksa tile üzerinden kestirilir."""
    if results:
        rp = Path(results) / "cases" / slide_id / "case_result.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            if r.get("stain_matrix"):
                he = np.array(json.loads(r["stain_matrix"]))
                return {"io": np.array(json.loads(r["background_io"])), "pinv": np.linalg.pinv(he),
                        "scale": np.array(json.loads(r["stain_conc_scale"])), "thr": float(r["nuclear_threshold_h"]),
                        "mpp": mpp, "sd_lo": json.loads(r.get("stardist_norm_lo") or "[0,0,0]"),
                        "sd_hi": json.loads(r.get("stardist_norm_hi") or "[255,255,255]"), "source": "case_result"}
    # Yedek: tile'ın kendisinden (dış görüntüler / sentetik test)
    from skimage.filters import threshold_otsu
    io = np.array([245.0, 245.0, 245.0])
    bright = rgb.reshape(-1, 3)
    bright = bright[bright.mean(1) > 220]
    if len(bright) > 500:
        io = np.clip(np.median(bright, 0), 200, 255)
    od = kp.rgb_to_od(rgb, io).reshape(-1, 3)
    ref = np.asarray(cfg.ref_stain_matrix, float); ref /= np.linalg.norm(ref, axis=0)
    try:
        he = kp.macenko_stain_matrix(od, cfg.macenko_beta, cfg.macenko_alpha)
        if min(he[:, 0] @ ref[:, 0], he[:, 1] @ ref[:, 1]) < cfg.stain_min_cosine_to_reference:
            he = ref
    except Exception:  # noqa: BLE001
        he = ref
    pinv = np.linalg.pinv(he)
    st = od[np.linalg.norm(od, axis=1) > cfg.macenko_beta]
    conc = st @ pinv.T
    scale = np.asarray(cfg.ref_max_conc) / np.maximum(np.percentile(conc, 99, axis=0), 1e-6)
    thr = float(threshold_otsu(np.clip(conc[:, 0] * scale[0], 0, None))) * cfg.nuc_threshold_scale
    px = rgb.reshape(-1, 3)
    return {"io": io, "pinv": pinv, "scale": scale, "thr": thr, "mpp": mpp,
            "sd_lo": np.percentile(px, cfg.stardist_norm_percentiles[0], 0),
            "sd_hi": np.percentile(px, cfg.stardist_norm_percentiles[1], 0), "source": "tile"}


MIN_GT_PER_TILE = 10   # bundan az çizim içeren tile yarım kabul edilir ve değerlendirmeye alınmaz


def gt_path(annot: Path, row: dict) -> Path | None:
    """Etiket dosyası: <png adı>.gt.npz (hazır veri setleri, ör. MoNuSeg) veya <png adı>.geojson (QuPath)."""
    stem = Path(row["png"]).stem
    for ext in (".gt.npz", ".geojson"):
        if (annot / (stem + ext)).exists():
            return annot / (stem + ext)
    return None


def load_gt(annot: Path, row: dict) -> tuple[np.ndarray, dict]:
    p = gt_path(annot, row)
    if p.name.endswith(".gt.npz"):
        lab = np.load(p)["labels"].astype(np.int32)
        return lab, {int(k): "Unclassified" for k in np.unique(lab) if k}
    return read_gt(p, row)


def read_gt(geojson: Path, row: dict) -> tuple[np.ndarray, dict]:
    """GeoJSON → örnek (instance) etiket görüntüsü. Koordinatlar PNG pikseli ya da (otomatik algılanır)
    WSI seviye-0 pikseli olabilir. Döner: (etiketler, {etiket: sınıf})."""
    from skimage.draw import polygon as draw_polygon
    n = int(row["tile_px"])
    data = json.loads(Path(geojson).read_text(encoding="utf-8"))
    feats = data.get("features", [data]) if isinstance(data, dict) else data
    rings, classes = [], []
    for f in feats:
        props = f.get("properties", {}) or {}
        geom = f.get("nucleusGeometry") or props.get("nucleusGeometry") or f.get("geometry")
        if not geom:
            continue
        cl = props.get("classification") or {}
        name = cl.get("name") if isinstance(cl, dict) else None
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"] if geom["type"] == "MultiPolygon" else []
        for poly in polys:
            if poly:
                rings.append(np.asarray(poly[0], float))
                classes.append(name or "Unclassified")
    if not rings:
        return np.zeros((n, n), np.int32), {}
    allmax = max(r.max() for r in rings)
    wsi = allmax > n * 1.2   # WSI koordinatı → tile koordinatına çevir
    s = float(row["tile_l0"]) / n
    lab = np.zeros((n, n), np.int32)
    cmap = {}
    for k, (r, c) in enumerate(zip(rings, classes), 1):
        xs, ys = (r[:, 0], r[:, 1]) if not wsi else ((r[:, 0] - row["x_l0"]) / s, (r[:, 1] - row["y_l0"]) / s)
        rr, cc = draw_polygon(ys, xs, (n, n))
        lab[rr, cc] = k
        cmap[k] = c
    return lab, cmap


def match_instances(gt: np.ndarray, pr: np.ndarray, thr: float = IOU_MATCH) -> dict:
    """IoU > 0.5 eşleşmesi tektir (Kirillov ve ark. 2019) → TP/FP/FN, F1, SQ, PQ, piksel Dice."""
    g_ids = np.unique(gt[gt > 0]); p_ids = np.unique(pr[pr > 0])
    both = (gt > 0) & (pr > 0)
    inter = pd.Series(1, index=pd.MultiIndex.from_arrays([gt[both], pr[both]])).groupby(level=[0, 1]).sum() if both.any() else pd.Series(dtype=int)
    ga = pd.Series(np.bincount(gt.ravel()))
    pa = pd.Series(np.bincount(pr.ravel()))
    matches = []
    for (g, p), i in inter.items():
        iou = i / (ga[g] + pa[p] - i)
        if iou > thr:
            matches.append((g, p, iou))
    tp = len(matches); fp = len(p_ids) - tp; fn = len(g_ids) - tp
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    sq = float(np.mean([m[2] for m in matches])) if matches else float("nan")
    gfg, pfg = gt > 0, pr > 0
    dice = 2 * (gfg & pfg).sum() / max(gfg.sum() + pfg.sum(), 1)
    return {"n_gt": len(g_ids), "n_pred": len(p_ids), "TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec,
            "F1": f1, "SQ": sq, "PQ": sq * f1 if matches else 0.0, "pixel_dice": float(dice), "matches": matches}


def tile_entropy(nuc: pd.DataFrame, cfg: kp.Config) -> dict:
    out = {}
    for f in kp.MORPH_FEATURES:
        lo, hi, nb = cfg.feature_bins[f]
        out[f"{f}_entropy"] = kp.histogram_entropy(nuc[f], lo, hi, int(nb), cfg.log_base)["entropy"] if len(nuc) >= 10 else float("nan")
        out[f"{f}_median"] = float(nuc[f].median()) if len(nuc) else float("nan")
    return out


def evaluate_tile(rgb, gt, gt_cls, row, results, cfg, segmenter) -> tuple[dict, np.ndarray]:
    c = kp.Config(**{**cfg.__dict__, "segmenter": segmenter})
    mpp = float(row["mpp"])
    ctx = load_case_ctx(results, row["slide_id"], c, rgb, mpp)
    h, _ = kp.deconvolve(rgb, ctx["io"], ctx["pinv"], ctx["scale"])
    pred = kp.segment_nuclei(rgb, h, np.ones(h.shape, bool), ctx, c)
    m = match_instances(gt, pred)
    # Özellik ve entropi uyumu (pipeline ile aynı ölçüm; kenara değenler hariç)
    ng, _ = kp.measure_nuclei(gt, h, mpp, c)
    npred, _ = kp.measure_nuclei(pred, h, mpp, c)
    eg, ep = tile_entropy(ng, c), tile_entropy(npred, c)
    rec = {k: v for k, v in m.items() if k != "matches"}
    for k in eg:
        rec[f"gt_{k}"], rec[f"pred_{k}"] = eg[k], ep[k]
        rec[f"diff_{k}"] = ep[k] - eg[k]
    # Sınıf bazında duyarlılık (GT sınıfı varsa)
    matched_g = {g for g, _, _ in m["matches"]}
    for cls in sorted(set(gt_cls.values())):
        ids = [g for g, cc in gt_cls.items() if cc == cls and (gt == g).any()]
        if ids:
            rec[f"recall_{cls}"] = sum(g in matched_g for g in ids) / len(ids)
            rec[f"n_gt_{cls}"] = len(ids)
    rec["stain_ctx_source"] = ctx["source"]
    return rec, pred


def overlay(rgb, gt, preds: dict, path: Path):
    """GT (beyaz) ve tahmin konturları, GERÇEK 1:1 piksel (yumuşatma yok)."""
    from skimage.segmentation import find_boundaries
    ov = rgb.copy(); ov[find_boundaries(gt, mode="inner")] = (255, 255, 255)
    row = [(rgb, "orijinal (1:1)", None), (ov, f"ground truth (n={len(np.unique(gt)) - 1})", None)]
    for name, (pr, m) in preds.items():
        o = rgb.copy()
        o[find_boundaries(gt, mode="inner")] = (255, 255, 255)
        o[find_boundaries(pr, mode="inner")] = (0, 230, 0) if name == "classical" else (255, 210, 0)
        row.append((o, f"{name}: F1 {m['F1']:.2f} · PQ {m['PQ']:.2f} · Dice {m['pixel_dice']:.2f}", None))
    kp.pixel_montage([row], path, title="beyaz = ground truth · yeşil = klasik · sarı = StarDist")


# -----------------------------------------------------------------------------
# Komutlar
# -----------------------------------------------------------------------------
def export_tiles(results: Path, out: Path, n_cases: int, tiles_per_case: int, seed: int = 7,
                 only_slides: list[str] | None = None, crop_um: float | None = 128.0) -> pd.DataFrame:
    """Doğrulama için tile'lar. crop_um: analiz tile'ının ORTASINDAN bu boyutta (µm) kare kesilir
    (0.25 µm/px'te 128 µm = 512 px ≈ 60–100 çekirdek → elle çizim yapılabilir süre). None = tile'ın tamamı."""
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    cl = pd.read_csv(results / "case_level_results.csv")
    cl = cl[cl["analysis_status"].isin(kp.SUCCESS_STATUSES)]
    if only_slides is not None:
        cl = cl[cl["slide_id"].isin(only_slides)]
    rng = np.random.default_rng(seed)
    # grade'e göre dengeli (G1, G2, G3, G4 sırayla)
    pools = {g: list(rng.permutation(cl[cl["grade"] == g]["slide_id"].tolist())) for g in ("G1", "G2", "G3", "G4")}
    pools["?"] = list(rng.permutation(cl[~cl["grade"].isin(["G1", "G2", "G3", "G4"])]["slide_id"].tolist()))
    chosen = []
    for g in itertools.cycle(["G1", "G2", "G3", "G4", "?"]):
        if len(chosen) >= n_cases or not any(pools.values()):
            break
        if pools[g]:
            chosen.append(pools[g].pop())
    rows = []
    for sid in chosen:
        cdir = results / "cases" / sid
        res = json.loads((cdir / "case_result.json").read_text(encoding="utf-8"))
        tm = pd.read_csv(cdir / "tile_metrics.csv")
        tm = tm[tm["selected"].astype(bool)]
        if "focus_blocks_blurry" in tm:
            tm = tm[(tm["focus_blocks_blurry"] == 0) & (tm["focus_blocks_determinate"] > 0)]
        if "n_nuclei_all" in tm and len(tm) > 4:
            q1, q3 = tm["n_nuclei_all"].quantile([0.25, 0.75])
            tm = tm[tm["n_nuclei_all"].between(q1, q3)]     # tipik yoğunluk (aşırı boş/dolu değil)
        if tm.empty:
            continue
        pick = tm.sample(n=min(tiles_per_case, len(tm)), random_state=int(rng.integers(1e9)))
        src = kp.open_image(res["image_path"])
        ds = res["effective_mpp"] / res["image_resolution_or_mpp"]          # seviye-0 piksel / çıktı pikseli
        full_l0 = int(pick["tile_l0"].iloc[0])
        crop_l0 = full_l0 if not crop_um else min(full_l0, int(round(crop_um / res["image_resolution_or_mpp"])))
        off = (full_l0 - crop_l0) // 2
        n = int(round(crop_l0 / ds))
        for _, t in pick.iterrows():
            t = t.copy()
            t["x_l0"], t["y_l0"], t["tile_l0"] = int(t.x_l0) + off, int(t.y_l0) + off, crop_l0
            rgb = kp.read_tile(src, int(t.x_l0), int(t.y_l0), int(t.tile_l0), n)
            name = f"{sid}__{t.tile_id}.png"
            Image.fromarray(rgb).save(out / name)
            rows.append({"png": name, "case_id": res["case_id"], "slide_id": sid, "grade": res.get("grade"),
                         "tile_id": t.tile_id, "x_l0": int(t.x_l0), "y_l0": int(t.y_l0), "tile_l0": int(t.tile_l0),
                         "tile_px": n, "mpp": res["effective_mpp"]})
        src.close()
    man = pd.DataFrame(rows)
    man.to_csv(out / "tiles_manifest.csv", index=False)
    (out / "QUPATH_ETIKETLEME_REHBERI.md").write_text(GUIDE, encoding="utf-8")
    print(f"{len(man)} tile → {out}  (rehber: QUPATH_ETIKETLEME_REHBERI.md)")
    return man


def evaluate(annot: Path, results: Path | None, segmenters: list[str], cfg: kp.Config, out: Path | None = None,
             make_overlays: bool = True, quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    from PIL import Image
    out = out or annot / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(annot / "tiles_manifest.csv")
    recs = []
    for row in man.to_dict("records"):
        if gt_path(annot, row) is None:
            continue
        rgb = np.asarray(Image.open(annot / row["png"]).convert("RGB"))
        gt, gt_cls = load_gt(annot, row)
        n_gt = len(np.unique(gt)) - 1
        if n_gt < MIN_GT_PER_TILE:
            print(f"  ATLANDI: {row['png']} yalnız {n_gt} çizilmiş çekirdek içeriyor (< {MIN_GT_PER_TILE}; yarım/yanlışlıkla dışa aktarılmış?)")
            continue
        preds = {}
        for sname in segmenters:
            rec, pred = evaluate_tile(rgb, gt, gt_cls, row, results, cfg, sname)
            recs.append({"png": row["png"], "case_id": row["case_id"], "grade": row.get("grade"), "segmenter": sname, **rec})
            preds[sname] = (pred, rec)
        if make_overlays:
            overlay(rgb, gt, preds, out / f"overlay_{Path(row['png']).stem}.png")
    if not recs:
        raise SystemExit("Değerlendirilecek GeoJSON bulunamadı (<png adı>.geojson aynı klasörde olmalı).")
    per = pd.DataFrame(recs)
    summ = summarize(per)
    if out is not None:
        with pd.ExcelWriter(out / "validation_report.xlsx", engine="openpyxl") as xw:
            summ.to_excel(xw, sheet_name="ozet", index=False)
            per.to_excel(xw, sheet_name="tile_bazinda", index=False)
            pd.DataFrame([{"ölçüt": k, "açıklama": v} for k, v in METRIC_DOC.items()]).to_excel(xw, sheet_name="aciklamalar", index=False)
        summary_figure(per, summ, out)
    if not quiet:
        print(summ.to_string(index=False))
    return per, summ


METRIC_DOC = {
    "F1 (mikro)": "Tüm tile'lardaki TP/FP/FN toplamından; IoU>0.5 eşleşme. Proje hedefi ≥0.80 (tespit doğruluğu).",
    "PQ": "Panoptik kalite = SQ × F1; hem tespit hem sınır kalitesi.",
    "SQ": "Eşleşen çiftlerin ortalama IoU'su (sınır doğruluğu).",
    "pixel_dice": "Ön plan (çekirdek pikseli) Dice; proje hedefi ≥0.85.",
    "area_bias_%": "Tahmin medyan alanı / GT medyan alanı − 1 (sistematik büyük/küçük ölçme).",
    "entropy_MAE_<f>": "Tile düzeyi entropide GT'ye göre ortalama mutlak hata (bit) — ANA ÇALIŞMA İÇİN EN KRİTİK ölçüt.",
    "entropy_rho_<f>": "Tile entropisinin GT ile Spearman korelasyonu (sıralamayı koruyor mu?).",
}


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    for s, g in per.groupby("segmenter"):
        tp, fp, fn = g["TP"].sum(), g["FP"].sum(), g["FN"].sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        r = {"segmenter": s, "n_tiles": len(g), "n_gt_nuclei": int(g["n_gt"].sum()),
             "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "F1 (mikro)": f1,
             "F1 (tile ort.)": g["F1"].mean(), "SQ": g["SQ"].mean(), "PQ": g["PQ"].mean(), "pixel_dice": g["pixel_dice"].mean(),
             "area_bias_%": 100 * (g["pred_area_um2_median"] / g["gt_area_um2_median"] - 1).median()}
        for f in kp.MORPH_FEATURES:
            d = g[[f"gt_{f}_entropy", f"pred_{f}_entropy"]].dropna()
            r[f"entropy_MAE_{f}"] = float((d.iloc[:, 1] - d.iloc[:, 0]).abs().mean()) if len(d) else float("nan")
            r[f"entropy_bias_{f}"] = float((d.iloc[:, 1] - d.iloc[:, 0]).mean()) if len(d) else float("nan")
            r[f"entropy_rho_{f}"] = float(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]) if len(d) >= 5 else float("nan")
        for c in [c for c in g.columns if c.startswith("recall_")]:
            r[c] = g[c].mean()
        r["hedef_F1>=0.80"] = "EVET" if f1 >= TARGET_F1 else "HAYIR"
        r["hedef_Dice>=0.85"] = "EVET" if r["pixel_dice"] >= TARGET_DICE else "HAYIR"
        rows.append(r)
    return pd.DataFrame(rows)


def summary_figure(per: pd.DataFrame, summ: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"classical": "#2a78d6", "stardist": "#eb6834"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    mets = ["F1 (mikro)", "PQ", "pixel_dice"]
    x = np.arange(len(mets)); w = 0.8 / max(len(summ), 1)
    for i, (_, r) in enumerate(summ.iterrows()):
        vals = [r[m] for m in mets]
        axs[0].bar(x + i * w, vals, w, color=cols.get(r["segmenter"], "#52514e"), label=r["segmenter"], edgecolor="#fcfcfb", lw=2)
        for xi, v in zip(x, vals):
            axs[0].text(xi + i * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    axs[0].axhline(TARGET_F1, color="#52514e", ls="--", lw=1); axs[0].axhline(TARGET_DICE, color="#52514e", ls=":", lw=1)
    axs[0].set_xticks(x + w * (len(summ) - 1) / 2, mets); axs[0].set_ylim(0, 1.05); axs[0].legend(); axs[0].set_title("Segmentasyon doğruluğu")
    for s, g in per.groupby("segmenter"):
        axs[1].scatter(g["gt_area_um2_entropy"], g["pred_area_um2_entropy"], color=cols.get(s, "#52514e"), s=30, label=s, edgecolor="white")
        axs[2].scatter(g["gt_area_um2_median"], g["pred_area_um2_median"], color=cols.get(s, "#52514e"), s=30, label=s, edgecolor="white")
    for a, t, u in ((axs[1], "Tile alan entropisi: GT vs tahmin", "bit"), (axs[2], "Tile medyan çekirdek alanı", "µm²")):
        lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
        a.plot(lim, lim, color="#52514e", lw=1, ls="--"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"ground truth ({u})"); a.set_ylabel(f"tahmin ({u})"); a.set_title(t); a.legend(fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "validation_summary.png", dpi=130); plt.close(fig)


def tune(annot: Path, results: Path | None, cfg: kp.Config, seed: int = 11, objective: str = "PQ") -> dict:
    """Klasik yöntem için ızgara araması. Vakalar ikiye bölünür: AYAR yarısında en iyi parametre seçilir,
    TEST yarısında raporlanır (seçim yanlılığı olmadan). Sonuç: best_params.json (--config-json ile kullanılır)."""
    man = pd.read_csv(annot / "tiles_manifest.csv")
    have = [r for r in man.to_dict("records") if gt_path(annot, r) is not None]
    from PIL import Image
    cache = {}
    for r in have:
        gt, cls = load_gt(annot, r)
        if len(np.unique(gt)) - 1 < MIN_GT_PER_TILE:
            continue
        cache[r["png"]] = (np.asarray(Image.open(annot / r["png"]).convert("RGB")), gt, cls, r)
    cases = sorted({v[3]["case_id"] for v in cache.values()})
    if len(cases) < 2:
        raise SystemExit("Ayar/test bölmesi için en az 2 vakada etiketli tile gerekir.")
    rng = np.random.default_rng(seed)
    perm = [str(c) for c in rng.permutation(cases)]
    tune_c = set(perm[: len(perm) // 2]); test_c = set(perm[len(perm) // 2:])
    grid = {"nuc_threshold_scale": [0.8, 0.9, 1.0, 1.1, 1.2], "watershed_min_distance_um": [1.5, 2.0, 2.5, 3.0],
            "nuc_min_area_um2": [6.0, 8.0, 10.0]}
    res = []
    for vals in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), vals))
        c = kp.Config(**{**cfg.__dict__, **params, "segmenter": "classical"})
        recs = {"tune": [], "test": []}
        for png, (rgb, gt, cls, r) in cache.items():
            rec, _ = evaluate_tile(rgb, gt, cls, r, results, c, "classical")
            recs["tune" if r["case_id"] in tune_c else "test"].append(rec)
        row = dict(params)
        for part in ("tune", "test"):
            d = pd.DataFrame(recs[part])
            tp, fp, fn = d["TP"].sum(), d["FP"].sum(), d["FN"].sum()
            row[f"{part}_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
            row[f"{part}_PQ"] = d["PQ"].mean()
            row[f"{part}_entropy_MAE_area"] = (d["pred_area_um2_entropy"] - d["gt_area_um2_entropy"]).abs().mean()
        res.append(row)
    df = pd.DataFrame(res).sort_values(f"tune_{objective}", ascending=False)
    best = df.iloc[0].to_dict()
    default = df[(df.nuc_threshold_scale == 1.0) & (df.watershed_min_distance_um == 2.5) & (df.nuc_min_area_um2 == 8.0)].iloc[0].to_dict()
    out = annot / "evaluation"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "tuning_grid.csv", index=False)
    bp = {k: best[k] for k in grid}
    (out / "best_params.json").write_text(json.dumps(bp, indent=2), encoding="utf-8")
    print(f"Ayar vakaları: {sorted(tune_c)} | test vakaları: {sorted(test_c)}")
    print(f"En iyi ({objective}, ayar yarısı): {bp}")
    print(f"  TEST yarısı → F1 {best['test_F1']:.3f}, PQ {best['test_PQ']:.3f}, alan entropisi MAE {best['test_entropy_MAE_area']:.3f} bit")
    print(f"  Varsayılan parametreler TEST → F1 {default['test_F1']:.3f}, PQ {default['test_PQ']:.3f}")
    print(f"→ {out / 'best_params.json'} (kullanmak için: --config-json; seg_hash değişir, batch yeniden çalışır)")
    return {"best": bp, "best_row": best, "default_row": default}


def tune_stardist(annot: Path, results: Path | None, cfg: kp.Config, seed: int = 11,
                  prob_grid=(0.3, 0.4, 0.5, 0.6, 0.692478, 0.8), nms_grid=(0.2, 0.3, 0.4, 0.5)) -> dict:
    """StarDist eşikleri (prob_thresh, nms_thresh) için ızgara araması. Görüntüler HASTA bazında ikiye bölünür:
    AYAR yarısında en yüksek F1 seçilir, TEST yarısında (seçimde kullanılmayan) raporlanır.
    Model yeniden eğitilmez; yalnız karar eşikleri ayarlanır."""
    from PIL import Image
    man = pd.read_csv(annot / "tiles_manifest.csv")
    cache = {}
    for r in man.to_dict("records"):
        if gt_path(annot, r) is None:
            continue
        gt, cls = load_gt(annot, r)
        if len(np.unique(gt)) - 1 < MIN_GT_PER_TILE:
            continue
        cache[r["png"]] = (np.asarray(Image.open(annot / r["png"]).convert("RGB")), gt, cls, r)
    cases = sorted({v[3]["case_id"] for v in cache.values()})
    if len(cases) < 2:
        raise SystemExit("Ayar/test bölmesi için en az 2 hasta gerekir.")
    perm = [str(c) for c in np.random.default_rng(seed).permutation(cases)]
    tune_c, test_c = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
    rows = []
    total = len(prob_grid) * len(nms_grid)
    for k, (pt, nt) in enumerate(itertools.product(prob_grid, nms_grid), 1):
        c = kp.Config(**{**cfg.__dict__, "segmenter": "stardist", "stardist_prob_thresh": float(pt),
                         "stardist_nms_thresh": float(nt)})
        recs = {"tune": [], "test": []}
        for png, (rgb, gt, cls, r) in cache.items():
            rec, _ = evaluate_tile(rgb, gt, cls, r, results, c, "stardist")
            recs["tune" if r["case_id"] in tune_c else "test"].append(rec)
        row = {"prob_thresh": pt, "nms_thresh": nt}
        for part in ("tune", "test"):
            d = pd.DataFrame(recs[part])
            tp, fp, fn = d["TP"].sum(), d["FP"].sum(), d["FN"].sum()
            row[f"{part}_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
            row[f"{part}_dice"] = d["pixel_dice"].mean()
            row[f"{part}_PQ"] = d["PQ"].mean()
            ent = [f"diff_{f}_entropy" for f in kp.MORPH_FEATURES if f"diff_{f}_entropy" in d]
            row[f"{part}_entropy_MAE"] = float(d[ent].abs().mean().mean()) if ent else float("nan")
        rows.append(row)
        print(f"  [{k}/{total}] prob {pt:.2f} · nms {nt:.2f} → ayar F1 {row['tune_F1']:.3f} · test F1 {row['test_F1']:.3f}")
    df = pd.DataFrame(rows)
    df["varsayilan"] = (df.prob_thresh.round(3) == 0.692) & (df.nms_thresh == 0.3)
    df = df.sort_values(["tune_F1", "varsayilan"], ascending=[False, False])   # eşitlikte varsayılan tercih edilir
    best = df.iloc[0].to_dict()
    default = df[df["varsayilan"]].iloc[0].to_dict()
    out = annot / "evaluation"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "stardist_tuning_grid.csv", index=False)
    bp = {"stardist_prob_thresh": float(best["prob_thresh"]), "stardist_nms_thresh": float(best["nms_thresh"])}
    (out / "stardist_best_params.json").write_text(json.dumps(bp, indent=2), encoding="utf-8")
    return {"best": bp, "best_row": best, "default_row": default, "tune_cases": sorted(tune_c), "test_cases": sorted(test_c)}


# -----------------------------------------------------------------------------
# Hazır uzman etiketli veri seti: MoNuSeg (Kumar ve ark. 2017/2020) — TCGA 40x, 1000×1000 görüntüler
# -----------------------------------------------------------------------------
MONUSEG_HF = "https://huggingface.co/datasets/RationAI/MoNuSeg/resolve/main/data/{split}-00000-of-00001.parquet"
MONUSEG_TISSUES = ["Unknown", "Breast", "Kidney", "Liver", "Prostate", "Bladder", "Colon", "Stomach"]
MONUSEG_MPP = 0.25   # 40x tarama; görüntüler yeniden örneklenmeden kesilmiştir (yaklaşık değer, Yöntem'de belirtilir)


def _monuseg_download(cache: Path, split: str) -> Path:
    import urllib.request
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"monuseg_{split}.parquet"
    if not dst.exists() or dst.stat().st_size < 1_000_000:
        url = MONUSEG_HF.format(split=split)
        print(f"İndiriliyor: {url}")
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def prepare_monuseg(out: Path, cache: Path, organ: str = "Kidney", mpp: float = MONUSEG_MPP,
                    parquet_files: dict | None = None) -> pd.DataFrame:
    """MoNuSeg'in (eğitim + test) seçili organ görüntülerini PNG + örnek etiket (<ad>.gt.npz) olarak yazar
    ve tiles_manifest.csv oluşturur → evaluate()/tune() doğrudan kullanır. İnternet gerekir (≈100 MB, bir kez)."""
    import io
    from PIL import Image
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow gerekli:  pip install pyarrow")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "test"):
        pq = (parquet_files or {}).get(split) or _monuseg_download(cache, split)
        df = pd.read_parquet(pq)
        is_org = df["tissue"].map(lambda v: (MONUSEG_TISSUES[int(v)] if not isinstance(v, str) else v) == organ)
        for r in df[is_org].itertuples():
            pid = str(r.patient)
            rgb = np.asarray(Image.open(io.BytesIO(r.image["bytes"])).convert("RGB"))
            lab = np.zeros(rgb.shape[:2], np.int32)
            k = 0
            for inst in r.instances:
                m = np.asarray(Image.open(io.BytesIO(inst["bytes"])))
                m = (m.any(-1) if m.ndim == 3 else m) > 0
                if m.shape != lab.shape or not m.any():
                    continue
                k += 1
                lab[m] = k                          # örtüşmede sonraki çekirdek yazılır (nadir)
            name = f"MoNuSeg_{split}_{pid}.png"
            Image.fromarray(rgb).save(out / name)
            np.savez_compressed(out / (Path(name).stem + ".gt.npz"), labels=lab)
            rows.append({"png": name, "case_id": pid[:12], "slide_id": pid, "grade": f"MoNuSeg-{organ}",
                         "tile_id": "full", "x_l0": 0, "y_l0": 0, "tile_l0": rgb.shape[1], "tile_px": rgb.shape[1],
                         "mpp": mpp, "source": f"MoNuSeg {split}", "n_gt_nuclei": int(len(np.unique(lab)) - 1)})
    man = pd.DataFrame(rows)
    if man.empty:
        raise SystemExit(f"MoNuSeg'de '{organ}' görüntüsü bulunamadı.")
    man.to_csv(out / "tiles_manifest.csv", index=False)
    (out / "KAYNAK.txt").write_text(
        "MoNuSeg (Multi-Organ Nucleus Segmentation) — Kumar N. ve ark., IEEE Trans Med Imaging 2017;36(7):1550-60 "
        "ve 2020;39(5):1380-91. Lisans: CC BY-NC-SA 4.0. Kaynak: huggingface.co/datasets/RationAI/MoNuSeg\n"
        f"Seçilen organ: {organ}; {len(man)} görüntü; {int(man['n_gt_nuclei'].sum())} uzman etiketli çekirdek; "
        f"varsayılan çözünürlük {mpp} µm/px.\n", encoding="utf-8")
    print(f"MoNuSeg {organ}: {len(man)} görüntü, {int(man['n_gt_nuclei'].sum())} etiketli çekirdek → {out}")
    return man


# -----------------------------------------------------------------------------
# Self-test (sentetik ground truth)
# -----------------------------------------------------------------------------
def selftest(workdir: Path | None = None) -> bool:
    import tempfile
    from PIL import Image
    from skimage.measure import find_contours
    workdir = Path(workdir or tempfile.mkdtemp(prefix="kirc_val_"))
    annot = workdir / "annot"
    annot.mkdir(parents=True, exist_ok=True)
    rows = []
    ok_all = True
    for k, (seed, radii) in enumerate([(1, (3, 3)), (2, (2.5, 4.5)), (3, (3, 3)), (4, (2, 5))]):
        img, _, lab = kp._synthetic_he(size=512, mpp=0.5, radii_um=radii, seed=seed, n_nuclei=60, bg_frac=0.0, return_labels=True)
        name = f"SYN{k}__x0_y0.png"
        Image.fromarray(img).save(annot / name)
        feats = []
        for i in range(1, int(lab.max()) + 1):
            m = np.pad(lab == i, 1)
            if not m.any():
                continue
            c = max(find_contours(m.astype(float), 0.5), key=len)
            ring = [[float(q[1] - 1), float(q[0] - 1)] for q in c] + [[float(c[0][1] - 1), float(c[0][0] - 1)]]
            feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                          "properties": {"objectType": "annotation", "classification": {"name": "Tumor"}}})
        (annot / f"SYN{k}__x0_y0.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        rows.append({"png": name, "case_id": f"SYN{k}", "slide_id": f"SYN{k}", "grade": "G1", "tile_id": "x0_y0",
                     "x_l0": 0, "y_l0": 0, "tile_l0": 512, "tile_px": 512, "mpp": 0.5})
    pd.DataFrame(rows).to_csv(annot / "tiles_manifest.csv", index=False)
    segs = ["classical"]
    import importlib.util
    if importlib.util.find_spec("stardist"):
        segs.append("stardist")
    per, summ = evaluate(annot, None, segs, kp.Config(), quiet=True)

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[{'OK ' if cond else 'FAIL'}] {name} {detail}")

    # Mükemmel tahmin = GT → F1 = PQ-SQ = 1
    _, _, lab = kp._synthetic_he(size=512, mpp=0.5, seed=1, n_nuclei=60, bg_frac=0.0, return_labels=True)
    m = match_instances(lab, lab)
    check("GT ile GT karşılaştırması → F1=1, SQ=1, Dice=1", m["F1"] == 1 and abs(m["SQ"] - 1) < 1e-9 and m["pixel_dice"] == 1)
    m0 = match_instances(lab, np.zeros_like(lab))
    check("Boş tahmin → F1=0, recall=0", m0["F1"] == 0 and m0["recall"] == 0)
    shifted = np.roll(lab, 30, axis=1)
    check("Kaydırılmış tahmin → F1 düşük", match_instances(lab, shifted)["F1"] < 0.2)
    c = summ[summ.segmenter == "classical"].iloc[0]
    check("Klasik yöntem sentetik GT'de F1 ≥ 0.9", c["F1 (mikro)"] >= 0.9, f"({c['F1 (mikro)']:.3f})")
    check("Entropi MAE raporlanıyor", np.isfinite(c["entropy_MAE_area_um2"]), f"({c['entropy_MAE_area_um2']:.3f} bit)")
    check("Sınıf bazında duyarlılık hesaplanıyor", "recall_Tumor" in summ.columns)
    t = tune(annot, None, kp.Config())
    check("Ayar/test bölmesi çalışıyor ve best_params.json yazıldı", (annot / "evaluation" / "best_params.json").exists()
          and math.isfinite(t["best_row"]["test_F1"]))
    print(summ[["segmenter", "F1 (mikro)", "PQ", "pixel_dice", "area_bias_%", "entropy_MAE_area_um2"]].to_string(index=False))
    print(("Tüm doğrulama testleri geçti." if ok_all else "BAŞARISIZ test var.") + f" Çıktılar: {workdir}")
    return ok_all


def main(argv=None):
    ap = argparse.ArgumentParser(description="KIRC — segmentasyon doğrulaması (Dice/F1/PQ)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export-tiles")
    p.add_argument("--results", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-cases", type=int, default=10); p.add_argument("--tiles-per-case", type=int, default=2)
    for name in ("evaluate", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--annot", type=Path, required=True)
        p.add_argument("--results", type=Path, help="Pipeline çıktı klasörü (vaka boya parametreleri için)")
        p.add_argument("--config-json", type=Path)
        if name == "evaluate":
            p.add_argument("--segmenters", nargs="+", default=["classical", "stardist"])
    p = sub.add_parser("selftest"); p.add_argument("--workdir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        sys.exit(0 if selftest(a.workdir) else 1)
    if a.cmd == "export-tiles":
        export_tiles(a.results, a.out, a.n_cases, a.tiles_per_case)
        return
    cfg = kp.load_config(a.results, a.config_json, {})
    if a.cmd == "evaluate":
        evaluate(a.annot, a.results, a.segmenters, cfg)
    elif a.cmd == "tune":
        tune(a.annot, a.results, cfg)


if __name__ == "__main__":
    main()
