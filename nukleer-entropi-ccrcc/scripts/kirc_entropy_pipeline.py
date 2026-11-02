#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA-KIRC Nükleer Morfometrik Entropi Pipeline'ı  (v3.0)
========================================================

TÜBİTAK 2209-A: "H&E Boyalı Tümör Dokularında Nükleer Morfometrik Entropi Analizi"

ANA ÖLÇÜT (proje formu ve kontrol listesi Aşama 7):
    Odaklı doku bölgelerinde segmente edilen çekirdeklerin alan, çevre, dairesellik ve
    eksantrisite DAĞILIMLARININ Shannon entropisi:  H = -Σ p_i log2 p_i  (sabit, fiziksel bin'ler)
İKİNCİL: log2(alan) entropisi (boyuttan bağımsız göreli çeşitlilik), en heterojen bölge (tile p90),
    piksel (gri/RGB/H/E) entropileri — bunlar pleomorfizmin eş anlamlısı DEĞİLDİR.

Komutlar (hiçbiri 200 vakayı kendiliğinden başlatmaz):
    python kirc_entropy_pipeline.py selftest
    python kirc_entropy_pipeline.py make-manifest --excel X.xlsx --out OUT                  # GDC indirme listesi
    python kirc_entropy_pipeline.py validate --excel X.xlsx --data-dir DATA --out OUT
    python kirc_entropy_pipeline.py screen   --excel X.xlsx --data-dir DATA --out OUT      # H&E/boya taraması (hızlı)
    python kirc_entropy_pipeline.py single   --excel X.xlsx --data-dir DATA --out OUT --case TCGA-BP-4165
    python kirc_entropy_pipeline.py batch    --excel X.xlsx --data-dir DATA --out OUT --confirm
    python kirc_entropy_pipeline.py entropy  --out OUT [--config-json p.json]               # 2. aşama
    python kirc_entropy_pipeline.py import-qupath --out OUT --slide-id <barcode> --geojson det.geojson
    python kirc_entropy_pipeline.py calibrate-focus --out OUT --slide-id <barcode>
    python kirc_entropy_pipeline.py summarize --out OUT
    python kirc_validate.py ...      # segmentasyon doğrulaması (Dice/F1/PQ)
    python kirc_stats.py ...         # 3. aşama: istatistik + figürler

Segmentasyon: --segmenter classical (Otsu + watershed) | stardist (2D_versatile_he, otomatik ölçekleme)
Uyarı: Bu yazılım tanı koymaz; doku maskesi tümör/normal/nekroz ayrımı yapmaz.
Tümör bölgesi için patolog onaylı ROI (QuPath GeoJSON) --roi-geojson ile verilebilir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PIPELINE_VERSION = "3.0.0"
SEG_VERSION = "2.0"       # segmentasyon/özellik kodu değişince artırılır → 1. aşama yeniden
ENTROPY_VERSION = "2.0"   # entropi kodu değişince artırılır → yalnız 2. aşama yeniden
SUPPORTED_WSI_EXT = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".svslide"}
SUPPORTED_ARRAY_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# TCGA slide barcode: TCGA-BP-4165-01A-01-BS1  (BP = doku kaynak merkezi / TSS)
BARCODE_RE = re.compile(r"^(TCGA-([A-Z0-9]{2})-[A-Z0-9]{4})-(\d{2})([A-Z])-(\d{2})-([A-Z]{2})([A-Z0-9])$")
CASE_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
SUCCESS_STATUSES = {"success", "success_with_warnings"}
MORPH_FEATURES = ("area_um2", "perimeter_um", "circularity", "eccentricity")   # birincil
EXTRA_FEATURES = ("log2_area",)                                               # ikincil (türetilmiş)

log = logging.getLogger("kirc_entropy")


# =============================================================================
# 1. KONFİGÜRASYON  (uzunluklar µm cinsinden → 20x ve 40x slaytlarda aynı fiziksel anlam)
# =============================================================================
@dataclass
class Config:
    """Tüm analiz parametreleri. seg_hash (1. aşama) ve entropy_hash (2. aşama) ayrı izlenir."""
    seed: int = 42
    # --- Çözünürlük ---
    # 0.50 µm/px: 20x ve 40x slaytların ortak ölçeği. Tüm kohort 40x ise (DX slaytlar) 0.25 önerilir.
    # Hedeften KABA slaytlar büyütülmez (yapay çözünürlük yaratılmaz) → RESOLUTION_INSUFFICIENT.
    target_mpp: float = 0.50
    mpp_tolerance: float = 0.10
    tile_size_um: float = 256.0         # 0.5 µm/px'te 512 px
    max_tiles: int | None = None        # None = uygun tüm tile'lar; sayı = seed'li rastgele örneklem
    min_tile_tissue_fraction: float = 0.60
    # --- Doku maskesi (düşük çözünürlük) ---
    mask_downsample: int = 16
    mask_min_saturation: float = 0.05
    mask_close_radius: int = 2
    mask_min_object_um2: float = 20000.0
    mask_max_hole_um2: float = 20000.0
    pen_hue_range: tuple = (0.15, 0.60)
    pen_min_saturation: float = 0.25
    tile_pen_exclude: bool = True       # tile düzeyinde (tam çözünürlük) kalem izi + kenar payı analizden çıkarılır
    tile_pen_margin_um: float = 5.0
    dark_artifact_gray_max: int = 30
    background_gray_min: int = 200
    # --- Boya ayrıştırma (Macenko 2009) ---
    stain_method: str = "macenko"
    macenko_beta: float = 0.15
    macenko_alpha: float = 1.0
    stain_fit_tiles: int = 30
    stain_fit_max_pixels: int = 300000
    stain_min_cosine_to_reference: float = 0.80
    ref_stain_matrix: tuple = ((0.5626, 0.2159), (0.7201, 0.8012), (0.4062, 0.5581))  # sütunlar: H, E (OD)
    ref_max_conc: tuple = (1.9705, 1.0308)
    ref_io: float = 240.0
    # --- H&E / boya taraması (screen + her vakada) ---
    he_min_hue_fraction: float = 0.60   # doku piksellerinin ≥%60'ı pembe-mor tonda olmalı
    dab_max_fraction: float = 0.08      # kahverengi (DAB) piksel oranı > bu → olası IHC
    he_min_cosine_typical: float = 0.90  # Macenko vektörleri referansa bu kadar yakınsa "tipik"
    faded_max_mean_od: float = 0.08     # doku ortalama ||OD|| < bu → soluk (ccRCC doğal olarak soluktur: ilk vaka 0.15)
    # --- Piksel entropisi ---
    pixel_hist_bins: int = 256
    conc_hist_range: tuple = (0.0, 3.0)
    log_base: float = 2.0
    # --- Çekirdek segmentasyonu ---
    segmenter: str = "classical"        # "classical" (Otsu + watershed) | "stardist"
    nuc_smooth_sigma_um: float = 0.0    # >0 sınırları genişletir (sentetik testte +%40 alan)
    nuc_threshold_scale: float = 1.0
    nuc_open_radius_um: float = 0.5
    nuc_min_area_um2: float = 8.0
    nuc_max_area_um2: float = 400.0
    nuc_min_solidity: float = 0.75
    watershed_min_distance_um: float = 2.5
    exclude_border_nuclei: bool = True
    stardist_model: str = "2D_versatile_he"
    stardist_model_mpp: float = 0.25    # modelin eğitim ölçeği (~40x); tile bu ölçeğe ölçeklenir
    stardist_prob_thresh: float | None = None   # None = modelin kendi eşiği
    stardist_nms_thresh: float | None = None
    stardist_norm_percentiles: tuple = (1.0, 99.8)  # SLAYT düzeyinde sabit (tile'dan tile'a değişmez)
    # --- Çekirdek tipi (heuristik; yalnız duyarlılık analizi, doğrulanmış sınıflandırma DEĞİL) ---
    spindle_min_eccentricity: float = 0.95
    spindle_max_minor_axis_um: float = 2.5
    # QuPath'ten sınıflı çekirdek içe aktarıldıysa entropide yalnız bu sınıflar (boş = hepsi)
    nucleus_classes_include: tuple = ()
    # --- Morfometrik entropi (SABİT aralıklar; tüm vakalarda aynı) ---
    feature_bins: dict = field(default_factory=lambda: {
        "area_um2": [0.0, 200.0, 40],     # 5 µm²
        "perimeter_um": [0.0, 80.0, 40],  # 2 µm
        "circularity": [0.0, 1.0, 20],    # 0.05
        "eccentricity": [0.0, 1.0, 20],   # 0.05
        "log2_area": [2.0, 8.0, 30],      # 4–256 µm², 0.2 log2 birimi (boyuttan bağımsız)
    })
    sensitivity_bin_counts: tuple = (10, 20, 40, 80)
    subsample_n: int = 500
    subsample_reps: int = 200
    min_nuclei_case: int = 500
    min_nuclei_tile: int = 30
    hotspot_percentile: float = 90.0    # en heterojen bölge ölçüsü: tile entropilerinin p90'ı
    # --- Odak (bulanıklık) kontrolü: blok düzeyi bant oranı ---
    # oran = sqrt( E[(G_s0 - G_s1)²] / E[(G_s1 - G_s2)²] ); σ'lar µm cinsinden.
    # Eşik TCGA-BP-4165 (20x, JPEG Q70) üzerinde görsel kalibre edildi: ≤0.28 bulanık, ≥0.32 keskin.
    # Farklı tarayıcı/çözünürlük için `calibrate-focus` ile yeniden kontrol edin.
    focus_block_um: float = 64.0
    focus_sigmas_um: tuple = (0.35, 0.75, 2.0)
    focus_min_band_ratio: float = 0.29
    focus_mid_energy_floor: float = 20.0
    focus_exclude: bool = True
    focus_warn_fraction: float = 0.50
    focus_fail_fraction: float = 0.80
    # --- Kalite kontrol ---
    qc_min_tissue_fraction: float = 0.02
    qc_min_tiles: int = 10
    qc_tile_dark_gray: int = 40
    qc_example_tiles: int = 6
    heatmap_entropy_range: tuple = (1.5, 4.0)
    save_nuclei_table: bool = True

    # Türetilmiş piksel değerleri (analiz ölçeğinde)
    def px(self, um: float) -> float:
        return um / self.target_mpp

    @property
    def tile_size_px(self) -> int:
        return int(round(self.tile_size_um / self.target_mpp))

    @property
    def focus_block_px(self) -> int:
        b = int(round(self.focus_block_um / self.target_mpp))
        while self.tile_size_px % b:   # tile'ı tam bölen en yakın blok
            b -= 1
        return max(b, 8)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    def _hash(self, keys, version) -> str:
        d = self.to_dict()
        payload = json.dumps({"v": version, **{k: d[k] for k in sorted(keys)}}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def seg_hash(self) -> str:
        """1. AŞAMA (ağır): segmentasyon + özellik çıkarımı parametreleri."""
        return self._hash(set(self.to_dict()) - ENTROPY_FIELDS - COSMETIC_FIELDS, SEG_VERSION)

    def entropy_hash(self) -> str:
        """2. AŞAMA (hafif): kayıtlı çekirdek tablosundan hesaplanan her şey."""
        return self._hash(ENTROPY_FIELDS, ENTROPY_VERSION)

    def config_hash(self) -> str:
        return self.seg_hash() + "-" + self.entropy_hash()


# Değişirse yalnızca kayıtlı nuclei_features tablosundan yeniden hesaplanan alanlar
ENTROPY_FIELDS = {"feature_bins", "sensitivity_bin_counts", "subsample_n", "subsample_reps", "min_nuclei_case",
                  "min_nuclei_tile", "hotspot_percentile", "nucleus_classes_include", "spindle_min_eccentricity",
                  "spindle_max_minor_axis_um", "focus_exclude"}
# Sonuçları etkilemeyen alanlar
COSMETIC_FIELDS = {"qc_example_tiles", "heatmap_entropy_range", "save_nuclei_table"}


def config_from_dict(d: dict) -> Config:
    cfg = Config()
    for k, v in d.items():
        if not hasattr(cfg, k) or k in ("tile_size_px", "focus_block_px"):
            raise SystemExit(f"Bilinmeyen/türetilmiş config alanı: {k}")
        setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) and isinstance(v, list) else v)
    return cfg


def library_versions() -> dict:
    import cv2
    import scipy
    import skimage
    vers = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-image": skimage.__version__, "opencv": cv2.__version__}
    try:
        import openslide
        vers["openslide-python"] = openslide.__version__
        vers["openslide-library"] = openslide.__library_version__
    except Exception as exc:  # noqa: BLE001
        vers["openslide"] = f"yok ({exc})"
    try:
        import matplotlib
        vers["matplotlib"] = matplotlib.__version__
    except Exception:  # noqa: BLE001
        pass
    return vers


# =============================================================================
# 2. ENTROPİ FONKSİYONLARI (matematiksel çekirdek)
# =============================================================================
def shannon_entropy_from_counts(counts, base: float = 2.0) -> float:
    """H = -Σ p_i log_b p_i ;  p_i = n_i / N.  0·log0 = 0 kabul edilir.
    Boş histogram (N=0) için NaN döner (0 DEĞİL: 'ölçülemedi' ≠ 'heterojenlik yok')."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    if np.any(c < 0) or not np.all(np.isfinite(c)):
        raise ValueError("Histogram sayıları negatif veya sonsuz olamaz.")
    n = c.sum()
    if n <= 0:
        return float("nan")
    p = c[c > 0] / n
    h = -np.sum(p * np.log(p)) / np.log(base)
    return float(max(h, 0.0))  # -0.0 yerine 0.0


def miller_madow_entropy(counts, base: float = 2.0) -> float:
    """Miller–Madow düzeltmesi: H_MM = H_plugin + (K_dolu - 1) / (2N) [nat] → base'e çevrilir.
    Plug-in tahmincinin küçük örneklemdeki aşağı yanlılığını kısmen düzeltir."""
    c = np.asarray(counts, dtype=np.float64).ravel()
    n = c.sum()
    if n <= 0:
        return float("nan")
    k = int(np.count_nonzero(c))
    return shannon_entropy_from_counts(c, base) + (k - 1) / (2.0 * n) / np.log(base)


def fixed_histogram(values, lo: float, hi: float, n_bins: int):
    """Sabit kenarlı histogram. Aralık dışı değerler uç bin'lere KIRPILIR ve sayılır
    (veri kaybı yok, ama kırpma oranı raporlanır)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n_low = int(np.sum(v < lo))
    n_high = int(np.sum(v > hi))
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
    return counts, edges, n_low, n_high


def histogram_entropy(values, lo: float, hi: float, n_bins: int, base: float = 2.0) -> dict:
    counts, _, n_low, n_high = fixed_histogram(values, lo, hi, n_bins)
    n = int(counts.sum())
    h = shannon_entropy_from_counts(counts, base)
    h_max = math.log(n_bins) / math.log(base)
    return {"entropy": h,
            "entropy_norm": h / h_max if n > 0 else float("nan"),  # 0-1 aralığı
            "entropy_mm": miller_madow_entropy(counts, base),
            "n": n, "n_clipped_low": n_low, "n_clipped_high": n_high,
            "n_bins": n_bins, "n_bins_nonzero": int(np.count_nonzero(counts))}


def subsampled_entropy(values, lo, hi, n_bins, n_sub, reps, seed, base=2.0) -> tuple[float, float]:
    """Her vakadan n_sub çekirdek (iadesiz) çekilip entropi hesaplanır, reps kez tekrarlanır.
    Farklı çekirdek sayılarına sahip vakaları eşit örneklem büyüklüğünde karşılaştırmak içindir."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < n_sub:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    hs = np.empty(reps)
    for r in range(reps):
        s = rng.choice(v, size=n_sub, replace=False)
        hs[r] = histogram_entropy(s, lo, hi, n_bins, base)["entropy"]
    return float(hs.mean()), float(hs.std(ddof=1)) if reps > 1 else 0.0


def summary_stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "min", "max", "iqr", "cv")} | {"n": 0}
    q1, q3 = np.percentile(x, [25, 75])
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    return {"mean": mean, "std": std, "median": float(np.median(x)), "min": float(x.min()),
            "max": float(x.max()), "iqr": float(q3 - q1),
            "cv": std / mean if mean not in (0.0,) and np.isfinite(std) else float("nan"), "n": int(x.size)}


# =============================================================================
# 3. ID YÖNETİMİ (case / sample / slide ayrımı) ve EXCEL DOĞRULAMA
# =============================================================================
def clean_id(x) -> str | None:
    """Excel hücresindeki \\t, \\n, boşluk vb. tüm görünmez karakterleri temizler."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = re.sub(r"\s+", "", str(x))
    return s or None


def parse_tcga_barcode(barcode: str | None) -> dict:
    """TCGA-BP-4165-01A-01-BS1 →
       case_id=TCGA-BP-4165 | sample_id=TCGA-BP-4165-01A | sample_type=01 (Primary Solid Tumor)
       portion=01 | slide_section=BS1 | slide_kind=frozen (TS/BS/MS) veya ffpe_diagnostic (DX)
       tss_code=BP (doku kaynak merkezi; boyama/tarayıcı/grade karıştırıcısı olabilir)"""
    out = {"case_id": None, "tss_code": None, "sample_id": None, "sample_type_code": None, "vial": None,
           "portion": None, "slide_section": None, "slide_kind": None, "barcode_valid": False}
    if not barcode:
        return out
    m = BARCODE_RE.match(barcode)
    if not m:
        return out
    case, tss, st, vial, portion, sec, num = m.groups()
    kind = {"TS": "frozen_top", "BS": "frozen_bottom", "MS": "frozen_middle",
            "DX": "ffpe_diagnostic"}.get(sec, "unknown")
    out.update(case_id=case, tss_code=tss, sample_id=f"{case}-{st}{vial}", sample_type_code=st, vial=vial,
               portion=portion, slide_section=sec + num, slide_kind=kind, barcode_valid=True)
    return out


EXCEL_COLUMNS = {  # Excel sütunu -> iç ad
    "No.": "row_no", "Case ID": "case_id_excel", "Grade": "grade", "Grade Group": "grade_group",
    "Selected?": "selected", "Selected Slide ID": "slide_uuid_excel", "File ID": "file_id_excel",
    "Submitter ID": "slide_barcode_excel", "No. of Slides": "n_slides_available",
    "Tumor Cells (%)": "gdc_tumor_cells_pct", "Tumor Nuclei (%)": "gdc_tumor_nuclei_pct",
    "Normal Cells (%)": "gdc_normal_cells_pct", "Stromal Cells (%)": "gdc_stromal_cells_pct",
    "Image Quality": "manual_image_quality", "Nuclear Visibility": "manual_nuclear_visibility",
    "Tumor Area Adequacy": "manual_tumor_area_adequacy",
    "Segmentation Suitability": "manual_segmentation_suitability",
    "Artifact / Necrosis": "manual_artifact_necrosis", "Exclusion Reason": "manual_exclusion_reason",
    "Notes": "manual_notes",
}


def _is_real_text(v) -> bool:
    """'_', '-', 'NA', boş vb. yer tutucular → False."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return str(v).strip().lower() not in ("", "_", "-", "na", "n/a", "none", "nan", "yok", "no")


def load_case_table(xlsx_path: str | Path, sheet: str = "Case_Selection") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Excel'i okur, ID'leri temizler/doğrular. Döner: (vaka tablosu, sorun listesi).
    '_' gibi yer tutucular NaN yapılır (ASLA 0 ile doldurulmaz)."""
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel bulunamadı: {xlsx_path}")
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=object)
    missing = [c for c in ("Case ID", "Selected Slide ID", "File ID", "Submitter ID") if c not in raw.columns]
    if missing:
        raise ValueError(f"Excel'de beklenen sütunlar yok: {missing}. Bulunanlar: {list(raw.columns)}")
    df = raw.rename(columns={k: v for k, v in EXCEL_COLUMNS.items() if k in raw.columns}).copy()
    df["excel_row"] = np.arange(len(df)) + 2  # Excel'deki satır numarası (başlık = 1)
    issues: list[dict] = []

    def add_issue(row, level, code, msg):
        issues.append({"excel_row": int(row["excel_row"]), "row_no": row.get("row_no"),
                       "case_id_excel": row.get("case_id_excel"), "level": level, "code": code, "message": msg})

    for col in ("case_id_excel", "slide_uuid_excel", "file_id_excel", "slide_barcode_excel"):
        orig = df[col].copy()
        df[col] = df[col].map(clean_id)
        changed = orig.astype(str).ne(df[col].astype(str)) & orig.notna()
        for i in df.index[changed]:
            add_issue(df.loc[i], "info", "WHITESPACE_CLEANED", f"{col}: görünmez karakterler temizlendi ({orig[i]!r})")
    for col in ("gdc_tumor_cells_pct", "gdc_tumor_nuclei_pct", "gdc_normal_cells_pct", "gdc_stromal_cells_pct"):
        if col in df:
            df[col] = pd.to_numeric(df[col].replace({"_": np.nan, "-": np.nan, "": np.nan}), errors="coerce")
    for col in ("slide_uuid_excel", "file_id_excel"):
        df[col] = df[col].map(lambda s: s.lower() if isinstance(s, str) else s)

    for i, row in df.iterrows():
        if not row["case_id_excel"] or not CASE_RE.match(row["case_id_excel"]):
            add_issue(row, "error", "BAD_CASE_ID", f"Geçersiz Case ID: {row['case_id_excel']!r}")
        for col, code in (("slide_uuid_excel", "BAD_SLIDE_UUID"), ("file_id_excel", "BAD_FILE_ID")):
            v = row[col]
            if not v or not UUID_RE.match(v):
                add_issue(row, "warning", code, f"{col} geçerli UUID değil: {v!r} (dosya diğer ID ile eşlenmeye çalışılacak)")
        bc = parse_tcga_barcode(row["slide_barcode_excel"])
        if not bc["barcode_valid"]:
            add_issue(row, "warning", "BAD_SUBMITTER_ID",
                      f"Submitter ID geçersiz: {row['slide_barcode_excel']!r} (dosya adından alınmaya çalışılacak)")
        elif bc["case_id"] != row["case_id_excel"]:
            add_issue(row, "error", "CASE_BARCODE_MISMATCH",
                      f"Submitter ID {row['slide_barcode_excel']} Case ID {row['case_id_excel']} ile uyuşmuyor")
        else:
            if bc["sample_type_code"] != "01":
                add_issue(row, "warning", "NOT_PRIMARY_TUMOR", f"Sample type {bc['sample_type_code']} (01 değil)")
            if bc["slide_kind"] != "ffpe_diagnostic":
                add_issue(row, "info", "FROZEN_SECTION_SLIDE",
                          f"{bc['slide_section']}: dondurulmuş kesit (DX/FFPE tanısal slayt değil)")
        sel = str(row.get("selected", "Yes")).strip().lower()
        if sel not in ("yes", "evet"):
            add_issue(row, "info", "NOT_SELECTED", f"Selected? = {row.get('selected')!r}; analiz dışı")
        if _is_real_text(row.get("manual_exclusion_reason")):
            add_issue(row, "info", "MANUAL_EXCLUSION", f"Exclusion Reason = {row.get('manual_exclusion_reason')!r}; analiz dışı")
    df["manual_excluded"] = df.get("manual_exclusion_reason", pd.Series(index=df.index, dtype=object)).map(_is_real_text)

    # Tekrarlar: aynı hasta birden fazla satırda → yalnızca ilk satır analiz edilir
    df["is_duplicate"] = False
    for col, code in (("case_id_excel", "DUPLICATE_CASE"), ("slide_uuid_excel", "DUPLICATE_SLIDE"),
                      ("file_id_excel", "DUPLICATE_FILE")):
        dup = df[col].notna() & df.duplicated(col, keep="first")
        for i in df.index[dup]:
            first = df.index[(df[col] == df.loc[i, col])][0]
            add_issue(df.loc[i], "error", code,
                      f"{col}={df.loc[i, col]} Excel satırı {int(df.loc[first, 'excel_row'])} ile tekrar; bu satır atlanacak")
        df.loc[dup, "is_duplicate"] = True
    issues_df = pd.DataFrame(issues, columns=["excel_row", "row_no", "case_id_excel", "level", "code", "message"])
    return df, issues_df


def build_file_index(data_dir: str | Path) -> pd.DataFrame:
    """data_dir altında tüm WSI dosyalarını tarar. GDC yapısı: <file_id>/<barcode>.<slide_uuid>.svs"""
    data_dir = Path(data_dir)
    rows = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {data_dir}")
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_WSI_EXT | SUPPORTED_ARRAY_EXT:
            continue
        parts = p.name.split(".")
        barcode = parts[0] if len(parts) >= 2 and BARCODE_RE.match(parts[0]) else None
        uuid = parts[1].lower() if len(parts) >= 3 and UUID_RE.match(parts[1].lower()) else None
        parent = p.parent.name.lower()
        rows.append({"path": str(p.resolve()), "file_name": p.name, "ext": p.suffix.lower(),
                     "size_bytes": p.stat().st_size, "barcode_from_name": barcode,
                     "slide_uuid_from_name": uuid, "file_id_from_dir": parent if UUID_RE.match(parent) else None})
    return pd.DataFrame(rows, columns=["path", "file_name", "ext", "size_bytes", "barcode_from_name",
                                       "slide_uuid_from_name", "file_id_from_dir"])


def read_gdc_manifests(data_dir: str | Path) -> dict:
    """gdc-client'ın MANIFEST.txt dosyalarından file_id -> md5 eşlemesi (varsa)."""
    out = {}
    for mf in Path(data_dir).rglob("MANIFEST.txt"):
        try:
            m = pd.read_csv(mf, sep="\t", dtype=str)
            if {"id", "md5"} <= set(m.columns):
                out.update({r["id"].lower(): r["md5"].lower() for _, r in m.iterrows()})
        except Exception as exc:  # noqa: BLE001
            log.warning("MANIFEST okunamadı %s: %s", mf, exc)
    return out


def resolve_case_file(row: pd.Series, index: pd.DataFrame) -> dict:
    """Bir Excel satırını tek bir dosyaya eşler. Öncelik: File ID klasörü > Slide UUID > barcode.
    Dosyadan okunan case ID, Excel case ID ile uyuşmazsa ANALİZ EDİLMEZ."""
    res = {"image_path": None, "match_method": None, "id_warnings": [], "id_error": None}
    cand = pd.DataFrame()
    for method, col, val in (("file_id", "file_id_from_dir", row.get("file_id_excel")),
                             ("slide_uuid", "slide_uuid_from_name", row.get("slide_uuid_excel")),
                             ("barcode", "barcode_from_name", row.get("slide_barcode_excel"))):
        if val and not index.empty:
            cand = index[index[col] == val]
            if len(cand):
                res["match_method"] = method
                break
    if cand.empty:
        res["id_error"] = "MISSING_FILE"
        return res
    if len(cand) > 1:
        res["id_warnings"].append(f"{len(cand)} eşleşen dosya bulundu; ilk dosya kullanıldı")
    f = cand.iloc[0]
    res["image_path"] = f["path"]
    res["slide_uuid_file"] = f["slide_uuid_from_name"]
    res["barcode_file"] = f["barcode_from_name"]
    res["file_id_dir"] = f["file_id_from_dir"]
    bc = parse_tcga_barcode(f["barcode_from_name"])
    if bc["barcode_valid"] and bc["case_id"] != row.get("case_id_excel"):
        res["id_error"] = f"CASE_MISMATCH: dosya {bc['case_id']} ≠ Excel {row.get('case_id_excel')}"
    if f["slide_uuid_from_name"] and row.get("slide_uuid_excel") and f["slide_uuid_from_name"] != row.get("slide_uuid_excel"):
        res["id_warnings"].append(f"Slide UUID uyuşmazlığı: Excel {row.get('slide_uuid_excel')} / dosya {f['slide_uuid_from_name']}")
    if f["barcode_from_name"] and row.get("slide_barcode_excel") != f["barcode_from_name"]:
        res["id_warnings"].append(f"Submitter ID Excel'de {row.get('slide_barcode_excel')!r}; dosya adından {f['barcode_from_name']} alındı")
    if f["file_id_from_dir"] and row.get("file_id_excel") and f["file_id_from_dir"] != row.get("file_id_excel"):
        res["id_warnings"].append(f"File ID uyuşmazlığı: Excel {row.get('file_id_excel')} / klasör {f['file_id_from_dir']}")
    return res


# =============================================================================
# 4. GÖRÜNTÜ KAYNAĞI (OpenSlide ile tile tabanlı; tüm WSI RAM'e yüklenmez)
# =============================================================================
class ImageSource:
    """Ortak arayüz: width, height, mpp, fmt, properties, read_region_l0(), thumbnail()."""
    width: int
    height: int
    mpp: float | None
    fmt: str
    properties: dict

    def read_region_l0(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        raise NotImplementedError

    def thumbnail(self, w: int, h: int) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


class OpenSlideSource(ImageSource):
    def __init__(self, path: str | Path):
        import openslide
        self.slide = openslide.OpenSlide(str(path))
        self.width, self.height = self.slide.dimensions
        p = dict(self.slide.properties)
        self.properties = p
        self.fmt = p.get("openslide.vendor", "unknown")
        mx, my = p.get("openslide.mpp-x"), p.get("openslide.mpp-y")
        self.mpp = None
        self.mpp_note = ""
        if mx and my:
            mx, my = float(mx), float(my)
            if abs(mx - my) / max(mx, my) > 0.01:
                self.mpp_note = f"mpp-x ({mx}) ≠ mpp-y ({my}); ortalama kullanıldı"
            self.mpp = (mx + my) / 2
        self.objective_power = p.get("openslide.objective-power")
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples

    def read_region_l0(self, x, y, w, h):
        rgba = np.asarray(self.slide.read_region((int(x), int(y)), 0, (int(w), int(h))))
        return _rgba_to_rgb_white(rgba)

    def thumbnail(self, w, h):
        level = self.slide.get_best_level_for_downsample(max(self.width / w, self.height / h))
        lw, lh = self.slide.level_dimensions[level]
        rgba = np.asarray(self.slide.read_region((0, 0), level, (lw, lh)))
        import cv2
        return cv2.resize(_rgba_to_rgb_white(rgba), (w, h), interpolation=cv2.INTER_AREA)

    def close(self):
        self.slide.close()


class ArraySource(ImageSource):
    """Küçük (WSI olmayan) PNG/JPEG/TIFF görüntüler için. MPP dosyada yoksa kullanıcı vermelidir;
    pipeline MPP TAHMİN ETMEZ."""
    def __init__(self, path: str | Path, mpp: float | None):
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            self.fmt = (im.format or Path(path).suffix).lower()
            self.arr = np.asarray(im.convert("RGB"))
        self.height, self.width = self.arr.shape[:2]
        self.mpp = mpp
        self.mpp_note = "MPP kullanıcı tarafından verildi" if mpp else ""
        self.properties = {}
        self.objective_power = None
        self.level_dimensions = ((self.width, self.height),)
        self.level_downsamples = (1.0,)

    def read_region_l0(self, x, y, w, h):
        out = np.full((h, w, 3), 255, np.uint8)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, self.width), min(y + h, self.height)
        if x1 > x0 and y1 > y0:
            out[y0 - y:y1 - y, x0 - x:x1 - x] = self.arr[y0:y1, x0:x1]
        return out

    def thumbnail(self, w, h):
        import cv2
        return cv2.resize(self.arr, (w, h), interpolation=cv2.INTER_AREA)


def _rgba_to_rgb_white(rgba: np.ndarray) -> np.ndarray:
    """Şeffaf (slayt dışı) pikselleri beyaza karıştırır."""
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(rgb * a + 255.0 * (1 - a), 0, 255).astype(np.uint8)


def open_image(path: str | Path, mpp: float | None = None) -> ImageSource:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Görüntü yok: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Boş dosya (0 bayt): {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_WSI_EXT | SUPPORTED_ARRAY_EXT:
        raise ValueError(f"Desteklenmeyen uzantı: {ext}")
    if ext in SUPPORTED_WSI_EXT:
        try:
            src = OpenSlideSource(path)
            if mpp is not None and src.mpp is None:
                src.mpp, src.mpp_note = mpp, "MPP kullanıcı tarafından verildi"
            return src
        except Exception as exc:  # noqa: BLE001
            if ext not in SUPPORTED_ARRAY_EXT:
                raise IOError(f"OpenSlide dosyayı açamadı (bozuk olabilir): {exc}") from exc
            log.info("OpenSlide açamadı (%s); düz görüntü olarak deneniyor", exc)
    try:
        return ArraySource(path, mpp)
    except Exception as exc:  # noqa: BLE001
        raise IOError(f"Görüntü okunamadı (bozuk olabilir): {exc}") from exc


def md5sum(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# =============================================================================
# 5. DOKU MASKESİ ve TILE SEÇİMİ
# =============================================================================
def compute_tissue_mask(thumb: np.ndarray, mask_mpp: float | None, cfg: Config) -> tuple[np.ndarray, dict]:
    """Doku maskesi (düşük çözünürlükte):
       1) HSV doygunluk (S) üzerinde Otsu; eşik ≥ mask_min_saturation (gri-beyaz arka plan S≈0)
       2) Yeşil/mavi kalem izi (hue aralığı) ve çok koyu (siyah mürekkep) pikseller dışlanır
       3) Morfolojik kapama + küçük delik doldurma (berrak hücre sitoplazması doku sayılsın)
       4) Küçük parçalar atılır.
    Bu maske DOKU/ARKA PLAN ayırır; tümör/normal/nekroz ayırmaz."""
    import cv2
    from skimage.filters import threshold_otsu
    from scipy import ndimage as ndi
    from skimage.morphology import disk, remove_small_holes

    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat = hsv[..., 0] / 180.0, hsv[..., 1] / 255.0
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    info: dict[str, Any] = {}
    if sat.max() - sat.min() < 1e-6:
        info["sat_threshold"] = float("nan")
        return np.zeros(sat.shape, bool), info | {"note": "sabit doygunluk; doku yok"}
    thr = max(float(threshold_otsu(sat)), cfg.mask_min_saturation)
    info["sat_threshold"] = thr
    tissue = sat > thr
    pen = (hue > cfg.pen_hue_range[0]) & (hue < cfg.pen_hue_range[1]) & (sat > cfg.pen_min_saturation)
    dark = gray < cfg.dark_artifact_gray_max
    info["pen_fraction"] = float(pen.mean())
    info["dark_fraction"] = float(dark.mean())
    tissue &= ~pen & ~dark
    tissue = ndi.binary_closing(np.pad(tissue, cfg.mask_close_radius), structure=disk(cfg.mask_close_radius))[
        cfg.mask_close_radius:-cfg.mask_close_radius, cfg.mask_close_radius:-cfg.mask_close_radius] if cfg.mask_close_radius > 0 else tissue
    px_area_um2 = (mask_mpp ** 2) if mask_mpp else None
    if px_area_um2:
        hole_px = max(int(cfg.mask_max_hole_um2 / px_area_um2), 1)
        obj_px = max(int(cfg.mask_min_object_um2 / px_area_um2), 1)
    else:
        hole_px, obj_px = 64, 64
    if _skimage_has_max_size():
        tissue = remove_small_holes(tissue, max_size=hole_px)
    else:
        tissue = remove_small_holes(tissue, area_threshold=hole_px)
    tissue = _remove_small(tissue, obj_px)
    tissue &= ~pen
    info["tissue_fraction"] = float(tissue.mean())
    return tissue.astype(bool), info


def _skimage_has_max_size() -> bool:
    """skimage ≥0.26 'max_size' parametresine geçti; eski sürümlerle uyumluluk."""
    import inspect
    from skimage.morphology import remove_small_objects
    return "max_size" in inspect.signature(remove_small_objects).parameters


def _remove_small(mask: np.ndarray, n_px: int) -> np.ndarray:
    from skimage.morphology import remove_small_objects
    if _skimage_has_max_size():
        return remove_small_objects(mask, max_size=n_px)
    return remove_small_objects(mask, min_size=n_px + 1)


def load_roi_mask(geojson_path: str | Path, shape: tuple[int, int], sx: float, sy: float) -> np.ndarray:
    """QuPath GeoJSON (seviye-0 piksel koordinatları) → maske ölçeğinde ROI maskesi."""
    from skimage.draw import polygon as draw_polygon
    data = json.loads(Path(geojson_path).read_text())
    feats = data["features"] if data.get("type") == "FeatureCollection" else [data]
    roi = np.zeros(shape, bool)
    for f in feats:
        geom = f.get("geometry", f)
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"] if geom["type"] == "MultiPolygon" else []
        for poly in polys:
            for k, ring in enumerate(poly):
                ring = np.asarray(ring, float)
                rr, cc = draw_polygon(ring[:, 1] / sy, ring[:, 0] / sx, shape)
                roi[rr, cc] = (k == 0)  # ilk halka dış sınır, sonrakiler delik
    return roi


def plan_tiles(src: ImageSource, mask: np.ndarray, sx: float, sy: float, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Seviye-0 üzerinde örtüşmeyen tile ızgarası. Örtüşme olmadığı ve kenara değen çekirdekler
    dışlandığı için bir çekirdek iki kez sayılmaz."""
    scale = cfg.target_mpp / src.mpp
    if abs(scale - 1.0) < cfg.mpp_tolerance:
        scale = 1.0
    tile_l0 = int(round(cfg.tile_size_px * scale))
    eff_mpp = src.mpp * tile_l0 / cfg.tile_size_px
    resolution_ok = scale >= 1.0 - 1e-9   # hedeften kaba slayt büyütülmez
    rows = []
    for ty, y in enumerate(range(0, src.height - tile_l0 + 1, tile_l0)):
        for tx, x in enumerate(range(0, src.width - tile_l0 + 1, tile_l0)):
            my0, my1 = int(y / sy), max(int(math.ceil((y + tile_l0) / sy)), int(y / sy) + 1)
            mx0, mx1 = int(x / sx), max(int(math.ceil((x + tile_l0) / sx)), int(x / sx) + 1)
            frac = float(mask[my0:my1, mx0:mx1].mean()) if mask[my0:my1, mx0:mx1].size else 0.0
            rows.append({"tile_id": f"x{x}_y{y}", "grid_x": tx, "grid_y": ty, "x_l0": x, "y_l0": y,
                         "tile_l0": tile_l0, "mask_tissue_fraction": frac})
    tiles = pd.DataFrame(rows)
    if tiles.empty:
        return tiles, {"scale": scale, "tile_l0": tile_l0, "effective_mpp": eff_mpp, "n_grid": 0, "n_eligible": 0,
                       "resolution_ok": resolution_ok}
    tiles["eligible"] = tiles["mask_tissue_fraction"] >= cfg.min_tile_tissue_fraction
    elig = tiles.index[tiles["eligible"]].to_numpy()
    if cfg.max_tiles is not None and len(elig) > cfg.max_tiles:
        rng = np.random.default_rng(cfg.seed)
        elig = np.sort(rng.choice(elig, size=cfg.max_tiles, replace=False))
    tiles["selected"] = False
    tiles.loc[elig, "selected"] = True
    return tiles, {"scale": scale, "tile_l0": tile_l0, "effective_mpp": eff_mpp, "resolution_ok": resolution_ok,
                   "n_grid": int(len(tiles)), "n_eligible": int(tiles["eligible"].sum()),
                   "n_selected": int(tiles["selected"].sum())}


def read_tile(src: ImageSource, x: int, y: int, tile_l0: int, out_px: int) -> np.ndarray:
    import cv2
    rgb = src.read_region_l0(x, y, tile_l0, tile_l0)
    if tile_l0 != out_px:
        rgb = cv2.resize(rgb, (out_px, out_px), interpolation=cv2.INTER_AREA)
    return rgb


def tile_tissue_mask(mask: np.ndarray, x, y, tile_l0, out_px, sx, sy) -> np.ndarray:
    import cv2
    my0, my1 = int(y / sy), max(int(math.ceil((y + tile_l0) / sy)), int(y / sy) + 1)
    mx0, mx1 = int(x / sx), max(int(math.ceil((x + tile_l0) / sx)), int(x / sx) + 1)
    crop = mask[my0:my1, mx0:mx1].astype(np.uint8)
    return cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_NEAREST).astype(bool)


# =============================================================================
# 6. RENK AYRIŞTIRMA (Macenko) ve NORMALİZASYON
# =============================================================================
def rgb_to_od(rgb: np.ndarray, io: np.ndarray) -> np.ndarray:
    """Beer–Lambert: OD = -log10(I / Io). Io = slayta özgü arka plan ışık şiddeti."""
    i = np.clip(rgb.astype(np.float32), 1.0, None)
    return np.clip(-np.log10(i / io.astype(np.float32)), 0.0, None)


def estimate_background_io(thumb: np.ndarray, mask: np.ndarray, cfg: Config) -> np.ndarray:
    bg = thumb[~mask]
    if bg.size:
        bg = bg[bg.mean(axis=1) >= cfg.background_gray_min]
    if bg.size < 100:
        return np.array([255.0, 255.0, 255.0])
    return np.clip(np.median(bg, axis=0).astype(np.float64), 200.0, 255.0)


def macenko_stain_matrix(od: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """Macenko ve ark. (2009). od: (N,3). Döner: 3x2 matris (sütun 0 = H, sütun 1 = E), birim vektörler.
    Boyalı piksel seçimi: ||OD||₂ > beta. (Her kanalda OD>beta koşulu soluk eozinli pikselleri dışlar,
    çünkü eozinin kırmızı kanal OD'si düşüktür; bu durumda model yalnız çekirdeklerden öğrenir.)"""
    od = od[np.linalg.norm(od, axis=1) > beta]
    if od.shape[0] < 1000:
        raise ValueError(f"Macenko için yeterli boyalı piksel yok ({od.shape[0]})")
    _, eigvecs = np.linalg.eigh(np.cov(od.T))
    plane = eigvecs[:, 1:3]                      # en büyük iki özvektör
    proj = od @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    v1 = v1 if v1.sum() > 0 else -v1
    v2 = v2 if v2.sum() > 0 else -v2
    he = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)  # H: kırmızı OD'si büyük
    return he / np.linalg.norm(he, axis=0, keepdims=True)


def fit_stain_model(src, tiles, mask, sx, sy, info_tiles, io, cfg) -> dict:
    """Slayta özgü boya matrisi ve konsantrasyon ölçekleri (Macenko normalizasyonu).
    Normalize konsantrasyon: C_norm = C × (maxC_ref / maxC_slayt), maxC = 99. persentil."""
    sel = tiles[tiles["selected"]]
    rng = np.random.default_rng(cfg.seed)
    pick = sel.sample(n=min(cfg.stain_fit_tiles, len(sel)), random_state=cfg.seed) if len(sel) else sel
    ods, rgbs = [], []
    for _, t in pick.iterrows():
        rgb = read_tile(src, t.x_l0, t.y_l0, t.tile_l0, cfg.tile_size_px)
        tm = tile_tissue_mask(mask, t.x_l0, t.y_l0, t.tile_l0, cfg.tile_size_px, sx, sy)
        ods.append(rgb_to_od(rgb, io)[tm])
        rgbs.append(rgb[tm])
    od = np.concatenate(ods) if ods else np.zeros((0, 3), np.float32)
    rgb_px = np.concatenate(rgbs) if rgbs else np.zeros((0, 3), np.uint8)
    if od.shape[0] > cfg.stain_fit_max_pixels:
        idx = rng.choice(od.shape[0], cfg.stain_fit_max_pixels, replace=False)
        od, rgb_px = od[idx], rgb_px[idx]
    ref = np.asarray(cfg.ref_stain_matrix, float)
    ref = ref / np.linalg.norm(ref, axis=0, keepdims=True)
    out = {"stain_fit_pixels": int(od.shape[0]), "stain_method_used": "reference", "stain_warning": ""}
    he = ref
    if cfg.stain_method == "macenko":
        try:
            he_fit = macenko_stain_matrix(od, cfg.macenko_beta, cfg.macenko_alpha)
            cos_h, cos_e = float(he_fit[:, 0] @ ref[:, 0]), float(he_fit[:, 1] @ ref[:, 1])
            out.update(cos_h_to_ref=cos_h, cos_e_to_ref=cos_e)
            if min(cos_h, cos_e) >= cfg.stain_min_cosine_to_reference:
                he, out["stain_method_used"] = he_fit, "macenko"
            else:
                out["stain_warning"] = (f"Macenko vektörleri H&E referansından uzak (cos H={cos_h:.2f}, "
                                        f"E={cos_e:.2f}); referans matris kullanıldı. H&E olmayan boya olabilir.")
        except Exception as exc:  # noqa: BLE001
            out["stain_warning"] = f"Macenko başarısız ({exc}); referans matris kullanıldı"
    pinv = np.linalg.pinv(he)
    stained = od[np.linalg.norm(od, axis=1) > cfg.macenko_beta] if od.size else od
    if stained.shape[0] < 100:
        raise ValueError("Boya modeli için yeterli boyalı piksel yok")
    conc = stained @ pinv.T
    max_c = np.percentile(conc, 99, axis=0)
    max_c = np.where(max_c > 1e-6, max_c, 1.0)
    scale = np.asarray(cfg.ref_max_conc) / max_c
    # Çekirdek eşiği: boyalı (||OD||>beta) doku piksellerindeki normalize H üzerinde slayt düzeyi Otsu.
    # Berrak sitoplazma/cam (H≈0) dışlanır; böylece Otsu 'çekirdek vs. eozinofilik zemin' ayrımını yapar.
    from skimage.filters import threshold_otsu
    h_norm = np.clip(conc[:, 0] * scale[0], 0, None)
    thr = float(threshold_otsu(h_norm)) * cfg.nuc_threshold_scale
    # StarDist için SLAYT düzeyi sabit yoğunluk normalizasyonu (tile içeriğine göre değişmez)
    plo, phi_ = cfg.stardist_norm_percentiles
    sd_lo = np.percentile(rgb_px, plo, axis=0) if len(rgb_px) else np.zeros(3)
    sd_hi = np.percentile(rgb_px, phi_, axis=0) if len(rgb_px) else np.full(3, 255.0)
    # Renk normalizasyonunun etkisi (kohort düzeyinde varyans azalması kirc_stats.py'de hesaplanır):
    ref_n = ref  # birim vektörlü referans
    raw_mean_od = stained.mean(axis=0)                       # normalizasyon ÖNCESİ ortalama OD (R,G,B)
    norm_mean_od = ref_n @ np.clip(conc * scale, 0, None).mean(axis=0)  # SONRASI (referans uzayında)
    out.update(stain_matrix=he.tolist(), pinv=pinv, conc_scale=scale.tolist(), max_conc_slide=max_c.tolist(),
               nuclear_threshold=thr, io=io.tolist(), sd_lo=sd_lo.tolist(), sd_hi=sd_hi.tolist(),
               raw_mean_od=raw_mean_od.tolist(), norm_mean_od=norm_mean_od.tolist())
    return out


# -----------------------------------------------------------------------------
# 6b. H&E / BOYA TARAMASI (küçük resim düzeyinde; saniyeler)
# -----------------------------------------------------------------------------
STAIN_CLASSES = {
    "HE_TYPICAL": "Tipik H&E",
    "HE_ATYPICAL": "H&E, atipik renk (normalizasyonla analiz edilir; görsel kontrol önerilir)",
    "HE_FADED": "H&E, soluk boyama (segmentasyon güvenilirliği düşebilir)",
    "POSSIBLE_IHC_DAB": "Olası immünohistokimya (kahverengi DAB) — H&E değil",
    "NOT_HE_LIKE": "Renk dağılımı H&E ile uyumsuz",
    "NO_TISSUE": "Değerlendirilebilir doku yok",
}
STAIN_FAIL_CLASSES = {"POSSIBLE_IHC_DAB", "NOT_HE_LIKE", "NO_TISSUE"}


def screen_stain(thumb: np.ndarray, mask: np.ndarray, io: np.ndarray, cfg: Config) -> dict:
    """Bir slaytın H&E olup olmadığını ve boyama kalitesini küçük resimden değerlendirir.
    Ölçütler: (1) doku piksellerinin pembe-mor ton oranı, (2) kahverengi (DAB) piksel oranı,
    (3) ortalama optik yoğunluk (soluk boyama), (4) Macenko vektörlerinin H&E referansına benzerliği."""
    import cv2
    out: dict[str, Any] = {"stain_class": "NO_TISSUE"}
    px = thumb[mask]
    if len(px) < 500:
        return out
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0] / 180.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    colored = sat > 0.10
    nc = int(colored.sum())
    he_hue = ((hue >= 0.70) | (hue <= 0.03)) & colored        # mor (H) – pembe/kırmızı (E)
    brown = (hue > 0.03) & (hue <= 0.12) & (sat > 0.25) & (val < 0.85)
    od = rgb_to_od(px, io)
    odn = np.linalg.norm(od, axis=1)
    out.update(screen_tissue_px=int(len(px)), screen_colored_frac=nc / len(px),
               he_hue_fraction=float(he_hue.sum() / max(nc, 1)), dab_fraction=float(brown.sum() / max(nc, 1)),
               mean_od=float(odn.mean()), mean_saturation=float(sat.mean()))
    ref = np.asarray(cfg.ref_stain_matrix, float)
    ref = ref / np.linalg.norm(ref, axis=0, keepdims=True)
    try:
        he = macenko_stain_matrix(od, cfg.macenko_beta, cfg.macenko_alpha)
        out["screen_cos_h"] = float(he[:, 0] @ ref[:, 0])
        out["screen_cos_e"] = float(he[:, 1] @ ref[:, 1])
        out["screen_he_angle_deg"] = float(np.degrees(np.arccos(np.clip(he[:, 0] @ he[:, 1], -1, 1))))
    except Exception as exc:  # noqa: BLE001
        out["screen_cos_h"] = out["screen_cos_e"] = float("nan")
        out["screen_note"] = f"Macenko yapılamadı: {exc}"
    cos_min = np.nanmin([out["screen_cos_h"], out["screen_cos_e"]]) if np.isfinite(out["screen_cos_h"]) else 0.0
    if out["dab_fraction"] > cfg.dab_max_fraction:
        cls = "POSSIBLE_IHC_DAB"
    elif nc < 0.2 * len(px) and out["mean_od"] < cfg.faded_max_mean_od:
        cls = "HE_FADED"
    elif out["he_hue_fraction"] < cfg.he_min_hue_fraction:
        cls = "NOT_HE_LIKE"
    elif out["mean_od"] < cfg.faded_max_mean_od:
        cls = "HE_FADED"
    elif cos_min < cfg.he_min_cosine_typical:
        cls = "HE_ATYPICAL"
    else:
        cls = "HE_TYPICAL"
    out["stain_class"] = cls
    out["stain_class_tr"] = STAIN_CLASSES[cls]
    return out


def deconvolve(rgb: np.ndarray, io: np.ndarray, pinv: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    od = rgb_to_od(rgb, io)
    c = od.reshape(-1, 3) @ pinv.T * scale
    h = c[:, 0].reshape(rgb.shape[:2])
    e = c[:, 1].reshape(rgb.shape[:2])
    return np.clip(h, 0, None), np.clip(e, 0, None)


def reconstruct_normalized_rgb(h: np.ndarray, e: np.ndarray, cfg: Config) -> np.ndarray:
    ref = np.asarray(cfg.ref_stain_matrix, float)
    ref = ref / np.linalg.norm(ref, axis=0, keepdims=True)
    c = np.stack([h.ravel(), e.ravel()], axis=1)
    rgb = cfg.ref_io * np.power(10.0, -(c @ ref.T))
    return np.clip(rgb, 0, 255).reshape(*h.shape, 3).astype(np.uint8)


# =============================================================================
# 7. TILE İŞLEME: piksel histogramları + çekirdek segmentasyonu + morfometri
# =============================================================================
def segment_nuclei_classical(h: np.ndarray, tissue: np.ndarray, thr: float, mpp: float, cfg: Config) -> np.ndarray:
    """Klasik: slayt düzeyi Otsu eşiği (normalize H) → açma → küçük nesne/delik temizliği →
    distance transform → yerel maksimumlar → watershed (yapışık çekirdekleri ayırma)."""
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian
    from skimage.measure import label
    from skimage.morphology import disk
    from skimage.segmentation import watershed

    sig = cfg.nuc_smooth_sigma_um / mpp
    hs = gaussian(h, sigma=sig, preserve_range=True) if sig > 0 else h
    binary = (hs > thr) & tissue
    r_open = int(round(cfg.nuc_open_radius_um / mpp))
    if r_open > 0:
        binary = ndi.binary_opening(binary, structure=disk(r_open))
    min_px = max(int(cfg.nuc_min_area_um2 / mpp ** 2 * 0.5), 1)
    binary = _remove_small(binary, min_px)
    binary = ndi.binary_fill_holes(binary)
    if not binary.any():
        return np.zeros(h.shape, np.int32)
    comp = label(binary)
    dist = ndi.distance_transform_edt(binary)
    md = max(int(round(cfg.watershed_min_distance_um / mpp)), 1)
    peaks = peak_local_max(dist, min_distance=md, labels=comp, exclude_border=False)
    markers = np.zeros(h.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    ws = watershed(-dist, markers, mask=binary)
    orphan = binary & (ws == 0)  # tepe noktası bulunamayan bileşenler korunur
    if orphan.any():
        ol = label(orphan)
        ws[orphan] = ol[orphan] + ws.max()
    return ws.astype(np.int32)


_STARDIST_CACHE: dict = {}


def _get_stardist(name: str):
    if name not in _STARDIST_CACHE:
        try:
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            from stardist.models import StarDist2D
        except ImportError as exc:
            raise ImportError("StarDist kurulu değil: pip install stardist tensorflow  "
                              "(GPU yoksa tensorflow-cpu). Ya da --segmenter classical kullanın.") from exc
        _STARDIST_CACHE[name] = StarDist2D.from_pretrained(name)
        if _STARDIST_CACHE[name] is None:
            raise ValueError(f"StarDist modeli bulunamadı: {name}")
    return _STARDIST_CACHE[name]


def segment_nuclei_stardist(rgb: np.ndarray, tissue: np.ndarray, ctx: dict, cfg: Config) -> np.ndarray:
    """StarDist (Schmidt ve ark. 2018), '2D_versatile_he' modeli.
    • Model ~0.25 µm/px (40x) ile eğitildi → tile f = mpp / 0.25 kat büyütülerek tahmin, etiketler geri küçültülür.
      (Ölçeklenmeden 0.5 µm/px'te çekirdeklerin çoğu kaçırılır: test tile'ında 41'e karşı 178.)
    • Yoğunluk normalizasyonu SLAYT düzeyinde sabit persentillerle (tile içeriğine göre değişmez).
    • Not: yıldız-konveks şekil önceli konturları düzgünleştirir → düzensizlik hafif düşük ölçülebilir."""
    import cv2
    from skimage.measure import regionprops_table
    model = _get_stardist(cfg.stardist_model)
    mpp = ctx["mpp"]
    f = mpp / cfg.stardist_model_mpp
    if abs(f - 1.0) < 0.1:
        f = 1.0
    lo, hi = np.asarray(ctx["sd_lo"], np.float32), np.asarray(ctx["sd_hi"], np.float32)
    img = (rgb.astype(np.float32) - lo) / np.maximum(hi - lo, 1e-3)
    hh, ww = rgb.shape[:2]
    if f != 1.0:
        img = cv2.resize(img, (int(round(ww * f)), int(round(hh * f))), interpolation=cv2.INTER_CUBIC)
    kw = {}
    if cfg.stardist_prob_thresh is not None:
        kw["prob_thresh"] = cfg.stardist_prob_thresh
    if cfg.stardist_nms_thresh is not None:
        kw["nms_thresh"] = cfg.stardist_nms_thresh
    lab, _ = model.predict_instances(img, verbose=False, **kw)
    if f != 1.0:
        lab = cv2.resize(lab.astype(np.float32), (ww, hh), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    lab = lab.astype(np.int32)
    if lab.max() == 0:
        return lab
    pr = regionprops_table(lab, properties=("label", "centroid"))
    cy = np.clip(np.round(pr["centroid-0"]).astype(int), 0, hh - 1)
    cx = np.clip(np.round(pr["centroid-1"]).astype(int), 0, ww - 1)
    drop = pr["label"][~tissue[cy, cx]]   # merkezi doku dışında kalan nesneler
    if len(drop):
        lab[np.isin(lab, drop)] = 0
    return lab


def segment_nuclei(rgb: np.ndarray, h: np.ndarray, tissue: np.ndarray, ctx: dict, cfg: Config) -> np.ndarray:
    if cfg.segmenter == "classical":
        return segment_nuclei_classical(h, tissue, ctx["thr"], ctx["mpp"], cfg)
    if cfg.segmenter == "stardist":
        return segment_nuclei_stardist(rgb, tissue, ctx, cfg)
    raise ValueError(f"Bilinmeyen segmenter: {cfg.segmenter}")


def measure_nuclei(labels: np.ndarray, h: np.ndarray, mpp: float, cfg: Config) -> tuple[pd.DataFrame, dict]:
    from skimage.measure import regionprops_table
    excl = {"border": 0, "area_small": 0, "area_large": 0, "low_solidity": 0, "zero_perimeter": 0}
    if labels.max() == 0:
        return pd.DataFrame(), excl
    props = regionprops_table(labels, intensity_image=h, properties=(
        "label", "area", "perimeter_crofton", "eccentricity", "solidity", "bbox", "centroid",
        "axis_major_length", "axis_minor_length", "intensity_mean"))
    d = pd.DataFrame(props)
    hh, ww = labels.shape
    keep = np.ones(len(d), bool)
    if cfg.exclude_border_nuclei:
        border = (d["bbox-0"] == 0) | (d["bbox-1"] == 0) | (d["bbox-2"] == hh) | (d["bbox-3"] == ww)
        excl["border"] = int(border.sum()); keep &= ~border
    area_um2 = d["area"] * mpp ** 2
    small, large = area_um2 < cfg.nuc_min_area_um2, area_um2 > cfg.nuc_max_area_um2
    excl["area_small"] = int((small & keep).sum()); keep &= ~small
    excl["area_large"] = int((large & keep).sum()); keep &= ~large
    lows = d["solidity"] < cfg.nuc_min_solidity
    excl["low_solidity"] = int((lows & keep).sum()); keep &= ~lows
    zp = d["perimeter_crofton"] <= 0
    excl["zero_perimeter"] = int((zp & keep).sum()); keep &= ~zp
    d = d[keep]
    out = pd.DataFrame({
        "centroid_y_px": d["centroid-0"], "centroid_x_px": d["centroid-1"],
        "area_um2": d["area"] * mpp ** 2,
        # Crofton çevre tahmincisi: skimage 'perimeter' küçük çekirdeklerde çevreyi %9'a kadar düşük,
        # büyüklerde %3 yüksek ölçer → boyuta bağlı dairesellik yanlılığı (grade ile karışır). Crofton ≤ ~%1.5.
        "perimeter_um": d["perimeter_crofton"] * mpp,
        "circularity": 4 * np.pi * d["area"] / d["perimeter_crofton"] ** 2,  # boyutsuz; >1 değerler raporlanır
        "eccentricity": d["eccentricity"],
        "solidity": d["solidity"],
        "major_axis_um": d["axis_major_length"] * mpp, "minor_axis_um": d["axis_minor_length"] * mpp,
        "h_mean": d["intensity_mean"],
    })
    return out.reset_index(drop=True), excl


def block_focus(gray: np.ndarray, tissue: np.ndarray, cfg: Config, mpp: float | None = None) -> dict:
    """Blok düzeyi odak ölçümü (bant oranı). Döner: oran, belirlenebilirlik ve bulanık blok matrisleri
    ile piksel düzeyinde bulanık maske. σ'lar µm → piksel (gerçek analiz ölçeği)."""
    import cv2
    mpp = mpp or cfg.target_mpp
    s0, s1, s2 = (s / mpp for s in cfg.focus_sigmas_um)
    g = gray.astype(np.float32)
    b0, b1, b2 = (cv2.GaussianBlur(g, (0, 0), s) for s in (s0, s1, s2))
    hi_e, mid_e = (b0 - b1) ** 2, (b1 - b2) ** 2
    B = cfg.focus_block_px
    ny, nx = g.shape[0] // B, g.shape[1] // B

    def blk(a):
        return a[:ny * B, :nx * B].reshape(ny, B, nx, B).mean(axis=(1, 3))

    hi_b, mid_b, tis_b = blk(hi_e), blk(mid_e), blk(tissue.astype(np.float32))
    ratio = np.sqrt(hi_b / np.maximum(mid_b, 1e-9))
    determinate = (mid_b >= cfg.focus_mid_energy_floor) & (tis_b >= 0.25)
    blurry = determinate & (ratio < cfg.focus_min_band_ratio)
    blurry_px = np.zeros(g.shape, bool)
    blurry_px[:ny * B, :nx * B] = np.kron(blurry, np.ones((B, B), bool))
    return {"ratio": ratio, "determinate": determinate, "blurry": blurry, "blurry_px": blurry_px}


def process_tile(rgb, tissue, ctx, cfg, keep_debug=False) -> dict:
    """Bir tile: odak blokları → (odaklı) doku pikselleri histogramları → çekirdek segmentasyonu
    (TÜM dokuda; çekirdekler blok sınırında kesilmesin diye) → çekirdeklere odak etiketi."""
    import cv2
    io, pinv, scale, mpp = ctx["io"], ctx["pinv"], ctx["scale"], ctx["mpp"]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # Y = 0.299R + 0.587G + 0.114B
    pen_frac = 0.0
    if cfg.tile_pen_exclude:   # yeşil/mavi kalem: H&E dokusunun tonu (≈0.68–0.95) bu aralığa girmez
        from scipy import ndimage as ndi
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, sat = hsv[..., 0] / 180.0, hsv[..., 1] / 255.0
        pen = (hue > cfg.pen_hue_range[0]) & (hue < cfg.pen_hue_range[1]) & (sat > cfg.pen_min_saturation)
        if pen.any():
            pen = _remove_small(pen, max(int(50.0 / mpp ** 2), 1))      # < 50 µm² lekeler kalem sayılmaz
        if pen.any():
            pen = ndi.binary_dilation(pen, iterations=max(int(round(cfg.tile_pen_margin_um / mpp)), 1))
            pen_frac = float(pen.mean())
            tissue = tissue & ~pen
    h, e = deconvolve(rgb, io, pinv, scale)
    nb = cfg.pixel_hist_bins
    lo, hi = cfg.conc_hist_range
    foc = block_focus(gray, tissue, cfg, mpp)

    def qhist(x):
        idx = np.clip(((x - lo) / (hi - lo) * nb).astype(np.int64), 0, nb - 1)
        return np.bincount(idx, minlength=nb)

    t = tissue & ~foc["blurry_px"] if cfg.focus_exclude else tissue
    hist = {"gray": np.bincount(gray[t], minlength=256),
            "red": np.bincount(rgb[..., 0][t], minlength=256),
            "green": np.bincount(rgb[..., 1][t], minlength=256),
            "blue": np.bincount(rgb[..., 2][t], minlength=256),
            "hematoxylin": qhist(h[t]), "eosin": qhist(e[t])}
    n_t = int(t.sum())
    labels = segment_nuclei(rgb, h, tissue, ctx, cfg)
    nuc, excl = measure_nuclei(labels, h, mpp, cfg)
    B = cfg.focus_block_px
    if len(nuc):
        by = np.clip((nuc["centroid_y_px"].to_numpy() // B).astype(int), 0, foc["ratio"].shape[0] - 1)
        bx = np.clip((nuc["centroid_x_px"].to_numpy() // B).astype(int), 0, foc["ratio"].shape[1] - 1)
        nuc["focus_ratio"] = foc["ratio"][by, bx]
        nuc["in_focus"] = ~foc["blurry"][by, bx]
    else:
        nuc = nuc.assign(focus_ratio=pd.Series(dtype=float), in_focus=pd.Series(dtype=bool))
    det = foc["determinate"]
    m = {"tissue_pixels_total": int(tissue.sum()), "tissue_pixels": n_t, "pen_fraction": pen_frac,
         "focus_ratio_median": float(np.median(foc["ratio"][det])) if det.any() else float("nan"),
         "focus_blocks_determinate": int(det.sum()), "focus_blocks_blurry": int(foc["blurry"].sum()),
         "gray_entropy": shannon_entropy_from_counts(hist["gray"], cfg.log_base),
         "h_entropy": shannon_entropy_from_counts(hist["hematoxylin"], cfg.log_base),
         "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
         "dark_fraction": float((gray[tissue] < cfg.qc_tile_dark_gray).mean()) if tissue.any() else float("nan"),
         "h_clip_high": int((h[t] > hi).sum()), "e_clip_high": int((e[t] > hi).sum()),
         "n_nuclei_all": int(len(nuc)), "n_nuclei_blurry": int((~nuc["in_focus"]).sum()) if len(nuc) else 0,
         **{f"excl_{k}": v for k, v in excl.items()}}
    out = {"metrics": m, "hist": hist, "nuclei": nuc,
           "blocks": {"ratio": foc["ratio"], "determinate": foc["determinate"], "blurry": foc["blurry"]}}
    if keep_debug:
        out["debug"] = {"rgb": rgb, "norm": reconstruct_normalized_rgb(h, e, cfg), "h": h, "labels": labels,
                        "blurry": foc["blurry"], "ratio": foc["ratio"]}
    return out


# =============================================================================
# 7b. 2. AŞAMA: KAYITLI ÇEKİRDEK TABLOSUNDAN ENTROPİ (slayt yeniden okunmaz)
# =============================================================================
def _entropy_block(nuc: pd.DataFrame, cfg: Config, suffix: str = "", full: bool = True) -> dict:
    """Bir çekirdek kümesi için vaka düzeyi entropiler (birincil 4 + log2_area)."""
    res: dict[str, Any] = {}
    norms = []
    for i, f in enumerate(MORPH_FEATURES + EXTRA_FEATURES):
        if f not in cfg.feature_bins or f not in nuc:
            continue
        lo, hi, nb = cfg.feature_bins[f]
        he_ = histogram_entropy(nuc[f], lo, hi, int(nb), cfg.log_base)
        res[f"{f}_entropy{suffix}"] = he_["entropy"]
        if not full:
            continue
        res[f"{f}_entropy_norm"] = he_["entropy_norm"]
        res[f"{f}_entropy_mm"] = he_["entropy_mm"]
        res[f"{f}_bins_nonzero"] = he_["n_bins_nonzero"]
        res[f"{f}_frac_clipped"] = (he_["n_clipped_low"] + he_["n_clipped_high"]) / he_["n"] if he_["n"] else float("nan")
        m_, s_ = subsampled_entropy(nuc[f], lo, hi, int(nb), cfg.subsample_n, cfg.subsample_reps, cfg.seed + i, cfg.log_base)
        res[f"{f}_entropy_n{cfg.subsample_n}"] = m_
        res[f"{f}_entropy_n{cfg.subsample_n}_sd"] = s_
        for nb2 in cfg.sensitivity_bin_counts:
            res[f"{f}_entropy_b{nb2}"] = histogram_entropy(nuc[f], lo, hi, int(nb2), cfg.log_base)["entropy"]
        st = summary_stats(nuc[f])
        for k_ in ("mean", "std", "median", "iqr", "cv"):
            res[f"{f}_{k_}"] = st[k_]
        if f in MORPH_FEATURES:
            norms.append(he_["entropy_norm"])
    if full:
        res["nuclear_entropy_index_exploratory"] = float(np.mean(norms)) if norms else float("nan")
        res["circularity_frac_gt1"] = float((nuc["circularity"] > 1).mean()) if len(nuc) else float("nan")
    return res


def compute_morph_entropies(nuclei: pd.DataFrame, cfg: Config) -> tuple[dict, pd.DataFrame]:
    """2. AŞAMA. Seçim sırası: (1) odaklı çekirdekler (focus_exclude), (2) QuPath sınıfı varsa
    nucleus_classes_include. Ayrıca iğsi (endotel/stroma olasılığı yüksek) çekirdekler hariç DUYARLILIK
    entropileri (*_entropy_nospindle) — heuristik, doğrulanmış sınıflandırma değildir.
    Döner: (sonuç sözlüğü, tile düzeyi entropi tablosu)."""
    res: dict[str, Any] = {}
    nuc = nuclei
    res["n_nuclei_detected"] = int(len(nuclei))
    if len(nuc) and "in_focus" in nuc and cfg.focus_exclude:
        nuc = nuc[nuc["in_focus"].astype(bool)]
    res["n_nuclei_excluded_blur"] = int(len(nuclei) - len(nuc))
    if cfg.nucleus_classes_include and "nucleus_class" in nuc:
        before = len(nuc)
        nuc = nuc[nuc["nucleus_class"].isin(cfg.nucleus_classes_include)]
        res["n_nuclei_excluded_class"] = int(before - len(nuc))
    if len(nuc):
        nuc = nuc.assign(log2_area=np.log2(np.clip(nuc["area_um2"].astype(float), 1e-6, None)))
        spindle = (nuc["eccentricity"] >= cfg.spindle_min_eccentricity) & (nuc["minor_axis_um"] <= cfg.spindle_max_minor_axis_um) \
            if "minor_axis_um" in nuc else pd.Series(False, index=nuc.index)
        res["spindle_fraction"] = float(spindle.mean())
    else:
        spindle = pd.Series(dtype=bool)
        res["spindle_fraction"] = float("nan")
    res["n_nuclei"] = int(len(nuc))
    if "nucleus_class" in nuc and len(nuc):
        for c, v in nuc["nucleus_class"].value_counts(normalize=True).items():
            res[f"class_frac_{re.sub(r'[^A-Za-z0-9]+', '_', str(c))}"] = float(v)

    # Tile düzeyi alan entropisi → uzamsal heterojenlik ve "en heterojen bölge" (hotspot)
    lo_a, hi_a, nb_a = cfg.feature_bins["area_um2"]
    if len(nuc) and "tile_id" in nuc:
        g = nuc.groupby("tile_id")["area_um2"]
        cnt = g.size()
        ent = g.apply(lambda v: histogram_entropy(v, lo_a, hi_a, int(nb_a), cfg.log_base)["entropy"])
        tile_ent = pd.DataFrame({"tile_id": cnt.index, "n_nuclei": cnt.values,
                                 "nuc_area_entropy": np.where(cnt.values >= cfg.min_nuclei_tile, ent.values, np.nan)})
    else:
        tile_ent = pd.DataFrame(columns=["tile_id", "n_nuclei", "nuc_area_entropy"])
    te = tile_ent["nuc_area_entropy"].astype(float)
    s_ = summary_stats(te)
    for st in ("mean", "std", "median", "min", "max"):
        res[f"patch_nuc_area_entropy_{st}"] = s_[st]
    res["patch_nuc_area_entropy_n_tiles"] = s_["n"]
    p = cfg.hotspot_percentile
    res[f"patch_nuc_area_entropy_p{int(p)}"] = float(np.nanpercentile(te, p)) if te.notna().sum() >= 5 else float("nan")
    res[f"patch_nuc_area_entropy_p{int(100 - p)}"] = float(np.nanpercentile(te, 100 - p)) if te.notna().sum() >= 5 else float("nan")

    if res["n_nuclei"] < cfg.min_nuclei_case:
        for f in MORPH_FEATURES + EXTRA_FEATURES:
            res[f"{f}_entropy"] = float("nan")
        res["nuclear_entropy_index_exploratory"] = float("nan")
        res["_insufficient"] = True
        return res, tile_ent
    res.update(_entropy_block(nuc, cfg))
    ns = nuc[~spindle.reindex(nuc.index, fill_value=False)]
    if len(ns) >= cfg.min_nuclei_case:
        res.update(_entropy_block(ns, cfg, suffix="_nospindle", full=False))
    res["_insufficient"] = False
    return res, tile_ent


def recompute_entropy(case_dir: Path, cfg: Config) -> dict | None:
    """Segmentasyon yapılmış bir vakada yalnızca entropi parametreleri değiştiyse: kayıtlı
    nuclei_features.csv.gz → yeni entropiler → case_result.json ve genel bakış görseli güncellenir."""
    case_dir = Path(case_dir)
    rp, npth = case_dir / "case_result.json", case_dir / "nuclei_features.csv.gz"
    if not rp.exists() or not npth.exists():
        return None
    res = json.loads(rp.read_text(encoding="utf-8"))
    nuclei = pd.read_csv(npth)
    old_keys = [k for k in res if any(k.startswith(f) for f in MORPH_FEATURES + EXTRA_FEATURES)
                or k.startswith(("patch_nuc_area", "class_frac_", "n_nuclei", "spindle_fraction",
                                 "nuclear_entropy_index"))]
    for k in old_keys:
        res.pop(k)
    ent, tile_ent = compute_morph_entropies(nuclei, cfg)
    insufficient = ent.pop("_insufficient")
    res.update(ent)
    flags = [f for f in (res.get("qc_flags") or "").split(";") if f and f != "INSUFFICIENT_NUCLEI"]
    if insufficient:
        flags.append("INSUFFICIENT_NUCLEI")
        res["analysis_status"], res["quality_control_status"] = "failed_qc", "FAIL"
        res["error_message"] = f"Çekirdek sayısı {res['n_nuclei']} < {cfg.min_nuclei_case}"
    elif res.get("analysis_status") == "failed_qc" and res.get("error_message", "").startswith("Çekirdek sayısı"):
        res["analysis_status"], res["error_message"] = "success_with_warnings", ""
        res["quality_control_status"] = "WARN" if flags else "PASS"
    res["qc_flags"] = ";".join(dict.fromkeys(flags))
    area = res.get("tissue_area_analyzed_mm2")
    res["nuclei_per_mm2"] = res["n_nuclei"] / area if area else None
    res["entropy_hash"] = cfg.entropy_hash()
    res["config_hash"] = f"{res.get('seg_hash', '')}-{res['entropy_hash']}"
    res["entropy_recomputed_at"] = datetime.now().isoformat(timespec="seconds")
    tm = case_dir / "tile_metrics.csv"
    if tm.exists():
        tiles = pd.read_csv(tm).drop(columns=["nuc_area_entropy", "n_nuclei"], errors="ignore").merge(tile_ent, on="tile_id", how="left")
        tiles.to_csv(tm, index=False)
        try:
            make_overview_figure(case_dir, res, tiles, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("Genel bakış görseli güncellenemedi: %s", exc)
    rp.write_text(json.dumps(_json_clean(res), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return res


def _json_clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            v = None  # JSON'da NaN yerine null
        out[k] = v
    return out


# =============================================================================
# 8. QC GÖRSELLERİ
# =============================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _font(size: int):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def pixel_montage(rows: list, path: Path, title: str | None = None, scale: int = 1, gap: int = 10,
                  cap_px: int = 30, font_size: int = 16) -> Path:
    """Görüntüleri PİKSEL KAYBI OLMADAN yan yana dizer (matplotlib yeniden örneklemesi yok).
    rows = [[(görüntü, başlık, çerçeve_rengi|None), ...], ...]. scale = tam sayı büyütme (en yakın komşu).
    Böylece kaydedilen dosyadaki her piksel slayttaki bir piksele (veya scale×scale bloğa) karşılık gelir."""
    from PIL import Image, ImageDraw

    def prep(im):
        im = np.asarray(im)
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        im = im[..., :3].astype(np.uint8)
        if scale > 1:
            im = np.kron(im, np.ones((scale, scale, 1), np.uint8))
        return im

    rows = [[(prep(im), cap, col) for im, cap, col in r] for r in rows]
    ncol = max(len(r) for r in rows)
    colw = [max((r[c][0].shape[1] for r in rows if c < len(r)), default=0) for c in range(ncol)]
    rowh = [max(it[0].shape[0] for it in r) for r in rows]
    top = 44 if title else 0
    W = sum(colw) + gap * (ncol + 1)
    H = top + sum(h + cap_px for h in rowh) + gap * (len(rows) + 1)
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(canvas)
    if title:
        d.text((gap, 10), title, fill=(11, 11, 11), font=_font(font_size + 4))
    y = top + gap
    for r, h in zip(rows, rowh):
        x = gap
        for c, (im, cap, col) in enumerate(r):
            d.text((x, y + 4), cap, fill=(82, 81, 78) if not col else col, font=_font(font_size))
            canvas.paste(Image.fromarray(im), (x, y + cap_px))
            if col:
                d.rectangle([x - 3, y + cap_px - 3, x + im.shape[1] + 2, y + cap_px + im.shape[0] + 2], outline=col, width=4)
            x += colw[c] + gap
        y += h + cap_px + gap
    canvas.save(path, optimize=True)
    return Path(path)


def make_overview_figure(case_dir: Path, res: dict, tiles: pd.DataFrame, cfg: Config) -> Path:
    """VAKA GENEL BAKIŞ GÖRSELİ (her vaka için otomatik).
    Panel 1: küçük resim | 2: doku maskesi + analiz edilen tile'lar | 3: odak haritası (bulanık blok oranı)
    | 4: tile düzeyi nükleer alan entropisi. Renk ölçekleri tüm vakalarda SABİT → vakalar karşılaştırılabilir.
    Slayt yeniden okunmaz: case_dir/thumbnail.png ve tissue_mask.png kullanılır.
    Kopyası output_dir/qc_gallery/ içine grup_grade_slayt adıyla yazılır."""
    from PIL import Image
    from matplotlib.patches import Rectangle
    plt = _plt()
    case_dir = Path(case_dir)
    thumb = np.asarray(Image.open(case_dir / "thumbnail.png").convert("RGB"))
    mask = np.asarray(Image.open(case_dir / "tissue_mask.png")) > 0
    sx = res["image_width"] / thumb.shape[1]
    sy = res["image_height"] / thumb.shape[0]
    sel = tiles[tiles["selected"].astype(bool)] if "selected" in tiles else tiles.iloc[0:0]
    gh, gw = int(tiles["grid_y"].max()) + 1, int(tiles["grid_x"].max()) + 1

    fig, ax = plt.subplots(1, 4, figsize=(24, 6.6), gridspec_kw={"wspace": 0.04})
    ax[0].imshow(thumb, interpolation="none")
    ax[0].set_title("Küçük resim", fontsize=11)
    ov = thumb.copy()
    ov[mask] = (0.6 * ov[mask] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
    ax[1].imshow(ov, interpolation="none")
    for _, t in sel.iterrows():
        ax[1].add_patch(Rectangle((t.x_l0 / sx, t.y_l0 / sy), t.tile_l0 / sx, t.tile_l0 / sy, fill=False, ec="blue", lw=0.35))
    ax[1].set_title(f"Doku maskesi (yeşil) + analiz edilen tile'lar (mavi, n={len(sel)})", fontsize=11)

    def grid(col):
        hm = np.full((gh, gw), np.nan)
        if col in sel:
            for _, t in sel.iterrows():
                hm[int(t.grid_y), int(t.grid_x)] = t[col]
        return hm

    if {"focus_blocks_blurry", "focus_blocks_determinate"} <= set(sel.columns):
        sel = sel.assign(blur_frac=sel["focus_blocks_blurry"] / sel["focus_blocks_determinate"].replace(0, np.nan))
    im2 = ax[2].imshow(grid("blur_frac"), cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02, label="bulanık blok oranı")
    bf = res.get("focus_blurry_tissue_fraction")
    ax[2].set_title(f"Odak haritası — bulanık doku %{100 * bf:.0f} (analizden çıkarıldı)" if bf is not None else "Odak haritası", fontsize=11)
    lo, hi = cfg.heatmap_entropy_range
    im3 = ax[3].imshow(grid("nuc_area_entropy"), cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.02, label="bit (sabit ölçek)")
    ax[3].set_title(f"Tile düzeyi nükleer alan entropisi (≥{cfg.min_nuclei_tile} odaklı çekirdek)", fontsize=11)
    for a in ax:
        a.axis("off")

    def f(k, d=2):
        v = res.get(k)
        return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"

    head = (f"{res.get('case_id')}   |   {res.get('slide_id')}   |   Grade {res.get('grade') or '?'} "
            f"({res.get('grade_group') or '?'})   |   {res.get('slide_kind') or ''}   |   "
            f"QC: {res.get('quality_control_status')} {('[' + res['qc_flags'] + ']') if res.get('qc_flags') else ''}")
    sub = (f"Odaklı çekirdek n={res.get('n_nuclei', '—')} ({res.get('segmenter', '')})   ·   Entropi (bit): alan {f('area_um2_entropy')}  "
           f"çevre {f('perimeter_um_entropy')}  dairesellik {f('circularity_entropy')}  "
           f"eksantrisite {f('eccentricity_entropy')}   ·   medyan alan {f('area_um2_median', 1)} µm²   ·   "
           f"{f('effective_mpp', 3)} µm/px   ·   boya: {res.get('stain_class', '?')}")
    fig.suptitle(head, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.905, sub, ha="center", fontsize=11, color="#333")
    fig.subplots_adjust(left=0.01, right=0.955, top=0.86, bottom=0.02)
    out = case_dir / "qc_overview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    gal = case_dir.parent.parent / "qc_gallery"
    gal.mkdir(exist_ok=True)
    name = f"{res.get('grade_group') or 'NA'}_{res.get('grade') or 'NA'}_{res.get('slide_id')}.png"
    (gal / name).write_bytes(out.read_bytes())
    return out


def make_segmentation_figure(case_dir: Path, debug_tiles: dict, cfg: Config):
    """Örnek tile'lar, GERÇEK 1:1 piksel: orijinal | Macenko normalize | H kanalı | segmentasyon (yeşil kontur)
    + bulanık bloklar (kırmızı, dışlanır). Matplotlib kullanılmaz → küçültme/yumuşatma yok."""
    from skimage.segmentation import find_boundaries
    if not debug_tiles:
        return
    B = cfg.focus_block_px
    vmax = cfg.conc_hist_range[1]
    rows = []
    for tid, dbg in debug_tiles.items():
        hgray = (255 * (1 - np.clip(dbg["h"] / vmax, 0, 1))).astype(np.uint8)
        ov = dbg["rgb"].astype(np.float32).copy()
        for (by, bx), bl in np.ndenumerate(dbg["blurry"]):
            if bl:
                sl = (slice(by * B, (by + 1) * B), slice(bx * B, (bx + 1) * B))
                ov[sl] = 0.7 * ov[sl] + 0.3 * np.array([255, 0, 0])
                ov[sl][[0, -1], :] = ov[sl][:, [0, -1]] = (220, 0, 0)
        ov = ov.astype(np.uint8)
        ov[find_boundaries(dbg["labels"], mode="inner")] = (0, 230, 0)
        n_obj = len(np.unique(dbg["labels"])) - 1
        rows.append([(dbg["rgb"], f"{tid} — orijinal (1:1)", None), (dbg["norm"], "Macenko normalize", None),
                     (hgray, "Hematoksilen (normalize)", None),
                     (ov, f"segmentasyon: {n_obj} nesne; kırmızı = bulanık (dışlandı)", None)])
    pixel_montage(rows, case_dir / "qc_segmentation_examples.png",
                  title="Segmentasyon örnekleri — her piksel slayttaki bir piksel (küçültme/yumuşatma yok)")


def make_feature_hist_figure(case_dir: Path, nuclei: pd.DataFrame, cfg: Config):
    plt = _plt()
    if not len(nuclei):
        return
    nuc = nuclei[nuclei["in_focus"].astype(bool)] if cfg.focus_exclude and "in_focus" in nuclei else nuclei
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for a, f in zip(ax, MORPH_FEATURES):
        lo, hi, nb = cfg.feature_bins[f]
        a.hist(np.clip(nuc[f], lo, hi), bins=np.linspace(lo, hi, int(nb) + 1), color="#4a6fa5", ec="white", lw=0.3)
        a.set_title(f"{f} (n={len(nuc)}, {int(nb)} sabit bin)")
        a.set_xlabel(f)
    fig.tight_layout()
    fig.savefig(case_dir / "qc_feature_histograms.png", dpi=110)
    plt.close(fig)


# =============================================================================
# 9. TEK VAKA ANALİZİ
# =============================================================================
def analyze_case(case_id: str, image_path: str | Path, output_dir: str | Path,
                 meta: dict | None = None, cfg: Config | None = None,
                 roi_geojson: str | Path | None = None, mpp_override: float | None = None,
                 expected_md5: str | None = None, write_tables: bool = True) -> dict:
    """
    GİRİŞ
      case_id      : TCGA hasta ID'si (ör. TCGA-BP-4165)
      image_path   : WSI yolu (.svs vb.) veya küçük görüntü (PNG/JPEG/TIFF; --mpp gerekir)
      output_dir   : çıktı kök klasörü
      meta         : Excel'den gelen ek bilgiler (slide_uuid, file_id, grade ...)
      cfg          : Config (None → varsayılan)
      roi_geojson  : İsteğe bağlı patolog ROI'si (QuPath GeoJSON, seviye-0 koordinatları)
    ÇIKTI (output_dir/cases/<slide_id>/)
      case_result.json, tile_metrics.csv, nuclei_features.csv.gz, case_log.txt, qc_*.png
    DÖNÜŞ
      Vaka düzeyi sonuç sözlüğü (case_level_results tablosunun bir satırı).
    Hata durumunda istisna fırlatmaz; analysis_status ve error_message doldurulur.
    """
    cfg = cfg or Config()
    meta = dict(meta or {})
    output_dir = Path(output_dir)
    image_path = Path(image_path) if image_path else None
    barcode = meta.get("slide_barcode") or (image_path.name.split(".")[0] if image_path else None)
    bc = parse_tcga_barcode(barcode)
    slide_id = barcode if bc["barcode_valid"] else (meta.get("slide_uuid") or case_id)
    case_dir = output_dir / "cases" / slide_id
    case_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(case_dir / "case_log.txt", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    t0 = time.time()
    warnings: list[str] = list(meta.pop("id_warnings", []) or [])
    res: dict[str, Any] = {
        "case_id": case_id, "sample_id": bc["sample_id"], "slide_id": slide_id, "tss_code": bc["tss_code"],
        "slide_uuid": meta.get("slide_uuid"), "file_id": meta.get("file_id"),
        "slide_kind": bc["slide_kind"], "grade": meta.get("grade"), "grade_group": meta.get("grade_group"),
        "excel_row": meta.get("excel_row"), "image_path": str(image_path) if image_path else None,
        "config_hash": cfg.config_hash(), "seg_hash": cfg.seg_hash(), "entropy_hash": cfg.entropy_hash(),
        "pipeline_version": PIPELINE_VERSION, "segmenter": cfg.segmenter, "target_mpp": cfg.target_mpp,
        "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        "histogram_bins": cfg.pixel_hist_bins, "log_base": cfg.log_base,
        "analysis_status": "failed_error", "error_message": "", "quality_control_status": "FAIL", "qc_flags": "",
    }
    qc_flags: list[str] = []
    src = None
    try:
        log.info("=== Vaka %s | slayt %s | %s", case_id, slide_id, image_path)
        if cfg.tile_size_px % cfg.focus_block_px:
            raise ValueError("tile_size_px, focus_block_px'in tam katı olmalı")
        if bc["barcode_valid"] and bc["case_id"] != case_id:
            raise ValueError(f"ID KARIŞIKLIĞI: dosya barcode'u {bc['case_id']} ≠ case_id {case_id}")
        if image_path is None or not image_path.exists():
            res.update(analysis_status="missing_file", error_message=f"Dosya yok: {image_path}")
            return res
        res["file_size_bytes"] = image_path.stat().st_size
        res["file_md5"] = md5sum(image_path)
        if expected_md5:
            res["md5_matches_gdc"] = res["file_md5"] == expected_md5.lower()
            if not res["md5_matches_gdc"]:
                raise IOError(f"MD5 uyuşmuyor (indirme bozuk olabilir): {res['file_md5']} ≠ {expected_md5}")

        src = open_image(image_path, mpp=mpp_override)
        res.update(image_width=src.width, image_height=src.height, image_format=src.fmt,
                   objective_power=getattr(src, "objective_power", None),
                   n_levels=len(src.level_dimensions), image_resolution_or_mpp=src.mpp,
                   scanner_date=src.properties.get("aperio.Date"),
                   scanner_id=src.properties.get("aperio.ScanScope ID"))
        if getattr(src, "mpp_note", ""):
            warnings.append(src.mpp_note)
        if src.mpp is None:
            res.update(analysis_status="failed_qc", quality_control_status="FAIL",
                       error_message="MPP_MISSING: µm/piksel bilgisi yok; fiziksel morfometri yapılamaz (tahmin edilmez).")
            qc_flags.append("MPP_MISSING")
            return res

        # --- Doku maskesi ---
        tw, th = max(src.width // cfg.mask_downsample, 1), max(src.height // cfg.mask_downsample, 1)
        thumb = src.thumbnail(tw, th)
        sx, sy = src.width / tw, src.height / th
        mask, minfo = compute_tissue_mask(thumb, src.mpp * (sx + sy) / 2, cfg)
        res["tissue_fraction"] = minfo.get("tissue_fraction", 0.0)
        res["tissue_area_mm2"] = float(mask.sum() * sx * sy * src.mpp ** 2 / 1e6)
        res["mask_sat_threshold"] = minfo.get("sat_threshold")
        res["mask_pen_fraction"] = minfo.get("pen_fraction")
        if roi_geojson:
            roi = load_roi_mask(roi_geojson, mask.shape, sx, sy)
            mask &= roi
            res["roi_used"] = str(roi_geojson)
            res["analysis_region"] = "patolog ROI ∩ doku maskesi"
        else:
            res["roi_used"] = None
            res["analysis_region"] = "tüm doku maskesi (tümör/normal/nekroz AYRIŞTIRILMADI)"
        from PIL import Image
        Image.fromarray(thumb).save(case_dir / "thumbnail.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(case_dir / "tissue_mask.png")
        if res["tissue_fraction"] < cfg.qc_min_tissue_fraction or not mask.any():
            qc_flags.append("NO_OR_LOW_TISSUE")
            res.update(analysis_status="failed_qc", error_message=f"Doku oranı çok düşük ({res['tissue_fraction']:.4f})")
            return res

        # --- H&E / boya taraması ---
        io = estimate_background_io(thumb, mask, cfg)
        scr = screen_stain(thumb, mask, io, cfg)
        res.update({k: v for k, v in scr.items() if k != "stain_class_tr"})
        if scr["stain_class"] in STAIN_FAIL_CLASSES:
            qc_flags.append(f"STAIN_{scr['stain_class']}")
            res.update(analysis_status="failed_qc",
                       error_message=f"Boya taraması: {STAIN_CLASSES[scr['stain_class']]} → slaytı değiştirin (aynı hastanın başka H&E slaytı / yedek vaka)")
            return res
        if scr["stain_class"] != "HE_TYPICAL":
            qc_flags.append(f"STAIN_{scr['stain_class']}")
            warnings.append(STAIN_CLASSES[scr["stain_class"]])

        # --- Tile planı ---
        tiles, tinfo = plan_tiles(src, mask, sx, sy, cfg)
        res.update(effective_mpp=tinfo["effective_mpp"], resample_scale=tinfo["scale"],
                   tile_size_um=cfg.tile_size_px * tinfo["effective_mpp"], n_tiles_grid=tinfo["n_grid"],
                   n_tiles_eligible=tinfo["n_eligible"], n_tiles_analyzed=tinfo.get("n_selected", 0))
        if not tinfo["resolution_ok"]:
            qc_flags.append("RESOLUTION_INSUFFICIENT")
            res.update(analysis_status="failed_qc",
                       error_message=f"Slayt {src.mpp:.3f} µm/px; hedef {cfg.target_mpp} µm/px'ten kaba. Büyütme yapılmaz "
                                     "(yapay çözünürlük). target_mpp'yi kohortun en kaba slaytına göre seçin.")
            return res
        if tinfo["scale"] != 1.0:
            warnings.append(f"Slayt {src.mpp:.3f} µm/px; {tinfo['effective_mpp']:.3f} µm/px'e küçültüldü")
        if tinfo.get("n_selected", 0) < cfg.qc_min_tiles:
            qc_flags.append("TOO_FEW_TILES")
            res.update(analysis_status="failed_qc", error_message=f"Uygun tile sayısı {tinfo.get('n_selected', 0)} < {cfg.qc_min_tiles}")
            return res

        # --- Boya modeli ---
        stain = fit_stain_model(src, tiles, mask, sx, sy, tinfo, io, cfg)
        res.update(stain_method_used=stain["stain_method_used"], stain_matrix=json.dumps(np.round(stain["stain_matrix"], 4).tolist()),
                   stain_cos_h_to_ref=stain.get("cos_h_to_ref"), stain_cos_e_to_ref=stain.get("cos_e_to_ref"),
                   background_io=json.dumps(np.round(io, 1).tolist()), nuclear_threshold_h=stain["nuclear_threshold"],
                   stain_conc_scale=json.dumps(np.round(stain["conc_scale"], 5).tolist()), stain_fit_pixels=stain["stain_fit_pixels"],
                   stardist_norm_lo=json.dumps(np.round(stain["sd_lo"], 2).tolist()),
                   stardist_norm_hi=json.dumps(np.round(stain["sd_hi"], 2).tolist()),
                   **{f"raw_mean_od_{c}": v for c, v in zip("rgb", stain["raw_mean_od"])},
                   **{f"norm_mean_od_{c}": v for c, v in zip("rgb", stain["norm_mean_od"])})
        res["preprocessing_method"] = (
            f"OpenSlide seviye-0 tile {cfg.tile_size_px}px @ {tinfo['effective_mpp']:.3f} µm/px; "
            f"doku maskesi HSV-S Otsu (≥{cfg.mask_min_saturation}) + kapama; "
            f"{stain['stain_method_used']} renk ayrıştırma + konsantrasyon normalizasyonu (99. persentil → referans); "
            + ("çekirdek: slayt-düzeyi Otsu(H) + watershed; " if cfg.segmenter == "classical" else
               f"çekirdek: StarDist {cfg.stardist_model} (×{tinfo['effective_mpp'] / cfg.stardist_model_mpp:.2f} ölçekli); ")
            + f"odak: {cfg.focus_block_px}px blok bant-oranı ≥{cfg.focus_min_band_ratio}"
            + (" (bulanık bloklar dışlandı)" if cfg.focus_exclude else ""))
        if stain["stain_warning"]:
            warnings.append(stain["stain_warning"]); qc_flags.append("STAIN_ATYPICAL")

        # --- Tile döngüsü ---
        ctx = {"io": io, "pinv": stain["pinv"], "scale": np.asarray(stain["conc_scale"]),
               "thr": stain["nuclear_threshold"], "mpp": tinfo["effective_mpp"],
               "sd_lo": stain["sd_lo"], "sd_hi": stain["sd_hi"]}
        if cfg.segmenter == "stardist":
            _get_stardist(cfg.stardist_model)  # kurulum/model hatası tile döngüsünden önce net mesajla çıksın
        sel = tiles[tiles["selected"]].copy()
        n_dbg = min(cfg.qc_example_tiles, len(sel))
        dbg_ids = set(sel["tile_id"].iloc[np.linspace(0, len(sel) - 1, n_dbg).astype(int)]) if n_dbg else set()
        agg = {k: np.zeros(cfg.pixel_hist_bins if k in ("hematoxylin", "eosin") else 256, np.int64)
               for k in ("gray", "red", "green", "blue", "hematoxylin", "eosin")}
        nuclei_parts, tile_rows, debug_tiles, block_rows = [], [], {}, []
        B = cfg.focus_block_px
        tile_errors = 0
        for k, (_, t) in enumerate(sel.iterrows(), 1):
            try:
                rgb = read_tile(src, t.x_l0, t.y_l0, t.tile_l0, cfg.tile_size_px)
                tm = tile_tissue_mask(mask, t.x_l0, t.y_l0, t.tile_l0, cfg.tile_size_px, sx, sy)
                out = process_tile(rgb, tm, ctx, cfg, keep_debug=t.tile_id in dbg_ids)
            except Exception as exc:  # noqa: BLE001
                tile_errors += 1
                log.warning("Tile %s hatası: %s", t.tile_id, exc)
                tile_rows.append({"tile_id": t.tile_id, "tile_error": str(exc)})
                continue
            for key in agg:
                agg[key] += out["hist"][key]
            nuc = out["nuclei"]
            if len(nuc):
                f = t.tile_l0 / cfg.tile_size_px
                nuc = nuc.assign(tile_id=t.tile_id,
                                 x_l0=t.x_l0 + nuc["centroid_x_px"] * f, y_l0=t.y_l0 + nuc["centroid_y_px"] * f)
                nuclei_parts.append(nuc.drop(columns=["centroid_x_px", "centroid_y_px"]))
            tile_rows.append({"tile_id": t.tile_id, **out["metrics"]})
            bl = out["blocks"]
            fb = t.tile_l0 / cfg.tile_size_px * B   # blok kenarı (seviye-0 piksel)
            for (by, bx), r_ in np.ndenumerate(bl["ratio"]):
                block_rows.append((t.tile_id, t.x_l0 + bx * fb, t.y_l0 + by * fb, fb, float(r_),
                                   bool(bl["determinate"][by, bx]), bool(bl["blurry"][by, bx])))
            if "debug" in out:
                debug_tiles[t.tile_id] = out["debug"]
            if k % 100 == 0 or k == len(sel):
                log.info("  tile %d/%d", k, len(sel))
        if sel.shape[0] and tile_errors == len(sel):
            raise RuntimeError("Hiçbir tile işlenemedi (case_log.txt'e bakın)")
        pd.DataFrame(block_rows, columns=["tile_id", "x_l0", "y_l0", "size_l0", "focus_ratio", "determinate", "blurry"]
                     ).to_csv(case_dir / "focus_blocks.csv.gz", index=False, compression="gzip")
        if tile_errors:
            warnings.append(f"{tile_errors} tile okunamadı/işlenemedi")
            if tile_errors > 0.05 * len(sel):
                qc_flags.append("MANY_TILE_ERRORS")
        tm_df = pd.DataFrame(tile_rows)
        nuclei = pd.concat(nuclei_parts, ignore_index=True) if nuclei_parts else pd.DataFrame(
            columns=[*MORPH_FEATURES, "tile_id", "in_focus", "focus_ratio"])
        if cfg.save_nuclei_table and len(nuclei):
            nuclei.to_csv(case_dir / "nuclei_features.csv.gz", index=False, compression="gzip")

        # --- Odak (bulanıklık) özeti ---
        n_det = int(tm_df.get("focus_blocks_determinate", pd.Series([0])).sum())
        n_blur = int(tm_df.get("focus_blocks_blurry", pd.Series([0])).sum())
        res["focus_blocks_determinate"] = n_det
        res["focus_blurry_tissue_fraction"] = n_blur / n_det if n_det else float("nan")
        res["focus_ratio_median"] = float(tm_df["focus_ratio_median"].median()) if "focus_ratio_median" in tm_df else float("nan")
        bf = res["focus_blurry_tissue_fraction"]
        if np.isfinite(bf) and bf > cfg.focus_fail_fraction:
            qc_flags.append("OUT_OF_FOCUS")
            res.update(analysis_status="failed_qc",
                       error_message=f"Dokunun %{100 * bf:.0f}'i odak dışı (> %{100 * cfg.focus_fail_fraction:.0f})")
        elif np.isfinite(bf) and bf > cfg.focus_warn_fraction:
            qc_flags.append("MOSTLY_OUT_OF_FOCUS")
        if np.isfinite(bf) and bf > 0:
            warnings.append(f"Bulanık doku %{100 * bf:.0f} — bu bloklardaki çekirdek/pikseller analiz dışı"
                            if cfg.focus_exclude else f"Bulanık doku %{100 * bf:.0f} (dışlanmadı!)")

        # --- Piksel entropileri (havuzlanmış histogram, yalnız ODAKLI doku pikselleri) ---
        res["tissue_pixel_count"] = int(agg["gray"].sum())
        res["tissue_area_analyzed_mm2"] = res["tissue_pixel_count"] * tinfo["effective_mpp"] ** 2 / 1e6
        for key, col in (("gray", "grayscale_entropy"), ("red", "red_channel_entropy"), ("green", "green_channel_entropy"),
                         ("blue", "blue_channel_entropy"), ("hematoxylin", "hematoxylin_entropy"), ("eosin", "eosin_entropy")):
            res[col] = shannon_entropy_from_counts(agg[key], cfg.log_base)
        (case_dir / "pixel_histograms.json").write_text(json.dumps({k: v.tolist() for k, v in agg.items()}), encoding="utf-8")
        res["he_entropy_range"] = json.dumps(list(cfg.conc_hist_range))
        n_pix = max(res["tissue_pixel_count"], 1)
        res["h_fraction_clipped_high"] = float(tm_df.get("h_clip_high", pd.Series([0])).sum() / n_pix)
        res["e_fraction_clipped_high"] = float(tm_df.get("e_clip_high", pd.Series([0])).sum() / n_pix)
        s_ = summary_stats(tm_df["gray_entropy"]) if "gray_entropy" in tm_df else summary_stats([])
        for st in ("mean", "std", "median", "min", "max"):
            res[f"patch_entropy_{st}"] = s_[st]
        res["patch_entropy_n_tiles"] = s_["n"]
        lap = tm_df.get("laplacian_var", pd.Series(dtype=float))
        res["tile_laplacian_var_median"] = float(lap.median()) if len(lap) else float("nan")
        res["tile_dark_fraction_median"] = float(tm_df.get("dark_fraction", pd.Series(dtype=float)).median())
        for c in [c for c in tm_df.columns if c.startswith("excl_")]:
            res[f"n_{c}"] = int(tm_df[c].sum())

        # --- ANA ÖLÇÜT: morfometrik entropiler (2. aşama fonksiyonu; recompute ile aynı kod) ---
        ent, tile_ent = compute_morph_entropies(nuclei, cfg)
        insufficient = ent.pop("_insufficient")
        res.update(ent)
        res["nuclei_per_mm2"] = res["n_nuclei"] / res["tissue_area_analyzed_mm2"] if res["tissue_area_analyzed_mm2"] else float("nan")
        if insufficient and res["analysis_status"] != "failed_qc":
            qc_flags.append("INSUFFICIENT_NUCLEI")
            res.update(analysis_status="failed_qc",
                       error_message=f"Çekirdek sayısı {res['n_nuclei']} < {cfg.min_nuclei_case}; morfometrik entropi hesaplanmadı")
        tiles = tiles.merge(tm_df, on="tile_id", how="left").merge(tile_ent, on="tile_id", how="left")
        tiles.to_csv(case_dir / "tile_metrics.csv", index=False)

        # --- QC kararı ---
        if res.get("tile_dark_fraction_median", 0) > 0.10:
            qc_flags.append("DARK_ARTIFACT_OR_FOLD")
        if bc["slide_kind"] and bc["slide_kind"].startswith("frozen"):
            warnings.append("Dondurulmuş kesit (TS/BS): donma artefaktı nükleer morfolojiyi etkileyebilir")
        if res["analysis_status"] != "failed_qc":
            res["analysis_status"] = "success_with_warnings" if (warnings or qc_flags) else "success"
        res["quality_control_status"] = "FAIL" if res["analysis_status"] == "failed_qc" else ("WARN" if qc_flags else "PASS")
        res["qc_flags"] = ";".join(dict.fromkeys(qc_flags))

        for fig_fn, args in ((make_overview_figure, (case_dir, res, tiles, cfg)),
                             (make_segmentation_figure, (case_dir, debug_tiles, cfg)),
                             (make_feature_hist_figure, (case_dir, nuclei, cfg))):
            try:
                fig_fn(*args)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{fig_fn.__name__} üretilemedi: {exc}")
                log.exception("QC görsel hatası")
    except Exception as exc:  # noqa: BLE001
        res["analysis_status"] = "failed_error"
        res["quality_control_status"] = "FAIL"
        res["error_message"] = f"{type(exc).__name__}: {exc}"
        log.error("HATA %s: %s\n%s", case_id, exc, traceback.format_exc())
    finally:
        if src is not None:
            src.close()
        res["qc_flags"] = ";".join(dict.fromkeys(qc_flags))
        res["warnings"] = " | ".join(dict.fromkeys(warnings))
        res["runtime_sec"] = round(time.time() - t0, 1)
        res = _json_clean(res)
        (case_dir / "case_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log.info("Bitti: %s → %s (%s s)", case_id, res["analysis_status"], res["runtime_sec"])
        log.removeHandler(fh); fh.close()
        if write_tables:
            write_run_metadata(output_dir, cfg)
            write_aggregate_tables(output_dir)
    return res


# =============================================================================
# 10. TOPLU ANALİZ (200 vaka) – YALNIZCA --confirm ile
# =============================================================================
def prepare_cases(excel: str | Path, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, issues = load_case_table(excel)
    index = build_file_index(data_dir)
    manifests = read_gdc_manifests(data_dir)
    recs, extra = [], []
    for _, row in df.iterrows():
        r = resolve_case_file(row, index)
        barcode = r.get("barcode_file") or row["slide_barcode_excel"]
        status = "ready"
        if row["is_duplicate"]:
            status = "skipped_duplicate"
        elif str(row.get("selected", "Yes")).strip().lower() not in ("yes", "evet"):
            status = "skipped_not_selected"
        elif row.get("manual_excluded"):
            status = "skipped_manual_exclusion"
        elif r["id_error"] == "MISSING_FILE":
            status = "missing_file"
        elif r["id_error"]:
            status = "skipped_id_error"
        for w in r["id_warnings"]:
            extra.append({"excel_row": row["excel_row"], "row_no": row.get("row_no"), "case_id_excel": row["case_id_excel"],
                          "level": "warning", "code": "FILE_ID_CHECK", "message": w})
        if r["id_error"] and r["id_error"] != "MISSING_FILE":
            extra.append({"excel_row": row["excel_row"], "row_no": row.get("row_no"), "case_id_excel": row["case_id_excel"],
                          "level": "error", "code": "FILE_CASE_MISMATCH", "message": r["id_error"]})
        fid = r.get("file_id_dir") or row["file_id_excel"]
        recs.append({"excel_row": row["excel_row"], "row_no": row.get("row_no"), "case_id": row["case_id_excel"],
                     "slide_barcode": barcode, "slide_uuid": r.get("slide_uuid_file") or row["slide_uuid_excel"],
                     "file_id": fid, "grade": row.get("grade"), "grade_group": row.get("grade_group"),
                     "n_slides_available": row.get("n_slides_available"), "image_path": r["image_path"], "match_method": r["match_method"], "prep_status": status,
                     "expected_md5": manifests.get(fid) if fid else None, "id_warnings": r["id_warnings"]})
    issues = pd.concat([issues, pd.DataFrame(extra)], ignore_index=True) if extra else issues
    return pd.DataFrame(recs), issues


def _meta_from_rec(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("slide_barcode", "slide_uuid", "file_id", "grade", "grade_group", "excel_row")} | {
        "id_warnings": list(rec.get("id_warnings") or [])}


def run_single_from_excel(excel, data_dir, output_dir, case: str, cfg: Config, roi_geojson=None) -> dict:
    cases, issues = prepare_cases(excel, data_dir)
    write_validation(output_dir, cases, issues)
    m = cases[(cases["case_id"] == case) | (cases["slide_barcode"] == case)]
    m = m[m["prep_status"] != "skipped_duplicate"]
    if m.empty:
        raise SystemExit(f"Excel'de vaka bulunamadı: {case}")
    rec = m.iloc[0].to_dict()
    if rec["prep_status"] != "ready":
        raise SystemExit(f"{case} analiz edilemez: {rec['prep_status']} (bkz. quality_control_report.xlsx)")
    return analyze_case(rec["case_id"], rec["image_path"], output_dir, meta=_meta_from_rec(rec), cfg=cfg,
                        roi_geojson=roi_geojson, expected_md5=rec.get("expected_md5"))


def _dir_bytes(d: Path) -> int:
    try:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    except OSError:
        return 0


def gdc_download(file_id: str, data_dir: Path, gdc_client: str, retries: int = 2, timeout_s: int = 4 * 3600,
                 n_processes: int = 8, stall_s: int = 600) -> Path | None:
    """Tek bir dosyayı gdc-client ile indirir (data_dir/<file_id>/<dosya>). Başarısızsa None.
    n_processes: paralel bağlantı sayısı (-n). stall_s: klasör boyutu bu kadar saniye hiç değişmezse
    (takılan bağlantı) gdc-client sonlandırılır ve yeniden denenir."""
    import subprocess
    import tempfile
    data_dir.mkdir(parents=True, exist_ok=True)
    fdir = data_dir / file_id
    cmd = [gdc_client, "download", file_id, "-d", str(data_dir), "-n", str(int(n_processes)),
           "--no-related-files", "--no-annotations"]
    for attempt in range(1, retries + 2):
        try:
            errf = tempfile.TemporaryFile(mode="w+")   # PIPE değil: dolarsa gdc-client bloke olur
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf, text=True)
        except FileNotFoundError:
            raise SystemExit(f"gdc-client bulunamadı: {gdc_client!r}. Yolunu proje_ayarlari.json → gdc_client alanına yazın.")
        t0 = last_change = time.time()
        last_size, reason = -1, None
        try:
            while proc.poll() is None:
                time.sleep(5)
                size = _dir_bytes(fdir)
                now = time.time()
                if size != last_size:
                    last_size, last_change = size, now
                elif now - last_change > stall_s:
                    reason = f"{stall_s // 60} dk boyunca ilerleme yok (takıldı)"
                elif now - t0 > timeout_s:
                    reason = "zaman aşımı"
                if reason:
                    proc.kill()
                    proc.wait()
                    break
        except BaseException:
            proc.kill()
            raise
        errf.seek(0)
        err = errf.read()
        errf.close()
        files = [f for f in fdir.glob("*") if f.suffix.lower() in SUPPORTED_WSI_EXT]
        if reason is None and proc.returncode == 0 and files:
            return files[0]
        log.warning("gdc-client deneme %d başarısız (%s): %s", attempt, reason or f"kod {proc.returncode}",
                    (err or "")[-300:])
        if reason and files:   # yarım dosya .svs adıyla kaldıysa sil, temiz başla
            for f in files:
                f.unlink(missing_ok=True)
    return None


def run_batch(excel, data_dir, output_dir, cfg: Config, force: bool = False, limit: int | None = None,
              gdc_client: str | None = None, delete_after: bool = False, min_free_gb: float = 5.0,
              only_case_ids: list[str] | None = None, gdc_processes: int = 8, prefetch: bool = True) -> pd.DataFrame:
    """Tüm vakaları sırayla analiz eder. Hata veren vaka süreci durdurmaz.
    • Devam ettirme: aynı parametrelerle daha önce işlenmiş vaka atlanır (indirilmez bile).
    • gdc_client verilirse: dosyası olmayan vaka İNDİRİLİR → analiz → (delete_after) SİLİNİR.
    • prefetch: bir slayt analiz edilirken SIRADAKİ slayt arka planda indirilir (diskte en çok 2 slayt).
      Yalnız bu çalışmanın indirdiği dosyalar silinir."""
    import shutil
    import threading
    from concurrent.futures import ThreadPoolExecutor
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cases, issues = prepare_cases(excel, data_dir)
    write_validation(output_dir, cases, issues)
    write_run_metadata(output_dir, cfg)
    todo = cases if limit is None else cases.head(limit)
    if only_case_ids:
        todo = todo[todo["case_id"].isin(only_case_ids)]
    counts: dict[str, int] = {}
    seen_cases: set[str] = set()
    n = len(todo)

    def bump(k):
        counts[k] = counts.get(k, 0) + 1

    # ---- 1) Plan: indirme gerektirmeyen kararlar (stub / devam ettirme) ----
    jobs: list[dict] = []
    for i, rec in enumerate(todo.to_dict("records"), 1):
        tag = f"[{i}/{n}] {rec['case_id']}"
        status = rec["prep_status"]
        if status in ("ready", "missing_file") and rec["case_id"] in seen_cases:
            status = "skipped_duplicate"
        downloadable = bool(status == "missing_file" and gdc_client and rec.get("file_id")
                            and UUID_RE.match(str(rec["file_id"])))
        if status != "ready" and not downloadable:
            _write_stub(output_dir, rec, status)
            bump(status)
            log.info("%s → %s", tag, status)
            continue
        seen_cases.add(rec["case_id"])
        prev = output_dir / "cases" / (rec["slide_barcode"] or rec["case_id"]) / "case_result.json"
        if prev.exists() and not force:
            try:
                p = json.loads(prev.read_text(encoding="utf-8"))
                if str(p.get("segmenter", "")).startswith("qupath"):
                    _refresh_metadata(prev, rec)
                    if p.get("entropy_hash") != cfg.entropy_hash():
                        recompute_entropy(prev.parent, cfg)
                    bump("qupath_import_kept")
                    log.info("%s → QuPath'ten içe aktarılmış çekirdekler korunuyor (--force ile yeniden segmente edilir)", tag)
                    continue
                same_seg = p.get("seg_hash") == cfg.seg_hash()
                segmented = same_seg and (p.get("analysis_status") in SUCCESS_STATUSES
                                          or "INSUFFICIENT_NUCLEI" in (p.get("qc_flags") or ""))
                if segmented:
                    _refresh_metadata(prev, rec)  # Excel'deki grade vb. güncellendiyse
                    if p.get("entropy_hash") == cfg.entropy_hash():
                        bump("resumed_skip")
                        log.info("%s → daha önce başarılı, parametreler aynı (atlandı)", tag)
                    else:
                        r = recompute_entropy(prev.parent, cfg)
                        bump("entropy_recomputed")
                        log.info("%s → yalnız entropi yeniden hesaplandı → %s", tag, r.get("analysis_status") if r else "?")
                    continue
                if same_seg and p.get("analysis_status") == "failed_qc":
                    _refresh_metadata(prev, rec)
                    bump("previous_qc_fail_skip")
                    log.info("%s → önceki çalışmada QC'den geçemedi (%s); atlandı (--force ile yeniden)", tag, p.get("qc_flags"))
                    continue
            except Exception as exc:  # noqa: BLE001
                log.warning("%s önceki sonuç okunamadı (%s); yeniden analiz edilecek", tag, exc)
        jobs.append({"rec": rec, "tag": tag, "downloadable": downloadable})
    if jobs:
        log.info("Analiz edilecek %d slayt (%d tanesi indirilecek)%s", len(jobs), sum(j["downloadable"] for j in jobs),
                 "; indirme ile analiz ÜST ÜSTE yürütülür" if prefetch and gdc_client else "")

    # ---- 2) İndirme (arka planda, bir slayt önden) ----
    pending_delete: set[Path] = set()   # bu çalışmanın indirdiği, henüz silinmemiş klasörler
    lock = threading.Lock()

    def fetch(job: dict) -> dict:
        rec, tag = job["rec"], job["tag"]
        if not job["downloadable"]:
            return {"kind": "local"}
        free_gb = shutil.disk_usage(data_dir).free / 1e9
        if free_gb < min_free_gb:
            return {"kind": "low_disk", "free_gb": free_gb}
        fdir = data_dir / str(rec["file_id"])
        pre_existing = fdir.exists()
        log.info("%s indiriliyor (%s) ...", tag, rec["file_id"])
        t0 = time.time()
        if not pre_existing:
            with lock:
                pending_delete.add(fdir)
        path = gdc_download(str(rec["file_id"]), data_dir, gdc_client, n_processes=gdc_processes)
        if path is None:
            return {"kind": "failed", "dir": None if pre_existing else fdir}
        log.info("%s indirildi: %.0f MB, %.0f sn", tag, path.stat().st_size / 1e6, time.time() - t0)
        return {"kind": "ok", "path": path, "dir": None if pre_existing else fdir}

    def cleanup(d: Path | None, tag: str, msg: bool = True):
        if d is None:
            return
        with lock:
            pending_delete.discard(d)
        if delete_after:
            shutil.rmtree(d, ignore_errors=True)
            if msg:
                log.info("%s → indirilen slayt silindi (sonuçlar korunur)", tag)

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gdc") if (prefetch and jobs) else None
    run_fetch = (lambda j: pool.submit(fetch, j)) if pool else None
    fut = run_fetch(jobs[0]) if (pool and jobs) else None
    try:
        for k, job in enumerate(jobs):
            rec, tag = job["rec"], job["tag"]
            got = fut.result() if fut is not None else fetch(job)
            fut = None
            if got["kind"] == "low_disk":
                log.error("Diskte %.1f GB boş alan kaldı (< %.1f GB). Toplu analiz güvenle DURDURULDU; yer açıp aynı komutu "
                          "tekrar çalıştırın, kaldığı yerden devam eder.", got["free_gb"], min_free_gb)
                bump("stopped_low_disk")
                break
            if pool and k + 1 < len(jobs):
                fut = run_fetch(jobs[k + 1])          # sıradaki slayt şimdiden inmeye başlasın
            if got["kind"] == "failed":
                cleanup(got["dir"], tag, msg=False)
                _write_stub(output_dir, rec, "download_failed")
                bump("download_failed")
                log.error("%s → indirme başarısız", tag)
                continue
            downloaded_dir = got.get("dir")
            try:
                if got["kind"] == "ok":
                    path = got["path"]
                    rec["image_path"] = str(path)
                    bc = parse_tcga_barcode(path.name.split(".")[0])
                    if bc["barcode_valid"]:
                        if bc["case_id"] != rec["case_id"]:
                            _write_stub(output_dir, rec, "skipped_id_error")
                            bump("skipped_id_error")
                            log.error("%s → indirilen dosya başka hastaya ait (%s)!", tag, bc["case_id"])
                            continue
                        rec["slide_barcode"] = path.name.split(".")[0]
                log.info("%s analiz ediliyor ...", tag)
                r = analyze_case(rec["case_id"], rec["image_path"], output_dir, meta=_meta_from_rec(rec), cfg=cfg,
                                 expected_md5=rec.get("expected_md5"), write_tables=False)
                bump(r["analysis_status"])
                log.info("%s → %s (%s s) | özet: %s", tag, r["analysis_status"], r.get("runtime_sec"), counts)
            finally:
                cleanup(downloaded_dir, tag)
    finally:
        if pool is not None:
            if fut is not None and not fut.done():
                log.info("Arka plandaki indirme bitirilmeden kapatılıyor ...")
            pool.shutdown(wait=True, cancel_futures=True)
            if delete_after:   # durdurma/hata: önden indirilmiş ama analiz edilmemiş slayt kalmasın
                with lock:
                    left = list(pending_delete)
                for d in left:
                    shutil.rmtree(d, ignore_errors=True)
                    log.info("Analiz edilmemiş önden indirilen slayt silindi: %s", d.name)
    table = write_aggregate_tables(output_dir)
    write_action_list(output_dir)
    log.info("TOPLU ANALİZ BİTTİ. Durumlar: %s", counts)
    return table


ACTION_RULES = [  # (koşul anahtarı, açıklama, öneri)
    ("STAIN_POSSIBLE_IHC_DAB", "Olası immünohistokimya (H&E değil)", "Aynı hastanın başka H&E slaytını seçin veya aynı grade'den yedek vaka"),
    ("STAIN_NOT_HE_LIKE", "Renk dağılımı H&E ile uyumsuz", "Görsel kontrol; H&E değilse slaytı/vakayı değiştirin"),
    ("STAIN_NO_TISSUE", "Değerlendirilebilir doku yok", "Slaytı değiştirin"),
    ("NO_OR_LOW_TISSUE", "Doku çok az", "Slaytı değiştirin"),
    ("OUT_OF_FOCUS", "Dokunun çoğu odak dışı", "Aynı hastanın başka slaytını deneyin"),
    ("RESOLUTION_INSUFFICIENT", "Çözünürlük hedef için yetersiz", "target_mpp ayarını gözden geçirin veya slaytı değiştirin"),
    ("INSUFFICIENT_NUCLEI", "Çekirdek sayısı yetersiz", "Görsel kontrol; doku azsa slaytı değiştirin"),
    ("MPP_MISSING", "µm/piksel bilgisi yok", "Slaytı değiştirin (ölçek tahmin edilmez)"),
    ("STAIN_HE_FADED", "Soluk boyama (analiz edildi)", "Görsel kontrol; mümkünse daha iyi boyalı slayt"),
    ("MOSTLY_OUT_OF_FOCUS", "Dokunun yarıdan fazlası bulanık (analiz edildi)", "Görsel kontrol; mümkünse başka slayt"),
]


def write_action_list(output_dir: Path) -> pd.DataFrame:
    """Değiştirilmesi/kontrol edilmesi gereken vakalar + küçük resim galerisi (aksiyon_listesi.xlsx/png).
    Yedek vaka AYNI grade'den ve entropi sonuçlarına BAKILMADAN seçilmelidir."""
    output_dir = Path(output_dir)
    p = output_dir / "case_level_results.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        flags = str(r.get("qc_flags") or "")
        st = str(r.get("analysis_status"))
        hit = next(((d, rec) for k, d, rec in ACTION_RULES if k in flags), None)
        if hit is None and st in ("failed_error", "download_failed", "missing_file", "skipped_id_error"):
            hit = (st, "Hata/indirme sorunu: case_log.txt'e bakın; gerekirse tekrar çalıştırın")
        if hit:
            rows.append({"excel_row": r.get("excel_row"), "case_id": r.get("case_id"), "slide_id": r.get("slide_id"),
                         "grade": r.get("grade"), "durum": st, "neden": hit[0], "öneri": hit[1],
                         "zorunlu": st not in SUCCESS_STATUSES, "error_message": r.get("error_message")})
    act = pd.DataFrame(rows, columns=["excel_row", "case_id", "slide_id", "grade", "durum", "neden", "öneri", "zorunlu", "error_message"])
    with pd.ExcelWriter(output_dir / "aksiyon_listesi.xlsx", engine="openpyxl") as xw:
        act.sort_values(["zorunlu", "grade"], ascending=[False, True]).to_excel(xw, sheet_name="aksiyon", index=False)
        if len(act):
            act[act["zorunlu"]].groupby("grade").size().rename("degistirilecek_vaka").reset_index().to_excel(
                xw, sheet_name="grade_bazinda_eksik", index=False)
    thumbs = []
    from PIL import Image
    for r in act.itertuples():
        tp = output_dir / "cases" / str(r.slide_id) / "thumbnail.png"
        if tp.exists():
            im = Image.open(tp).convert("RGB")
            im.thumbnail((300, 300))
            canvas = Image.new("RGB", (300, 300), (252, 252, 251))
            canvas.paste(im, ((300 - im.width) // 2, (300 - im.height) // 2))
            thumbs.append((np.asarray(canvas), f"{r.case_id} {r.grade}: {r.neden}"[:44],
                           (200, 30, 30) if r.zorunlu else (230, 150, 0)))
    if thumbs:
        pixel_montage([thumbs[k:k + 5] for k in range(0, len(thumbs), 5)], output_dir / "aksiyon_listesi_galeri.png",
                      title="Değiştirilmesi (kırmızı) / kontrol edilmesi (turuncu) gereken slaytlar", font_size=13)
    return act


def _refresh_metadata(result_path: Path, rec: dict):
    """Segmentasyonu tekrar etmeden Excel'den gelen etiketleri (grade, grade_group) günceller."""
    p = json.loads(result_path.read_text(encoding="utf-8"))
    changed = False
    for k in ("grade", "grade_group", "excel_row", "slide_uuid", "file_id"):
        v = rec.get(k)
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and p.get(k) != v:
            p[k] = v.item() if isinstance(v, (np.integer, np.floating)) else v
            changed = True
    if changed:
        result_path.write_text(json.dumps(p, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def recompute_all_entropies(output_dir: Path, cfg: Config) -> pd.DataFrame:
    """2. AŞAMA komutu: tüm vakalarda kayıtlı çekirdek tablolarından entropiyi yeniden hesaplar."""
    output_dir = Path(output_dir)
    n = 0
    for rp in sorted((output_dir / "cases").glob("*/case_result.json")):
        p = json.loads(rp.read_text(encoding="utf-8"))
        if p.get("analysis_status") not in SUCCESS_STATUSES and "INSUFFICIENT_NUCLEI" not in (p.get("qc_flags") or ""):
            continue
        if recompute_entropy(rp.parent, cfg) is not None:
            n += 1
            log.info("Entropi güncellendi: %s", rp.parent.name)
    write_run_metadata(output_dir, cfg)
    log.info("%d vaka güncellendi", n)
    return write_aggregate_tables(output_dir)


def _write_stub(output_dir: Path, rec: dict, status: str):
    folder = rec.get("slide_barcode") if rec.get("slide_barcode") and BARCODE_RE.match(str(rec["slide_barcode"])) else f"ROW{int(rec['excel_row']):04d}_{rec['case_id']}"
    d = output_dir / "cases" / folder
    if (d / "case_result.json").exists() and status == "skipped_duplicate":
        d = output_dir / "cases" / f"ROW{int(rec['excel_row']):04d}_{rec['case_id']}_duplicate"
    d.mkdir(parents=True, exist_ok=True)
    stub = {"case_id": rec["case_id"], "slide_id": rec.get("slide_barcode"), "slide_uuid": rec.get("slide_uuid"),
            "file_id": rec.get("file_id"), "grade": rec.get("grade"), "grade_group": rec.get("grade_group"),
            "excel_row": rec.get("excel_row"), "image_path": rec.get("image_path"), "analysis_status": status,
            "quality_control_status": "NOT_ANALYZED", "error_message": status, "config_hash": None}
    (d / "case_result.json").write_text(json.dumps(stub, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# =============================================================================
# 10a. GDC İNDİRME LİSTESİ (manifest) — Excel'deki File ID'lerden
# =============================================================================
GDC_API = "https://api.gdc.cancer.gov"
GDC_FIELDS = "file_id,file_name,file_size,md5sum,state,experimental_strategy,data_format,cases.submitter_id"


def _gdc_files_query(filters: dict, size: int = 10000) -> list[dict]:
    """GDC /files uç noktası (POST, JSON). Döner: hit listesi."""
    import urllib.request
    body = json.dumps({"filters": filters, "fields": GDC_FIELDS, "format": "JSON", "size": str(size)}).encode()
    req = urllib.request.Request(f"{GDC_API}/files", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["data"]["hits"]


def _hit_case(h: dict) -> str | None:
    cs = h.get("cases") or []
    return cs[0].get("submitter_id") if len(cs) == 1 else None


def gdc_inventory(case_ids: list[str], file_ids: list[str], query=_gdc_files_query) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) 200 hastanın GDC'deki TÜM SVS slaytları, (2) Excel'de seçili File ID'lerin GDC kaydı.
    query parametresi test için değiştirilebilir."""
    inv, sel = [], []
    for k in range(0, len(case_ids), 100):
        inv += query({"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids[k:k + 100]}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}}]})
    for k in range(0, len(file_ids), 100):
        sel += query({"op": "in", "content": {"field": "file_id", "value": file_ids[k:k + 100]}})

    def frame(hits):
        rows = [{"file_id": h.get("file_id") or h.get("id"), "file_name": h.get("file_name"), "file_size": h.get("file_size"),
                 "md5sum": h.get("md5sum"), "state": h.get("state"), "experimental_strategy": h.get("experimental_strategy"),
                 "data_format": h.get("data_format"), "gdc_case": _hit_case(h)} for h in hits]
        d = pd.DataFrame(rows, columns=["file_id", "file_name", "file_size", "md5sum", "state", "experimental_strategy",
                                        "data_format", "gdc_case"]).drop_duplicates("file_id")
        d["slide_section"] = d["file_name"].astype(str).str.extract(r"-([A-Z]{2})[A-Z0-9]\.")[0]
        return d
    return frame(inv), frame(sel)


def suggest_fixes(bad: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Sorunlu satırlar için GDC envanterinden düzeltme önerisi.
    Esas alınan: Excel'deki Submitter ID (slayt barcode'u) — slaytı incelerken kaydettiğin kimlik budur."""
    inv = inv.assign(barcode=inv["file_name"].astype(str).str.split(".").str[0],
                     slide_uuid=inv["file_name"].astype(str).str.split(".").str[1].str.lower())
    out = []
    for r in bad.itertuples():
        sub, case = r.slide_barcode_excel, r.case_id_excel
        fbar = str(r.file_name).split(".")[0] if isinstance(r.file_name, str) else None
        fuid = str(r.file_name).split(".")[1].lower() if isinstance(r.file_name, str) else None
        rec = {"excel_row": r.excel_row, "case_id": case, "grade": r.grade, "sorun": r.sorun,
               "excel_submitter_id": sub, "excel_file_id": r.file_id_excel, "excel_slide_id": r.slide_uuid_excel,
               "gdc_dosya_adi": r.file_name if isinstance(r.file_name, str) else None,
               "oneri_file_id": None, "oneri_slide_id": None, "otomatik": False, "onem": "elle", "aciklama": ""}
        cand = inv[(inv["gdc_case"] == case) & (inv["barcode"] == sub)]
        same_as_slide = str(r.file_id_excel).lower() == str(r.slide_uuid_excel).lower()
        if fbar is not None and fbar == sub and r.sorun.startswith("Selected Slide ID"):
            rec.update(oneri_file_id=r.file_id_excel, oneri_slide_id=fuid, otomatik=True, onem="zararsız",
                       aciklama=f"File ID doğru slayt ({sub}); yalnız Selected Slide ID sütunu yanlış → {fuid}")
        elif len(cand) == 1:
            c = cand.iloc[0]
            what = ("File ID sütununa Slide ID kopyalanmış" if (fbar is None and same_as_slide) else
                    "File ID GDC'de yok" if fbar is None else f"File ID başka bir slayta ({fbar}) işaret ediyor")
            rec.update(oneri_file_id=c["file_id"], oneri_slide_id=c["slide_uuid"], otomatik=True, onem="önemli",
                       aciklama=f"{what}; Submitter ID {sub} olan slaytın doğru File ID'si önerildi ({c['size_MB']} MB)")
        elif len(cand) > 1:
            rec["aciklama"] = f"{sub} için GDC'de {len(cand)} dosya var; tum_slaytlar sayfasından birini seçin"
        else:
            opts = inv[(inv["gdc_case"] == case) & inv["slide_section"].isin(["TS", "BS", "MS"])]["barcode"].tolist()
            rec["aciklama"] = (f"Submitter ID {sub} GDC'de bulunamadı. Bu hastanın dondurulmuş slaytları: {', '.join(opts) or 'yok'}"
                               " → birini seçip GDC'de gözden geçirin")
        out.append(rec)
    return pd.DataFrame(out, columns=["excel_row", "case_id", "grade", "onem", "otomatik", "aciklama", "sorun",
                                      "excel_submitter_id", "excel_file_id", "excel_slide_id", "gdc_dosya_adi",
                                      "oneri_file_id", "oneri_slide_id"])


def write_fixed_excel(excel: Path, fixes: pd.DataFrame, dst: Path) -> Path | None:
    """Orijinal Excel'in KOPYASINDA otomatik düzeltmeleri uygular (biçim korunur; değişen hücreler sarı)."""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill
    auto = fixes[fixes["otomatik"]]
    if auto.empty:
        return None
    wb = load_workbook(excel)
    ws = wb["Case_Selection"]
    head = {str(c.value).strip(): c.column for c in ws[1] if c.value}
    need = ["File ID", "Selected Slide ID"]
    if any(h not in head for h in need):
        log.warning("Excel başlıkları bulunamadı (%s); düzeltilmiş kopya yazılmadı", need)
        return None
    yellow = PatternFill("solid", fgColor="FFF2CC")
    stamp = datetime.now().strftime("%Y-%m-%d")
    for r in auto.itertuples():
        row = int(r.excel_row)
        for col, val in (("File ID", r.oneri_file_id), ("Selected Slide ID", r.oneri_slide_id)):
            cell = ws.cell(row=row, column=head[col])
            if str(cell.value or "").strip().lower() != str(val).lower():
                cell.value = val
                cell.fill = yellow
        if "Notes" in head:
            nc = ws.cell(row=row, column=head["Notes"])
            prev = "" if nc.value in (None, "") else f"{nc.value} | "
            nc.value = f"{prev}[otomatik düzeltme {stamp}] {r.aciklama}"
            nc.fill = yellow
    wb.save(dst)
    return dst


def make_manifest(excel: Path, out: Path, query=_gdc_files_query) -> Path:
    """Excel'deki File ID'leri GDC'de DOĞRULAR ve envanter çıkarır (internet gerekir):
      • GDC'de bulunamayan / başka hastaya ait / slayt olmayan File ID'ler → Excel satırıyla
      • her hastanın GDC'deki tüm slaytları (TS/BS/DX) ve boyutları → DX'e geçiş kararı için
      • gdc_manifest.txt (bulunan dosyalar, gdc-client biçimi)  →  out/gdc_slayt_envanteri.xlsx"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df, _ = load_case_table(excel)
    keep = df[~df["is_duplicate"] & ~df["manual_excluded"]
              & df["selected"].astype(str).str.strip().str.lower().isin(["yes", "evet"])].copy()
    ids = [i for i in keep["file_id_excel"].dropna().tolist() if UUID_RE.match(i)]
    (out / "file_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    cases = sorted(keep["case_id_excel"].dropna().unique().tolist())
    try:
        inv, sel = gdc_inventory(cases, ids, query)
    except Exception as exc:  # noqa: BLE001
        log.warning("GDC API'ye ulaşılamadı (%s). İnternet bağlantısını kontrol edin; analiz yine de çalışır.", exc)
        return out / "file_ids.txt"

    # 1) Seçili dosyaların doğrulanması
    m = keep[["excel_row", "case_id_excel", "grade", "file_id_excel", "slide_uuid_excel", "slide_barcode_excel"]].merge(
        sel, left_on="file_id_excel", right_on="file_id", how="left")
    m["sorun"] = ""
    m.loc[m["file_name"].isna(), "sorun"] = "File ID GDC'de bulunamadı (yanlış/eski ID?)"
    wrong_case = m["file_name"].notna() & m["gdc_case"].notna() & (m["gdc_case"] != m["case_id_excel"])
    m.loc[wrong_case, "sorun"] = "File ID başka hastaya ait: " + m.loc[wrong_case, "gdc_case"].astype(str)
    not_svs = m["file_name"].notna() & (m["data_format"] != "SVS")
    m.loc[not_svs, "sorun"] = "Dosya bir slayt (SVS) değil: " + m.loc[not_svs, "data_format"].astype(str)
    uuid_mis = m["file_name"].notna() & m["slide_uuid_excel"].notna() & ~m.apply(   # GDC bazı adlarda BÜYÜK harf kullanır
        lambda r: str(r["slide_uuid_excel"]).lower() in str(r["file_name"]).lower(), axis=1)
    m.loc[uuid_mis & (m["sorun"] == ""), "sorun"] = "Selected Slide ID dosya adıyla uyuşmuyor"
    m["size_MB"] = (m["file_size"] / 1e6).round(1)
    bad = m[m["sorun"] != ""]

    # 2) Envanter: hasta başına TS/BS/DX
    inv = inv.merge(keep[["case_id_excel", "grade"]], left_on="gdc_case", right_on="case_id_excel", how="left")
    inv["size_MB"] = (inv["file_size"] / 1e6).round(1)
    rows = []
    for c, g in inv.groupby("gdc_case"):
        dx = g[g["slide_section"] == "DX"].sort_values("file_size")
        fr = g[g["slide_section"].isin(["TS", "BS", "MS"])]
        rows.append({"case_id": c, "grade": g["grade"].iloc[0], "n_frozen_TS_BS": len(fr), "n_DX": len(dx),
                     "en_kucuk_DX_file_id": dx["file_id"].iloc[0] if len(dx) else None,
                     "en_kucuk_DX_dosya": dx["file_name"].iloc[0] if len(dx) else None,
                     "en_kucuk_DX_MB": dx["size_MB"].iloc[0] if len(dx) else None})
    per_case = pd.DataFrame(rows)
    fixes = suggest_fixes(bad, inv)
    if len(fixes):
        write_fixed_excel(excel, fixes, out / (Path(excel).stem + "_DUZELTILMIS.xlsx"))
    sel_ok = m[m["sorun"] == ""]
    ozet = pd.DataFrame([
        ("Excel'deki vaka", len(keep)),
        ("GDC'de doğrulanan seçili dosya", len(sel_ok)),
        ("Sorunlu satır (aşağıdaki sayfa)", len(bad)),
        ("Seçili slaytların toplam boyutu (GB)", round(sel_ok["file_size"].sum() / 1e9, 1)),
        ("Seçili slaytların ortalama boyutu (MB)", round(sel_ok["file_size"].mean() / 1e6, 0) if len(sel_ok) else None),
        ("Seçili slayt türleri", sel_ok["slide_section"].value_counts().to_dict()),
        ("DX (tanısal) slaytı olan hasta", int((per_case["n_DX"] > 0).sum()) if len(per_case) else 0),
        ("DX'e geçilirse toplam boyut (her hastanın en küçük DX'i, GB)",
         round(per_case["en_kucuk_DX_MB"].sum() / 1e3, 1) if len(per_case) else None),
    ], columns=["ölçüt", "değer"])
    if len(per_case):
        dxg = per_case.assign(DX_var=per_case["n_DX"] > 0).groupby("grade")["DX_var"].agg(["sum", "count"])
        for g_, r_ in dxg.iterrows():
            ozet.loc[len(ozet)] = (f"DX'i olan hasta — {g_}", f"{int(r_['sum'])} / {int(r_['count'])}")
    with pd.ExcelWriter(out / "gdc_slayt_envanteri.xlsx", engine="openpyxl") as xw:
        ozet.to_excel(xw, sheet_name="ozet", index=False)
        bad.drop(columns=["file_id"], errors="ignore").to_excel(xw, sheet_name="sorunlu_satirlar", index=False)
        fixes.to_excel(xw, sheet_name="duzeltme_onerileri", index=False)
        m.drop(columns=["file_id"], errors="ignore").to_excel(xw, sheet_name="secili_dosyalar", index=False)
        per_case.to_excel(xw, sheet_name="hasta_basina_slaytlar", index=False)
        inv.drop(columns=["case_id_excel"]).sort_values(["gdc_case", "slide_section"]).to_excel(
            xw, sheet_name="tum_slaytlar", index=False)
    mp = out / "gdc_manifest.txt"
    sel_ok.assign(file_size=sel_ok["file_size"].astype("int64")).rename(
        columns={"file_id_excel": "id", "file_name": "filename", "md5sum": "md5", "file_size": "size"})[
        ["id", "filename", "md5", "size", "state"]].to_csv(mp, sep="\t", index=False)

    print("\n" + ozet.to_string(index=False))
    if len(bad):
        print(f"\n⚠ {len(bad)} SORUNLU SATIR:")
        for r in fixes.itertuples():
            print(f"   satır {r.excel_row} · {r.case_id} · [{r.onem}] {r.aciklama}")
        n_auto = int((fixes["otomatik"]).sum())
        if n_auto:
            print(f"\n   → {n_auto} satır için düzeltilmiş Excel KOPYASI yazıldı: {out / (Path(excel).stem + '_DUZELTILMIS.xlsx')}")
            print("     Orijinal Excel'e dokunulmadı. Kopyayı açıp kontrol edin (değişen hücreler sarı, Notes sütununda açıklama).")
        if n_auto < len(fixes):
            print("   → Otomatik düzeltilemeyenler için: gdc_slayt_envanteri.xlsx → tum_slaytlar sayfasında hastanın slaytlarına bakın.")
    else:
        print("\n✓ Tüm seçili File ID'ler GDC'de doğrulandı ve doğru hastaya ait.")
    print(f"\nAyrıntı: {out / 'gdc_slayt_envanteri.xlsx'}")
    return mp


DX_RULE = ("Her hasta için GDC'de 'released' durumdaki DX (FFPE tanısal) slaytlarından dosya boyutu EN BÜYÜK olan seçildi "
           "(boyut, taranan doku miktarının vekili; kural sonuçlara bakılmadan önceden belirlendi).")


def make_dx_excel(excel: Path, inventory_xlsx: Path, dst: Path) -> pd.DataFrame:
    """Excel'in DX (FFPE) sürümünü yazar. Hastalar, grade'ler, satır sırası ve diğer sütunlar AYNEN kalır;
    yalnız Submitter ID / File ID / Selected Slide ID DX slaytıyla değiştirilir. Eski (dondurulmuş) kimlikler
    'Frozen ...' sütunlarında saklanır. Orijinal Excel'e dokunulmaz. Döner: seçim tablosu."""
    from openpyxl import load_workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Font, PatternFill
    inv = pd.read_excel(inventory_xlsx, sheet_name="tum_slaytlar", dtype={"file_id": str})
    dx = inv[(inv["slide_section"] == "DX") & (inv["state"].astype(str) == "released")].copy()
    dx["barcode"] = dx["file_name"].astype(str).str.split(".").str[0]
    dx["slide_uuid"] = dx["file_name"].astype(str).str.split(".").str[1].str.lower()
    dx = dx.sort_values(["gdc_case", "file_size", "file_name"], ascending=[True, False, True])
    wb = load_workbook(excel)
    ws = wb["Case_Selection"]
    head = {str(c.value).strip(): c.column for c in ws[1] if c.value}
    for need in ("Case ID", "Submitter ID", "File ID", "Selected Slide ID"):
        if need not in head:
            raise SystemExit(f"Excel'de '{need}' sütunu yok")
    extra = ["Frozen Submitter ID", "Frozen File ID", "Frozen Slide ID", "DX seçenek sayısı", "DX dosya boyutu (MB)"]
    col = ws.max_column
    for name in extra:
        if name not in head:
            col += 1
            ws.cell(row=1, column=col, value=name).font = Font(bold=True)
            head[name] = col
    blue = PatternFill("solid", fgColor="DDEBF7")
    rows = []
    for r in range(2, ws.max_row + 1):
        case = str(ws.cell(row=r, column=head["Case ID"]).value or "").strip()
        if not case:
            continue
        g = dx[dx["gdc_case"] == case]
        old = {k: ws.cell(row=r, column=head[k]).value for k in ("Submitter ID", "File ID", "Selected Slide ID")}
        rec = {"excel_row": r, "case_id": case, "eski_submitter_id": old["Submitter ID"], "n_DX": len(g),
               "dx_submitter_id": None, "dx_file_id": None, "dx_size_MB": None,
               "diger_DX": ", ".join(g["barcode"].iloc[1:].tolist())}
        if str(old["Submitter ID"] or "").strip().upper().endswith(tuple(f"-DX{k}" for k in "123456789ABC")):
            rec.update(dx_submitter_id=old["Submitter ID"], dx_file_id=old["File ID"], durum="zaten DX")
        elif len(g) == 0:
            rec["durum"] = "DX YOK — satır dondurulmuş slaytla kaldı (analizde FROZEN olarak işaretlenir)"
        else:
            c = g.iloc[0]
            for k, v in (("Frozen Submitter ID", old["Submitter ID"]), ("Frozen File ID", old["File ID"]),
                         ("Frozen Slide ID", old["Selected Slide ID"]),
                         ("Submitter ID", c["barcode"]), ("File ID", c["file_id"]), ("Selected Slide ID", c["slide_uuid"]),
                         ("DX seçenek sayısı", len(g)), ("DX dosya boyutu (MB)", round(float(c["file_size"]) / 1e6, 1))):
                cell = ws.cell(row=r, column=head[k], value=v)
                if k in ("Submitter ID", "File ID", "Selected Slide ID"):
                    cell.fill = blue
            rec.update(dx_submitter_id=c["barcode"], dx_file_id=c["file_id"], dx_size_MB=round(float(c["file_size"]) / 1e6, 1),
                       durum="DX seçildi")
        rows.append(rec)
    note = ("Bu sütunlardaki değerlendirmeler DONDURULMUŞ (TS/BS) slayta aittir; DX slaytta pilot/QC galerisiyle yeniden "
            "kontrol edilir. GDC Tumor/Normal/Stromal % değerleri de dondurulmuş kesitlerden gelir.")
    for k in ("Tumor Cells (%)", "Image Quality"):
        if k in head:
            ws.cell(row=1, column=head[k]).comment = Comment(note, "dx-gecis")
    sel = pd.DataFrame(rows)
    if "DX_gecis" in wb.sheetnames:
        del wb["DX_gecis"]
    wl = wb.create_sheet("DX_gecis")
    wl.append(["Kural", DX_RULE]); wl.append(["Tarih", datetime.now().strftime("%Y-%m-%d %H:%M")])
    wl.append(["Kaynak Excel", Path(excel).name]); wl.append(["Not", note]); wl.append([])
    wl.append(list(sel.columns))
    for t in sel.itertuples(index=False):
        wl.append(list(t))
    dst = Path(dst)
    wb.save(dst)
    return sel




# =============================================================================
# 10b. BOYA TARAMASI (screen) — ağır analizden ÖNCE tüm slaytlar, küçük resim düzeyinde
# =============================================================================
def run_screen(excel, data_dir, output_dir, cfg: Config, downsample: int = 32) -> pd.DataFrame:
    """Her slayt için: H&E sınıfı, çözünürlük uygunluğu, doku miktarı. Sorunlu slaytlar için öneri.
    Çıktı: stain_screen/ → stain_screen.xlsx, galeri sayfaları (PNG), boya dağılım grafiği."""
    from PIL import Image
    output_dir = Path(output_dir)
    out = output_dir / "stain_screen"
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    cases, issues = prepare_cases(excel, data_dir)
    write_validation(output_dir, cases, issues)
    rows = []
    todo = cases[cases["prep_status"].isin(["ready"])]
    for i, rec in enumerate(todo.to_dict("records"), 1):
        row = {k: rec.get(k) for k in ("excel_row", "case_id", "slide_barcode", "grade", "grade_group", "n_slides_available", "image_path")}
        bc = parse_tcga_barcode(rec.get("slide_barcode"))
        row.update(tss_code=bc["tss_code"], slide_kind=bc["slide_kind"])
        try:
            src = open_image(rec["image_path"])
            row.update(mpp=src.mpp, objective_power=getattr(src, "objective_power", None), width=src.width, height=src.height)
            tw, th = max(src.width // downsample, 1), max(src.height // downsample, 1)
            thumb = src.thumbnail(tw, th)
            src.close()
            mask, minfo = compute_tissue_mask(thumb, (src.mpp or 0.5) * downsample, cfg)
            io = estimate_background_io(thumb, mask, cfg)
            row.update(tissue_fraction=minfo.get("tissue_fraction"), **screen_stain(thumb, mask, io, cfg))
            row["resolution_ok"] = bool(src.mpp and src.mpp <= cfg.target_mpp * (1 + cfg.mpp_tolerance))
            im = Image.fromarray(thumb)
            im.thumbnail((400, 400))
            im.save(out / "thumbs" / f"{rec['slide_barcode']}.png")
        except Exception as exc:  # noqa: BLE001
            row.update(stain_class="ERROR", error=f"{type(exc).__name__}: {exc}")
        cls = row.get("stain_class")
        n_alt = row.get("n_slides_available")
        if cls in STAIN_FAIL_CLASSES or cls == "ERROR":
            row["recommendation"] = (f"SLAYTI DEĞİŞTİRİN: aynı hastanın başka bir H&E slaytı (GDC'de {n_alt} slayt var) "
                                     f"veya aynı grade'den ({row.get('grade')}) yedek vaka")
        elif cls in ("HE_ATYPICAL", "HE_FADED"):
            row["recommendation"] = "Görsel kontrol edin; analiz edilir ve işaretlenir (normalizasyon uygulanır)"
        elif row.get("resolution_ok") is False:
            row["recommendation"] = f"Çözünürlük ({row.get('mpp')}) hedef {cfg.target_mpp} µm/px için yetersiz"
        else:
            row["recommendation"] = "Uygun"
        rows.append(row)
        log.info("[%d/%d] %s → %s", i, len(todo), rec["case_id"], cls)
    df = pd.DataFrame(rows)
    missing = cases[cases["prep_status"] != "ready"][["excel_row", "case_id", "slide_barcode", "grade", "prep_status"]]
    with pd.ExcelWriter(out / "stain_screen.xlsx", engine="openpyxl") as xw:
        if len(df):
            df.groupby("stain_class").size().rename("n").reset_index().to_excel(xw, sheet_name="ozet", index=False)
            pd.crosstab(df["stain_class"], df["grade"]).to_excel(xw, sheet_name="sinif_x_grade")
            pd.crosstab(df["tss_code"], df["grade"]).to_excel(xw, sheet_name="merkez_x_grade")
            df[df["recommendation"] != "Uygun"].to_excel(xw, sheet_name="aksiyon_gerekenler", index=False)
            df.to_excel(xw, sheet_name="tum_slaytlar", index=False)
        missing.to_excel(xw, sheet_name="dosyasi_olmayanlar", index=False)
        pd.DataFrame([{"sinif": k, "anlam": v} for k, v in STAIN_CLASSES.items()]).to_excel(xw, sheet_name="sinif_aciklamalari", index=False)
    if len(df):
        _screen_figures(df, out, cfg)
    log.info("Boya taraması bitti → %s", out)
    return df


_CLASS_COLOR = {"HE_TYPICAL": "#1baf7a", "HE_ATYPICAL": "#eda100", "HE_FADED": "#2a78d6",
                "POSSIBLE_IHC_DAB": "#e34948", "NOT_HE_LIKE": "#4a3aa7", "NO_TISSUE": "#52514e", "ERROR": "#0b0b0b"}


def _screen_figures(df: pd.DataFrame, out: Path, cfg: Config, per_page: int = 30):
    from PIL import Image
    plt = _plt()
    order = {"POSSIBLE_IHC_DAB": 0, "NOT_HE_LIKE": 1, "ERROR": 2, "NO_TISSUE": 3, "HE_FADED": 4, "HE_ATYPICAL": 5, "HE_TYPICAL": 6}
    d = df.assign(_o=df["stain_class"].map(order).fillna(9)).sort_values(["_o", "grade", "case_id"])
    for page, start in enumerate(range(0, len(d), per_page), 1):
        chunk = d.iloc[start:start + per_page]
        fig, axs = plt.subplots(5, 6, figsize=(18, 15.5))
        for ax in axs.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axs.ravel(), chunk.iterrows()):
            tp = out / "thumbs" / f"{r['slide_barcode']}.png"
            if tp.exists():
                ax.imshow(np.asarray(Image.open(tp)))
            c = _CLASS_COLOR.get(r["stain_class"], "#000")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(c); sp.set_linewidth(4)
            ax.set_title(f"{r['case_id']} · {r['grade']}\n{r['stain_class']}", fontsize=8.5, color=c, fontweight="bold")
        fig.suptitle(f"Boya taraması — sayfa {page} (sorunlu sınıflar önce)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"stain_screen_gallery_p{page:02d}.png", dpi=80)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for cls, g in df.groupby("stain_class"):
        c = _CLASS_COLOR.get(cls, "#000")
        if {"screen_cos_h", "screen_cos_e"} <= set(g.columns):
            ax[0].scatter(g["screen_cos_h"], g["screen_cos_e"], s=26, color=c, label=f"{cls} (n={len(g)})", edgecolor="white", lw=0.6)
        if {"he_hue_fraction", "dab_fraction"} <= set(g.columns):
            ax[1].scatter(g["he_hue_fraction"], g["dab_fraction"], s=26, color=c, edgecolor="white", lw=0.6)
    ax[0].axvline(cfg.he_min_cosine_typical, color="#52514e", ls="--", lw=1); ax[0].axhline(cfg.he_min_cosine_typical, color="#52514e", ls="--", lw=1)
    ax[0].set(xlabel="H vektörü – referans benzerliği (cos)", ylabel="E vektörü – referans benzerliği (cos)", title="Boya vektörleri (Macenko)")
    ax[0].legend(fontsize=8)
    ax[1].axvline(cfg.he_min_hue_fraction, color="#52514e", ls="--", lw=1); ax[1].axhline(cfg.dab_max_fraction, color="#52514e", ls="--", lw=1)
    ax[1].set(xlabel="pembe-mor (H&E) ton oranı", ylabel="kahverengi (DAB) ton oranı", title="Renk tonu dağılımı")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "stain_screen_scatter.png", dpi=130)
    plt.close(fig)


# =============================================================================
# 10c. QuPath İÇE AKTARMA — çekirdek sınırları alınır, ÖLÇÜMLER Python'da aynı tanımlarla yapılır
# =============================================================================
def _qupath_features(geojson_path: Path) -> list[dict]:
    data = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        feats = data.get("features", [data])
    else:
        feats = data
    out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        otype = props.get("objectType", props.get("object_type", "detection"))
        if otype == "annotation" and not props.get("isNucleus"):
            continue
        geom = f.get("nucleusGeometry") or props.get("nucleusGeometry") or f.get("geometry")
        if not geom:
            continue
        cl = props.get("classification") or {}
        name = cl.get("name") if isinstance(cl, dict) else None
        if name is None and isinstance(cl, dict) and cl.get("names"):
            name = ":".join(cl["names"])
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"] if geom["type"] == "MultiPolygon" else []
        for poly in polys:
            if poly:
                out.append({"ring": np.asarray(poly[0], float), "nucleus_class": name or "Unclassified"})
    return out


def polygons_to_nuclei(polys: list[dict], scale_l0: float, mpp: float, cfg: Config,
                       origin_xy=(0.0, 0.0)) -> pd.DataFrame:
    """Poligonları (seviye-0 koordinat) analiz ölçeğinde rasterize eder ve measure_nuclei ile ölçer
    → pipeline'ın kendi segmentasyonuyla BİREBİR aynı özellik tanımları (Crofton çevre vb.)."""
    from skimage.draw import polygon as draw_polygon
    rows = []
    ox, oy = origin_xy
    for k, p in enumerate(polys):
        ring = p["ring"]
        xs, ys = (ring[:, 0] - ox) / scale_l0, (ring[:, 1] - oy) / scale_l0
        x0, y0 = math.floor(xs.min()) - 2, math.floor(ys.min()) - 2
        w, h = math.ceil(xs.max()) - x0 + 3, math.ceil(ys.max()) - y0 + 3
        if w <= 2 or h <= 2 or w > 400 or h > 400:
            continue
        lab = np.zeros((h, w), np.int32)
        rr, cc = draw_polygon(ys - y0, xs - x0, (h, w))
        lab[rr, cc] = 1
        if lab.sum() == 0:
            continue
        m, excl = measure_nuclei(lab, np.zeros(lab.shape, np.float32), mpp, cfg)
        if len(m):
            r = m.iloc[0].to_dict()
            r["x_l0"] = ox + (x0 + r.pop("centroid_x_px")) * scale_l0
            r["y_l0"] = oy + (y0 + r.pop("centroid_y_px")) * scale_l0
            r["nucleus_class"] = p["nucleus_class"]
            r.pop("h_mean", None)
            rows.append(r)
    return pd.DataFrame(rows)


def import_qupath(output_dir: Path, slide_id: str, geojson: Path, cfg: Config) -> dict:
    """QuPath'te (ör. StarDist veya hücre sınıflandırıcı) bulunan çekirdekleri içe aktarır.
    Ön koşul: vaka önce `single` ile analiz edilmiş olmalı (tile ızgarası, odak haritası, mpp oradan alınır).
    Yalnız analiz edilen tile'lardaki çekirdekler alınır; odak etiketi aynı odak haritasından verilir."""
    case_dir = Path(output_dir) / "cases" / slide_id
    rp = case_dir / "case_result.json"
    if not rp.exists():
        raise SystemExit(f"{rp} yok. Önce bu vakayı `single` ile analiz edin (QC + tile ızgarası için).")
    res = json.loads(rp.read_text(encoding="utf-8"))
    tiles = pd.read_csv(case_dir / "tile_metrics.csv")
    sel = tiles[tiles["selected"].astype(bool)]
    tile_l0 = float(sel["tile_l0"].iloc[0])
    scale_l0 = tile_l0 / cfg.tile_size_px
    mpp = float(res["effective_mpp"])
    polys = _qupath_features(Path(geojson))
    if not polys:
        raise SystemExit("GeoJSON'da çekirdek poligonu bulunamadı (detection/cell nesneleri dışa aktarıldı mı?)")
    nuc = polygons_to_nuclei(polys, scale_l0, mpp, cfg)
    n_in = len(polys)
    tx = np.floor(nuc["x_l0"] / tile_l0).astype(int) * int(tile_l0)
    ty = np.floor(nuc["y_l0"] / tile_l0).astype(int) * int(tile_l0)
    nuc["tile_id"] = [f"x{a}_y{b}" for a, b in zip(tx, ty)]
    nuc = nuc[nuc["tile_id"].isin(set(sel["tile_id"]))].copy()
    fb_path = case_dir / "focus_blocks.csv.gz"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        size = float(fb["size_l0"].iloc[0])
        t0x = nuc["tile_id"].str.extract(r"x(\d+)_y(\d+)").astype(float)
        bx = np.floor((nuc["x_l0"] - t0x[0]) / size).astype(int)
        by = np.floor((nuc["y_l0"] - t0x[1]) / size).astype(int)
        fb = fb.assign(bx=np.round((fb["x_l0"] - fb["tile_id"].str.extract(r"x(\d+)_")[0].astype(float)) / size).astype(int),
                       by=np.round((fb["y_l0"] - fb["tile_id"].str.extract(r"_y(\d+)")[0].astype(float)) / size).astype(int))
        key = fb.set_index(["tile_id", "bx", "by"])
        idx = pd.MultiIndex.from_arrays([nuc["tile_id"], bx, by])
        nuc["focus_ratio"] = key["focus_ratio"].reindex(idx).to_numpy()
        nuc["in_focus"] = ~key["blurry"].reindex(idx).fillna(False).astype(bool).to_numpy()
    else:
        nuc["in_focus"] = True
        log.warning("focus_blocks.csv.gz yok → odak filtresi uygulanamadı")
    orig = case_dir / "nuclei_features.csv.gz"
    if orig.exists() and not str(res.get("segmenter", "")).startswith("qupath"):
        orig.rename(case_dir / f"nuclei_features_{res.get('segmenter', 'pipeline')}.csv.gz")
    nuc.to_csv(orig, index=False, compression="gzip")
    res["segmenter"] = f"qupath:{Path(geojson).name}"
    res["qupath_polygons_in_file"] = n_in
    res["qupath_nuclei_in_analyzed_tiles"] = int(len(nuc))
    rp.write_text(json.dumps(_json_clean(res), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    r = recompute_entropy(case_dir, cfg)
    write_aggregate_tables(Path(output_dir))
    log.info("QuPath içe aktarıldı: %d poligon → %d çekirdek (analiz tile'larında); sınıflar: %s",
             n_in, len(nuc), nuc["nucleus_class"].value_counts().to_dict())
    return r


# =============================================================================
# 10d. ODAK EŞİĞİ KALİBRASYONU
# =============================================================================
def calibrate_focus(output_dir: Path, slide_id: str, cfg: Config, n_levels: int = 8, per_level: int = 5) -> Path:
    """Bant oranının farklı düzeylerinden blok örnekleri (1:1) — eşiği gözle doğrulamak için.
    Kırmızı çerçeve = mevcut eşiğe göre bulanık sayılan (dışlanan) bloklar."""
    import cv2
    case_dir = Path(output_dir) / "cases" / slide_id
    res = json.loads((case_dir / "case_result.json").read_text(encoding="utf-8"))
    fb = pd.read_csv(case_dir / "focus_blocks.csv.gz")
    fb = fb[fb["determinate"].astype(bool)]
    src = open_image(res["image_path"])
    qs = np.linspace(0.02, 0.98, n_levels)
    targets = np.quantile(fb["focus_ratio"], qs)
    B = cfg.focus_block_px
    rows = []
    for t in targets:
        pick = fb.iloc[(fb["focus_ratio"] - t).abs().argsort().iloc[:per_level]]
        blurry = t < cfg.focus_min_band_ratio
        col = (200, 30, 30) if blurry else (27, 122, 74)
        row = []
        for i, (_, b) in enumerate(pick.iterrows()):
            im = src.read_region_l0(int(b.x_l0), int(b.y_l0), int(round(b.size_l0)), int(round(b.size_l0)))
            if im.shape[0] != B:
                im = cv2.resize(im, (B, B), interpolation=cv2.INTER_AREA)
            cap = (f"oran ≈ {t:.3f} → {'BULANIK (dışlanır)' if blurry else 'odaklı'}" if i == 0 else f"{b.focus_ratio:.3f}")
            row.append((im, cap, col))
        rows.append(row)
    src.close()
    outp = case_dir / "focus_calibration.png"
    pixel_montage(rows, outp, scale=2,
                  title=f"{slide_id} — odak kalibrasyonu (eşik {cfg.focus_min_band_ratio}); 2× büyütme, en yakın komşu (yumuşatma yok)")
    return outp


# =============================================================================
# 11. ÇIKTI TABLOLARI
# =============================================================================
PRIMARY_COLUMNS = [
    "case_id", "sample_id", "slide_id", "slide_uuid", "file_id", "slide_kind", "grade", "grade_group", "image_path",
    "image_width", "image_height", "image_format", "image_resolution_or_mpp", "effective_mpp", "objective_power",
    "tissue_pixel_count", "tissue_fraction", "tissue_area_mm2", "tissue_area_analyzed_mm2", "analysis_region",
    "n_tiles_eligible", "n_tiles_analyzed", "quality_control_status", "qc_flags", "preprocessing_method",
    "n_nuclei", "n_nuclei_detected", "n_nuclei_excluded_blur", "nuclei_per_mm2", "focus_blurry_tissue_fraction",
    "area_um2_entropy", "perimeter_um_entropy", "circularity_entropy", "eccentricity_entropy",
    "area_um2_entropy_norm", "perimeter_um_entropy_norm", "circularity_entropy_norm", "eccentricity_entropy_norm",
    "area_um2_entropy_n500", "perimeter_um_entropy_n500", "circularity_entropy_n500", "eccentricity_entropy_n500",
    "nuclear_entropy_index_exploratory",
    "grayscale_entropy", "red_channel_entropy", "green_channel_entropy", "blue_channel_entropy",
    "hematoxylin_entropy", "eosin_entropy",
    "patch_entropy_mean", "patch_entropy_std", "patch_entropy_median", "patch_entropy_min", "patch_entropy_max",
    "patch_nuc_area_entropy_mean", "patch_nuc_area_entropy_std", "patch_nuc_area_entropy_median",
    "patch_nuc_area_entropy_min", "patch_nuc_area_entropy_max",
    "histogram_bins", "log_base", "analysis_status", "error_message", "warnings", "config_hash",
]

COLUMN_DICTIONARY = [
    ("case_id", "-", "TCGA hasta (patient) ID; istatistikte bağımsız gözlem birimi"),
    ("sample_id", "-", "Barcode'dan: hasta + örnek tipi + vial (01A = primer solid tümör)"),
    ("slide_id", "-", "Slayt barcode'u (Submitter ID); TS/BS = dondurulmuş, DX = FFPE tanısal"),
    ("slide_uuid / file_id", "-", "GDC slide UUID ve indirilen dosyanın GDC file UUID'si (farklı şeylerdir)"),
    ("image_resolution_or_mpp", "µm/piksel", "Tarayıcı metadata'sındaki seviye-0 MPP; yoksa boş (tahmin edilmez)"),
    ("effective_mpp", "µm/piksel", "Analizin yapıldığı ölçek (40x slaytlar ≈0.5 µm/px'e küçültülür)"),
    ("tissue_fraction", "0-1", "Doku maskesinin tüm slayt alanına oranı (düşük çözünürlük maskesi)"),
    ("tissue_pixel_count", "piksel", "Analiz edilen tile'lardaki doku pikseli sayısı (effective_mpp ölçeğinde)"),
    ("tissue_area_analyzed_mm2", "mm²", "Analiz edilen doku alanı"),
    ("n_nuclei", "adet", "Entropide kullanılan ODAKLI çekirdek sayısı (kenar, alan, solidity, odak filtreleri sonrası)"),
    ("n_nuclei_detected / n_nuclei_excluded_blur", "adet", "Tespit edilen tüm çekirdekler / bulanık bloklarda kaldığı için dışlananlar"),
    ("focus_blurry_tissue_fraction", "0-1", "Odağı belirlenebilen doku bloklarının bulanık olanlarının oranı (128 px blok, bant oranı < 0.29)"),
    ("focus_ratio_median", "-", "Tile medyan bant oranı (yüksek = keskin)"),
    ("<feature>_entropy", "bit", "ANA ÖLÇÜT. Sabit bin'li histogramın Shannon entropisi, log2. feature ∈ {area_um2, perimeter_um, circularity, eccentricity}"),
    ("<feature>_entropy_norm", "0-1", "H / log2(K), K = bin sayısı"),
    ("<feature>_entropy_mm", "bit", "Miller–Madow yanlılık düzeltmeli entropi"),
    ("<feature>_entropy_n500", "bit", "500 çekirdeklik 200 rastgele alt örneklemin ortalama entropisi (n eşitlenmiş)"),
    ("<feature>_entropy_bK", "bit", "Duyarlılık analizi: aynı aralık, K bin"),
    ("<feature>_mean/std/median/iqr/cv", "özelliğin birimi", "Klasik dağılım ölçüleri; entropinin bunlara EK bilgi verip vermediğini test etmek için"),
    ("nuclear_entropy_index_exploratory", "0-1", "4 normalize entropinin ortalaması. KEŞİFSEL; doğrulanmış bir indeks değildir"),
    ("grayscale_entropy", "bit (0-8)", "ODAKLI doku piksellerinin 8-bit gri seviye (Y=0.299R+0.587G+0.114B) histogram entropisi, 256 bin"),
    ("red/green/blue_channel_entropy", "bit (0-8)", "Her RGB kanalının doku piksellerindeki 256-bin entropisi"),
    ("hematoxylin_entropy / eosin_entropy", "bit (0-8)", "Normalize H/E konsantrasyonunun SABİT [0,3] aralığında 256-bin entropisi"),
    ("patch_entropy_*", "bit", "Tile başına gri seviye entropisinin tile'lar arası özet istatistiği"),
    ("patch_nuc_area_entropy_*", "bit", "Tile başına nükleer alan entropisi (≥30 çekirdekli tile'lar); tile'lar bağımsız gözlem DEĞİLDİR"),
    ("quality_control_status", "PASS/WARN/FAIL", "Otomatik QC; WARN gerekçesi qc_flags ve warnings sütunlarında"),
    ("analysis_status", "-", "success / success_with_warnings / failed_qc / failed_error / missing_file / skipped_*"),
]


def write_run_metadata(output_dir: Path, cfg: Config):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"pipeline_version": PIPELINE_VERSION, "config_hash": cfg.config_hash(),
            "written_at": datetime.now().isoformat(timespec="seconds"), "config": cfg.to_dict(),
            "library_versions": library_versions()}
    (output_dir / "analysis_config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _folder_candidates(rec: dict) -> set[str]:
    """Bir Excel satırının sonuç klasörü olabilecek adlar (analyze_case ve _write_stub ile aynı kural)."""
    bc = rec.get("slide_barcode")
    names = set()
    if bc and BARCODE_RE.match(str(bc)):
        names.add(str(bc))
    else:
        names.add(str(rec.get("slide_uuid") or rec.get("case_id")))
    row = f"ROW{int(rec['excel_row']):04d}_{rec['case_id']}"
    names |= {row, row + "_duplicate"}
    return names


def write_validation(output_dir, cases: pd.DataFrame, issues: pd.DataFrame):
    """Eşleştirme tablosu + AKTİF klasör listesi. Excel'de slayt değiştirilirse (ör. TS/BS → DX) eski
    slaytın sonuçları 'eski' sayılır ve kohort tablosuna girmez (aynı hasta iki kez sayılmasın)."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cases.drop(columns=["id_warnings"], errors="ignore").to_csv(output_dir / "case_file_mapping.csv", index=False)
    issues.to_csv(output_dir / "excel_validation_issues.csv", index=False)
    active = sorted(set().union(*[_folder_candidates(r) for r in cases.to_dict("records")])) if len(cases) else []
    (output_dir / "active_case_folders.json").write_text(json.dumps(active, indent=1), encoding="utf-8")


def write_aggregate_tables(output_dir: Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    recs, stale = [], []
    af = output_dir / "active_case_folders.json"
    active = set(json.loads(af.read_text(encoding="utf-8"))) if af.exists() else None
    for p in sorted((output_dir / "cases").glob("*/case_result.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("Okunamadı %s: %s", p, exc)
            continue
        (stale if active is not None and p.parent.name not in active else recs).append(r)
    if stale:
        pd.DataFrame(stale).to_csv(output_dir / "stale_results_not_in_current_excel.csv", index=False)
        log.warning("%d sonuç güncel Excel'de yok (slayt değişmiş olabilir) → kohort tablosuna ALINMADI: %s",
                    len(stale), ", ".join(str(r.get("slide_id")) for r in stale[:10]))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    if "excel_row" in df:
        df = df.sort_values("excel_row", na_position="last")
    first = [c for c in PRIMARY_COLUMNS if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(output_dir / "case_level_results.csv", index=False)
    with pd.ExcelWriter(output_dir / "case_level_results.xlsx", engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="case_level_results", index=False)
        pd.DataFrame(COLUMN_DICTIONARY, columns=["column", "unit", "description"]).to_excel(xw, sheet_name="column_dictionary", index=False)
    qc_cols = [c for c in ["excel_row", "case_id", "slide_id", "slide_kind", "grade_group", "analysis_status",
                           "quality_control_status", "qc_flags", "warnings", "error_message", "image_resolution_or_mpp",
                           "effective_mpp", "tissue_fraction", "tissue_area_mm2", "n_tiles_eligible", "n_tiles_analyzed",
                           "n_nuclei", "n_nuclei_detected", "n_nuclei_excluded_blur", "focus_blurry_tissue_fraction",
                           "focus_ratio_median", "nuclei_per_mm2", "n_excl_border", "n_excl_area_small", "n_excl_area_large",
                           "n_excl_low_solidity", "stain_method_used", "stain_cos_h_to_ref", "stain_cos_e_to_ref",
                           "nuclear_threshold_h", "seg_hash", "entropy_hash", "tile_laplacian_var_median", "tile_dark_fraction_median",
                           "circularity_frac_gt1", "area_um2_frac_clipped", "perimeter_um_frac_clipped",
                           "file_md5", "md5_matches_gdc", "runtime_sec"] if c in df.columns]
    with pd.ExcelWriter(output_dir / "quality_control_report.xlsx", engine="openpyxl") as xw:
        df[qc_cols].to_excel(xw, sheet_name="qc_per_case", index=False)
        summ = df.groupby("analysis_status", dropna=False).size().rename("n").reset_index()
        summ.to_excel(xw, sheet_name="status_summary", index=False)
        vi = output_dir / "excel_validation_issues.csv"
        if vi.exists():
            pd.read_csv(vi).to_excel(xw, sheet_name="excel_id_validation", index=False)
        mp = output_dir / "case_file_mapping.csv"
        if mp.exists():
            pd.read_csv(mp).to_excel(xw, sheet_name="case_file_mapping", index=False)
    return df


# =============================================================================
# 12. SELF-TEST (bilimsel/teknik doğrulama)
# =============================================================================
DAB_VECTOR = (0.2680, 0.5703, 0.7764)   # Ruifrok & Johnston (2001) DAB OD vektörü


def _render_stains(layers, seed: int = 0, io: float = 245.0) -> np.ndarray:
    """Beer–Lambert: I = Io·10^(-Σ C_k·v_k). layers = [(konsantrasyon_haritası, OD_vektörü), ...]"""
    od = 0.0
    for conc, vec in layers:
        v = np.asarray(vec, float)
        od = od + conc[..., None] * (v / np.linalg.norm(v))
    rgb = io * np.power(10.0, -od) + np.random.default_rng(seed).normal(0, 2.0, od.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_he(h_conc: np.ndarray, e_conc: np.ndarray, seed: int = 0, io: float = 245.0) -> np.ndarray:
    ref = np.asarray(Config().ref_stain_matrix, float)
    return _render_stains([(h_conc, ref[:, 0]), (e_conc, ref[:, 1])], seed, io)


def _synthetic_layout(size_um=1024.0, radii_um=(3.0, 3.0), seed=0, n_nuclei=400, bg_frac=0.3):
    """Çekirdek yerleşimi µm cinsinden (çözünürlükten bağımsız) → aynı doku farklı mpp'lerde çizilebilir."""
    rng = np.random.default_rng(seed)
    tissue_w = size_um * (1 - bg_frac)
    rmax = max(radii_um)
    placed = []
    tries = 0
    while len(placed) < n_nuclei and tries < n_nuclei * 50:
        tries += 1
        cy, cx = rng.uniform(rmax + 1, size_um - rmax - 1), rng.uniform(rmax + 1, tissue_w - rmax - 1)
        if all((cy - py) ** 2 + (cx - px) ** 2 > (2.5 * rmax) ** 2 for py, px, _ in placed):
            placed.append((cy, cx, rng.uniform(*radii_um)))
    return placed, tissue_w


def _synthetic_he(size=2048, mpp=0.5, radii_um=(3.0, 3.0), seed=0, n_nuclei=400, bg_frac=0.3,
                  return_labels=False, nucleus_stain=None):
    """Eozin zemin + bilinen yarıçaplı çekirdekler + sağda boş cam. Beklenen alan (a=b=r): π r² µm².
    nucleus_stain: None = hematoksilen; 'dab' = kahverengi (IHC benzeri)."""
    size_um = size * mpp
    placed, tissue_w_um = _synthetic_layout(size_um, radii_um, seed, n_nuclei, bg_frac)
    h = np.zeros((size, size)); e = np.zeros((size, size)); dab = np.zeros((size, size))
    lab = np.zeros((size, size), np.int32)
    e[:, :int(tissue_w_um / mpp)] = 0.45
    yy, xx = np.mgrid[:size, :size]
    for k, (cy, cx, r) in enumerate(placed, 1):
        cyp, cxp, rp = cy / mpp, cx / mpp, r / mpp
        y0, y1, x0, x1 = int(cyp - rp - 2), int(cyp + rp + 3), int(cxp - rp - 2), int(cxp + rp + 3)
        d = (yy[y0:y1, x0:x1] - cyp) ** 2 + (xx[y0:y1, x0:x1] - cxp) ** 2 <= rp ** 2
        h[y0:y1, x0:x1][d] = 1.2
        e[y0:y1, x0:x1][d] = 0.15
        lab[y0:y1, x0:x1][d] = k
    if nucleus_stain == "dab":
        ref = np.asarray(Config().ref_stain_matrix, float)
        dab = np.where(h > 0, 1.0, 0.0) + np.where(e > 0, 0.25, 0.0)   # yaygın kahverengi boyanma
        img = _render_stains([(h * 0.3, ref[:, 0]), (dab, DAB_VECTOR)], seed)
    else:
        img = _render_he(h, e, seed)
    return (img, len(placed), lab) if return_labels else (img, len(placed))


def selftest(workdir: str | Path | None = None) -> bool:
    import tempfile
    from PIL import Image
    workdir = Path(workdir or tempfile.mkdtemp(prefix="kirc_selftest_"))
    workdir.mkdir(parents=True, exist_ok=True)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print(f"[{'OK ' if cond else 'FAIL'}] {name} {detail}")

    # --- Entropi matematiği ---
    check("Sabit dağılım → H=0", shannon_entropy_from_counts([100, 0, 0, 0]) == 0.0)
    check("2 eşit sınıf → H=1 bit", abs(shannon_entropy_from_counts([5, 5]) - 1.0) < 1e-12)
    check("256 eşit sınıf → H=8 bit", abs(shannon_entropy_from_counts(np.ones(256)) - 8.0) < 1e-12)
    check("Doğal log tabanı: 2 sınıf → ln2", abs(shannon_entropy_from_counts([1, 1], base=math.e) - math.log(2)) < 1e-12)
    check("Boş histogram → NaN (0 değil)", math.isnan(shannon_entropy_from_counts([0, 0, 0])))
    check("Ölçek değişmezliği (sayılar ×10)", abs(shannon_entropy_from_counts([1, 2, 3]) - shannon_entropy_from_counts([10, 20, 30])) < 1e-12)
    try:
        shannon_entropy_from_counts([-1, 2]); check("Negatif sayım reddedilir", False)
    except ValueError:
        check("Negatif sayım reddedilir", True)
    he = histogram_entropy(np.full(1000, 25.0), 0, 200, 40)
    check("Tek değerli özellik → H=0, norm=0", he["entropy"] == 0 and he["entropy_norm"] == 0)
    he = histogram_entropy([-5, 250, 100], 0, 200, 40)
    check("Aralık dışı değerler uç bin'lere kırpılır ve sayılır", he["n_clipped_low"] == 1 and he["n_clipped_high"] == 1 and he["n"] == 3)
    u = np.random.default_rng(1).uniform(0, 1, 200000)
    he = histogram_entropy(u, 0, 1, 20)
    check("Uniform veri → norm. entropi ≈ 1", he["entropy_norm"] > 0.999, f"({he['entropy_norm']:.5f})")
    check("Miller–Madow ≥ plug-in", miller_madow_entropy([3, 1, 1]) >= shannon_entropy_from_counts([3, 1, 1]))
    a = subsampled_entropy(u, 0, 1, 20, 500, 20, 7); b = subsampled_entropy(u, 0, 1, 20, 500, 20, 7)
    check("Alt örneklem entropisi seed ile tekrarlanabilir", a == b)

    # --- ID ayrıştırma ---
    p = parse_tcga_barcode("TCGA-BP-4165-01A-01-BS1")
    check("Barcode → case/sample/slide", p["case_id"] == "TCGA-BP-4165" and p["sample_id"] == "TCGA-BP-4165-01A" and p["slide_kind"] == "frozen_bottom")
    check("Geçersiz barcode yakalanır", not parse_tcga_barcode("?")["barcode_valid"])
    check("Excel tab/newline temizliği", clean_id("\t\n072af46b-fc1f-44a4-90ef-c293fa0d1eb3") == "072af46b-fc1f-44a4-90ef-c293fa0d1eb3")

    cfg = Config(min_nuclei_case=50, qc_min_tiles=1, min_nuclei_tile=10, stain_fit_tiles=8, subsample_reps=20)
    out = workdir / "out"
    # --- Uç senaryolar: hata vermeden durum kodu üretmeli ---
    white = workdir / "white.png"; Image.fromarray(np.full((1024, 1024, 3), 255, np.uint8)).save(white)
    r = analyze_case("TEST-WHITE", white, out, meta={"slide_barcode": "WHITE"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    check("Tamamen beyaz görüntü → failed_qc (doku yok)", r["analysis_status"] == "failed_qc" and "NO_OR_LOW_TISSUE" in r["qc_flags"])
    mono = workdir / "mono.png"; Image.fromarray(np.full((1024, 1024, 3), (200, 120, 170), np.uint8)).save(mono)  # tek renk, gürültüsüz
    r = analyze_case("TEST-MONO", mono, out, meta={"slide_barcode": "MONO"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    check("Tek renkli görüntü → çökmez, durum kodu üretir", r["analysis_status"] in ("failed_qc", "failed_error", "success_with_warnings"),
          f"({r['analysis_status']}: {r['error_message'][:60]})")
    corrupt = workdir / "corrupt.svs"; corrupt.write_bytes(np.random.default_rng(0).bytes(5000))
    r = analyze_case("TEST-CORRUPT", corrupt, out, meta={"slide_barcode": "CORRUPT"}, cfg=cfg, write_tables=False)
    check("Bozuk SVS → failed_error, çökme yok", r["analysis_status"] == "failed_error", f"({r['error_message'][:60]})")
    empty = workdir / "empty.svs"; empty.write_bytes(b"")
    r = analyze_case("TEST-EMPTY", empty, out, meta={"slide_barcode": "EMPTY"}, cfg=cfg, write_tables=False)
    check("0 baytlık dosya → failed_error", r["analysis_status"] == "failed_error")
    r = analyze_case("TEST-MISSING", workdir / "yok.svs", out, meta={"slide_barcode": "MISSING"}, cfg=cfg, write_tables=False)
    check("Olmayan dosya → missing_file", r["analysis_status"] == "missing_file")
    syn, n_true = _synthetic_he()
    sp = workdir / "synthetic.png"; Image.fromarray(syn).save(sp)
    r = analyze_case("TEST-NOMPP", sp, out, meta={"slide_barcode": "NOMPP"}, cfg=cfg, write_tables=False)
    check("MPP yok → failed_qc (MPP tahmin edilmez)", r["analysis_status"] == "failed_qc" and "MPP_MISSING" in r["qc_flags"])

    # --- Sentetik çekirdekler: bilinen alan geri elde ediliyor mu? ---
    r1 = analyze_case("TEST-SYN", sp, out, meta={"slide_barcode": "SYN"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    exp_area = math.pi * 3.0 ** 2
    med = r1.get("area_um2_median", float("nan"))
    check("Sentetik: analiz başarılı", r1["analysis_status"] in SUCCESS_STATUSES, f"({r1['analysis_status']} {r1['error_message']})")
    check("Sentetik: medyan alan ≈ π·3² µm² (±15%)", abs(med - exp_area) / exp_area < 0.15, f"(ölçülen {med:.1f}, beklenen {exp_area:.1f})")
    check("Sentetik: tespit edilen çekirdek oranı ≥%80 (kenar hariç)", r1.get("n_nuclei", 0) >= 0.8 * n_true * 0.85,
          f"({r1.get('n_nuclei')} / {n_true} yerleştirilen)")
    check("Sentetik: dairesellik medyanı > 0.85", r1.get("circularity_median", 0) > 0.85, f"({r1.get('circularity_median', float('nan')):.3f})")
    check("Sentetik: tek boyutlu çekirdeklerde alan entropisi düşük (<1.5 bit)", r1.get("area_um2_entropy", 9) < 1.5,
          f"({r1.get('area_um2_entropy', float('nan')):.3f})")
    r2 = analyze_case("TEST-SYN", sp, out, meta={"slide_barcode": "SYN"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    keys = [k for k in r1 if k.endswith("entropy") or k.endswith("_n500") or k == "n_nuclei"]
    check("Tekrarlanabilirlik: aynı girdi+parametre → birebir aynı sonuçlar", all(
        (r1[k] == r2[k]) or (isinstance(r1[k], float) and math.isnan(r1[k]) and math.isnan(r2[k])) for k in keys), f"({len(keys)} ölçüt)")

    # --- Heterojen boyutlar → daha yüksek alan entropisi ---
    het, _ = _synthetic_he(radii_um=(1.8, 6.5), seed=3, n_nuclei=300)
    hp = workdir / "het.png"; Image.fromarray(het).save(hp)
    r3 = analyze_case("TEST-HET", hp, out, meta={"slide_barcode": "HET"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    check("Heterojen boyut → alan entropisi homojenden yüksek", r3.get("area_um2_entropy", 0) > r1.get("area_um2_entropy", 9),
          f"({r3.get('area_um2_entropy', float('nan')):.2f} > {r1.get('area_um2_entropy', float('nan')):.2f})")

    # --- Odak kontrolü: aynı görüntünün bulanıklaştırılmış hali bulanık bloklar üretmeli ---
    import cv2
    tile = syn[:512, :512]
    g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    sharp = block_focus(g, np.ones(g.shape, bool), cfg)
    blur = block_focus(cv2.GaussianBlur(g, (0, 0), 2.0), np.ones(g.shape, bool), cfg)
    check("Odak: keskin görüntüde bulanık blok yok/az", sharp["blurry"][sharp["determinate"]].mean() < 0.2 if sharp["determinate"].any() else True,
          f"({sharp['blurry'].sum()}/{sharp['determinate'].sum()})")
    check("Odak: Gauss σ=2 bulanıklaştırma → blokların çoğu bulanık", blur["determinate"].any() and blur["blurry"][blur["determinate"]].mean() > 0.8,
          f"({blur['blurry'].sum()}/{blur['determinate'].sum()})")
    bimg = workdir / "blurred.png"
    Image.fromarray(cv2.GaussianBlur(syn, (0, 0), 2.5)).save(bimg)
    rb = analyze_case("TEST-BLUR", bimg, out, meta={"slide_barcode": "BLUR"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    check("Tamamen bulanık görüntü → failed_qc (OUT_OF_FOCUS)", "OUT_OF_FOCUS" in (rb.get("qc_flags") or ""), f"({rb['analysis_status']}, bulanık %{100 * (rb.get('focus_blurry_tissue_fraction') or 0):.0f})")

    # --- 2. aşama: entropi yeniden hesaplama segmentasyonu tekrarlamadan aynı sonucu vermeli ---
    rr = recompute_entropy(out / "cases" / "TEST-SYN", cfg)
    check("2. aşama: aynı parametre → aynı entropi (yeniden segmentasyon yok)",
          rr is not None and abs(rr["area_um2_entropy"] - r2["area_um2_entropy"]) < 1e-12)
    cfg2 = Config(**{**cfg.__dict__, "feature_bins": {**cfg.feature_bins, "area_um2": [0.0, 200.0, 80]}})
    rr2 = recompute_entropy(out / "cases" / "TEST-SYN", cfg2)
    check("2. aşama: bin değişince seg_hash aynı, entropy_hash farklı", cfg2.seg_hash() == cfg.seg_hash() and cfg2.entropy_hash() != cfg.entropy_hash())
    check("2. aşama: yeni bin'lerle entropi güncellendi", rr2["entropy_hash"] == cfg2.entropy_hash())
    check("Genel bakış görseli oluşturuldu", (out / "cases" / "TEST-SYN" / "qc_overview.png").exists() and any((out / "qc_gallery").glob("*SYN*.png")))

    # --- v2: H&E / boya taraması ---
    def screen_img(img):
        th_ = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
        m_, _ = compute_tissue_mask(th_, 2.0, cfg)
        return screen_stain(th_, m_, estimate_background_io(th_, m_, cfg), cfg)["stain_class"]
    check("Boya taraması: sentetik H&E → H&E sınıfı", screen_img(syn) in ("HE_TYPICAL", "HE_ATYPICAL"), f"({screen_img(syn)})")
    dab_img, _ = _synthetic_he(nucleus_stain="dab", seed=5)
    check("Boya taraması: sentetik DAB (IHC) → POSSIBLE_IHC_DAB", screen_img(dab_img) == "POSSIBLE_IHC_DAB", f"({screen_img(dab_img)})")
    hh_, ee_ = np.zeros(syn.shape[:2]), np.zeros(syn.shape[:2])
    ee_[:, :1400] = 0.6
    tri = _render_stains([(ee_, (0.80, 0.55, 0.20)), (hh_, (0.1, 0.9, 0.4))], seed=6)   # anilin mavisi benzeri
    check("Boya taraması: mavi (trikrom benzeri) → NOT_HE_LIKE", screen_img(tri) == "NOT_HE_LIKE", f"({screen_img(tri)})")
    _, _, lab_ = _synthetic_he(return_labels=True)
    faded = _render_he((lab_ > 0) * 0.18, np.where(np.arange(2048)[None, :] < 1433, 0.06, 0.0) * np.ones((2048, 1)), seed=7)
    check("Boya taraması: çok soluk H&E → HE_FADED", screen_img(faded) == "HE_FADED", f"({screen_img(faded)})")
    dp = workdir / "dab.png"; Image.fromarray(dab_img).save(dp)
    rd = analyze_case("TEST-DAB", dp, out, meta={"slide_barcode": "DAB"}, cfg=cfg, mpp_override=0.5, write_tables=False)
    check("IHC slaytı analiz edilmez (failed_qc + öneri)", rd["analysis_status"] == "failed_qc" and "STAIN_POSSIBLE_IHC_DAB" in rd["qc_flags"])

    # --- v2: çözünürlük koruması ve µm tutarlılığı ---
    rlow = analyze_case("TEST-LOWRES", sp, out, meta={"slide_barcode": "LOWRES"}, cfg=cfg, mpp_override=1.0, write_tables=False)
    check("Hedeften kaba slayt büyütülmez → RESOLUTION_INSUFFICIENT", "RESOLUTION_INSUFFICIENT" in (rlow.get("qc_flags") or ""))
    s25, _ = _synthetic_he(size=4096, mpp=0.25)
    p25 = workdir / "syn_025.png"; Image.fromarray(s25).save(p25)
    cfg25 = Config(**{**cfg.__dict__, "target_mpp": 0.25})
    r25 = analyze_case("TEST-SYN25", p25, out, meta={"slide_barcode": "SYN25"}, cfg=cfg25, mpp_override=0.25, write_tables=False)
    r25d = analyze_case("TEST-SYN25D", p25, out, meta={"slide_barcode": "SYN25D"}, cfg=cfg, mpp_override=0.25, write_tables=False)
    a50, a25, a25d = r1.get("area_um2_median", np.nan), r25.get("area_um2_median", np.nan), r25d.get("area_um2_median", np.nan)
    check("Aynı doku 0.25 µm/px'te → aynı fiziksel alan (±10%)", abs(a25 - exp_area) / exp_area < 0.10, f"({a25:.1f} vs {exp_area:.1f})")
    check("40x slayt 0.5'e küçültülünce 20x ile uyumlu (±10%)", abs(a25d - a50) / a50 < 0.10, f"({a25d:.1f} vs {a50:.1f})")

    # --- v2: QuPath içe aktarma (poligon → aynı ölçüm tanımları) ---
    from skimage.measure import find_contours
    feats = []
    for k in range(1, int(lab_.max()) + 1):
        ys_, xs_ = np.nonzero(lab_ == k)
        if not len(ys_):
            continue
        y0, x0 = ys_.min() - 2, xs_.min() - 2
        sub_ = np.pad(lab_[y0 + 2:ys_.max() + 1, x0 + 2:xs_.max() + 1] == k, 2)
        c = max(find_contours(sub_.astype(float), 0.5), key=len)
        ring = [[float(x0 + q[1]), float(y0 + q[0])] for q in c]
        ring.append(ring[0])
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"objectType": "detection", "classification": {"name": "Tumor" if k % 2 else "Lymphocyte"}}})
    gj = workdir / "qupath_detections.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    ri = import_qupath(out, "TEST-SYN", gj, cfg)
    check("QuPath içe aktarma: alan ≈ π·3² (±10%)", abs(ri["area_um2_median"] - exp_area) / exp_area < 0.10, f"({ri['area_um2_median']:.1f})")
    check("QuPath içe aktarma: çekirdek sayısı pipeline ile uyumlu (±15%)", abs(ri["n_nuclei"] - r2["n_nuclei"]) / r2["n_nuclei"] < 0.15,
          f"({ri['n_nuclei']} vs {r2['n_nuclei']})")
    cfgT = Config(**{**cfg.__dict__, "nucleus_classes_include": ("Tumor",)})
    rT = recompute_entropy(out / "cases" / "TEST-SYN", cfgT)
    check("Sınıf filtresi (yalnız Tumor) ≈ yarısı", 0.35 < rT["n_nuclei"] / ri["n_nuclei"] < 0.65, f"({rT['n_nuclei']}/{ri['n_nuclei']})")

    # --- v2: güncel Excel'de olmayan (eski) sonuçlar kohort tablosuna girmez ---
    write_validation(out, pd.DataFrame([{"excel_row": 2, "case_id": "TEST-SYN", "slide_barcode": "SYN", "slide_uuid": None}]),
                     pd.DataFrame(columns=["excel_row"]))
    agg_ = write_aggregate_tables(out)
    check("Excel'den çıkarılan slaytlar kohorttan ayrılır (aynı hasta 2 kez sayılmaz)",
          len(agg_) == 1 and (out / "stale_results_not_in_current_excel.csv").exists(), f"(aktif {len(agg_)})")

    # --- v2: StarDist (kuruluysa) ---
    import importlib.util
    has_sd = importlib.util.find_spec("stardist") is not None
    if has_sd:
        cfgS = Config(**{**cfg.__dict__, "segmenter": "stardist"})
        rs = analyze_case("TEST-SD", sp, out, meta={"slide_barcode": "SD"}, cfg=cfgS, mpp_override=0.5, write_tables=False)
        check("StarDist: sentetik çekirdek alanı ≈ π·3² (±20%)", rs["analysis_status"] in SUCCESS_STATUSES and
              abs(rs.get("area_um2_median", 0) - exp_area) / exp_area < 0.20, f"({rs['analysis_status']}, {rs.get('area_um2_median')})")
    else:
        print("[ATLA] StarDist kurulu değil → StarDist testi atlandı")

    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} test geçti. Test çıktıları: {workdir}")
    return n_fail == 0


# =============================================================================
# 13. KOMUT SATIRI
# =============================================================================
def setup_logging(output_dir: Path | None):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(sh)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "analysis_log.txt", mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)


def load_config(out: Path | None, config_json: Path | None, overrides: dict) -> Config:
    """Öncelik: komut satırı > --config-json > çıktı klasöründe kayıtlı analysis_config.json > varsayılan.
    Kayıtlı config'in otomatik kullanılması, aynı projeye sonradan eklenen vakaların AYNI parametrelerle
    analiz edilmesini garanti eder (aksi halde seg_hash değişir ve kohort karışır)."""
    base: dict = {}
    saved = out / "analysis_config.json" if out else None
    if config_json:
        base = json.loads(Path(config_json).read_text(encoding="utf-8"))
        base = base.get("config", base)
    elif saved and saved.exists():
        base = json.loads(saved.read_text(encoding="utf-8")).get("config", {})
        if base:
            log.info("Kayıtlı konfigürasyon kullanılıyor: %s", saved)
    cfg = config_from_dict(base)
    for k, v in overrides.items():
        if v is not None:
            if getattr(cfg, k) != v:
                log.info("Parametre değişikliği: %s = %r (önceki %r)", k, v, getattr(cfg, k))
            setattr(cfg, k, v)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="TCGA-KIRC nükleer morfometrik entropi pipeline'ı (v2.0)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make-manifest", "validate", "screen", "single", "batch", "entropy", "import-qupath",
                 "calibrate-focus", "summarize", "selftest"):
        p = sub.add_parser(name)
        if name != "selftest":
            p.add_argument("--out", required=True, type=Path, help="Çıktı klasörü")
            p.add_argument("--config-json", type=Path, help="Config alanlarını değiştiren JSON dosyası")
        if name == "make-manifest":
            p.add_argument("--excel", type=Path, required=True)
        if name in ("validate", "screen", "single", "batch"):
            p.add_argument("--excel", type=Path, help="TCGA_KIRC_200_Case.xlsx")
            p.add_argument("--data-dir", type=Path, help="SVS dosyalarının bulunduğu klasör (alt klasörler taranır)")
        if name in ("screen", "single", "batch"):
            p.add_argument("--target-mpp", type=float, help="Analiz çözünürlüğü µm/px (0.5 = 20x, 0.25 = 40x)")
        if name in ("single", "batch"):
            p.add_argument("--max-tiles", type=int, default=None, help="Vaka başına en fazla tile (varsayılan: tümü)")
            p.add_argument("--segmenter", choices=["classical", "stardist"], help="Çekirdek segmentasyon yöntemi")
        if name == "single":
            p.add_argument("--case", help="Excel'deki Case ID veya slide barcode")
            p.add_argument("--image", type=Path, help="Excel olmadan doğrudan görüntü yolu")
            p.add_argument("--case-id", help="--image ile birlikte hasta ID'si")
            p.add_argument("--mpp", type=float, help="Yalnızca metadata'sında MPP olmayan görüntüler için (biliniyorsa!)")
            p.add_argument("--roi-geojson", type=Path, help="Patolog ROI (QuPath GeoJSON)")
        if name == "batch":
            p.add_argument("--gdc-client", help="Dosyası olmayan vakaları indirmek için gdc-client yolu")
            p.add_argument("--delete-after", action="store_true", help="İndirilen slaytı analizden sonra sil (disk tasarrufu)")
            p.add_argument("--min-free-gb", type=float, default=5.0, help="Bu kadar boş alan kalmazsa güvenle dur")
            p.add_argument("--confirm", action="store_true", help="Toplu analizi başlatmak için ZORUNLU")
            p.add_argument("--force", action="store_true", help="Başarılı vakaları da yeniden hesapla")
            p.add_argument("--limit", type=int, help="İlk N satır (deneme için)")
        if name in ("import-qupath", "calibrate-focus"):
            p.add_argument("--slide-id", required=True, help="Slayt barcode'u (cases/ altındaki klasör adı)")
        if name == "import-qupath":
            p.add_argument("--geojson", required=True, type=Path, help="QuPath'ten dışa aktarılan detection/cell GeoJSON")
            p.add_argument("--classes", help="Entropiye girecek sınıflar, virgülle (ör. Tumor). Boş = hepsi")
        if name == "selftest":
            p.add_argument("--workdir", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        setup_logging(None); log.setLevel(logging.WARNING)
        sys.exit(0 if selftest(a.workdir) else 1)
    setup_logging(a.out)
    overrides = {"target_mpp": getattr(a, "target_mpp", None), "segmenter": getattr(a, "segmenter", None),
                 "max_tiles": getattr(a, "max_tiles", None)}
    if getattr(a, "classes", None):
        overrides["nucleus_classes_include"] = tuple(c.strip() for c in a.classes.split(",") if c.strip())
    cfg = load_config(a.out, a.config_json, overrides)

    if a.cmd == "make-manifest":
        make_manifest(a.excel, a.out)
    elif a.cmd == "validate":
        cases, issues = prepare_cases(a.excel, a.data_dir)
        write_validation(a.out, cases, issues)
        print(cases["prep_status"].value_counts().to_string())
        print(issues.groupby(["level", "code"]).size().to_string())
        print(f"\nAyrıntılar: {a.out/'excel_validation_issues.csv'}, {a.out/'case_file_mapping.csv'}")
    elif a.cmd == "screen":
        df = run_screen(a.excel, a.data_dir, a.out, cfg)
        if len(df):
            print(df["stain_class"].value_counts().to_string())
            print(f"\nAyrıntılar ve öneriler: {a.out / 'stain_screen' / 'stain_screen.xlsx'}")
    elif a.cmd == "single":
        write_run_metadata(a.out, cfg)
        if a.image:
            if not a.case_id:
                raise SystemExit("--image ile --case-id gerekli")
            r = analyze_case(a.case_id, a.image, a.out, cfg=cfg, roi_geojson=a.roi_geojson, mpp_override=a.mpp)
        else:
            if not (a.excel and a.data_dir and a.case):
                raise SystemExit("--excel, --data-dir ve --case gerekli (veya --image/--case-id)")
            r = run_single_from_excel(a.excel, a.data_dir, a.out, a.case, cfg, a.roi_geojson)
        show = ["case_id", "slide_id", "segmenter", "analysis_status", "quality_control_status", "qc_flags", "stain_class",
                "n_tiles_analyzed", "n_nuclei", "area_um2_entropy", "perimeter_um_entropy", "circularity_entropy",
                "eccentricity_entropy", "log2_area_entropy", "patch_nuc_area_entropy_p90", "error_message", "warnings",
                "runtime_sec"]
        for k in show:
            print(f"{k:28s}: {r.get(k)}")
    elif a.cmd == "batch":
        if not a.confirm:
            raise SystemExit("Toplu analiz için --confirm gerekli. Önce tek vaka sonuçlarını kontrol edin.")
        t = run_batch(a.excel, a.data_dir, a.out, cfg, force=a.force, limit=a.limit, gdc_client=a.gdc_client,
                      delete_after=a.delete_after, min_free_gb=a.min_free_gb)
        print(t["analysis_status"].value_counts(dropna=False).to_string())
    elif a.cmd == "entropy":
        t = recompute_all_entropies(a.out, cfg)
        print(f"{len(t)} vaka tabloya yazıldı → {a.out}")
    elif a.cmd == "import-qupath":
        r = import_qupath(a.out, a.slide_id, a.geojson, cfg)
        for k in ("n_nuclei", "area_um2_entropy", "perimeter_um_entropy", "circularity_entropy", "eccentricity_entropy"):
            print(f"{k:24s}: {r.get(k) if r else None}")
    elif a.cmd == "calibrate-focus":
        print(calibrate_focus(a.out, a.slide_id, cfg))
    elif a.cmd == "summarize":
        t = write_aggregate_tables(a.out)
        write_action_list(a.out)
        print(f"{len(t)} vaka yazıldı → {a.out}")


if __name__ == "__main__":
    main()
