#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TÜBİTAK 2209-A — Nükleer Entropi Projesi: TEK GİRİŞ NOKTASI
============================================================

Senin klasör yapını kullanır:

  TUBITAK_2209_Nuclear_Entropy/
  ├── 01_literature/
  ├── 02_raw_data/              ← Excel + wsi/ (slaytlar; indir-analiz et-sil modunda geçici)
  ├── 03_preprocessed_images/   ← QC galerisi, boya/normalizasyon görselleri, defter çıktıları
  ├── 04_segmentation/          ← segmentasyon örnekleri, doğrulama (annot/)
  ├── 05_morphometry/           ← vaka düzeyi morfometri özeti
  ├── 06_entropy/               ← entropi tablosu
  ├── 07_statistics/            ← istatistik raporu
  ├── 08_figures/               ← tüm figürler
  ├── 09_results/               ← pipeline/ (ham çıktılar) + özet tablolar
  ├── scripts/                  ← BU DOSYA ve diğer .py dosyaları
  └── proje_ayarlari.json

Komutlar (sırasıyla):
  python scripts/proje.py kurulum              # klasörler, ayar dosyası, kütüphane kontrolü
  python scripts/proje.py durum                # NEREDEYİM? sıradaki adım ne?
  python scripts/proje.py liste                # Excel kontrolü + GDC indirme listesi
  python scripts/proje.py pilot                # her grade'den 3 vaka (slaytlar saklanır)
  python scripts/proje.py dx-gecis             # Excel'in DX (FFPE) sürümü + 0.25 µm/px; TS/BS pilotu arşivlenir
  python scripts/proje.py dondurulmus-karsilastir   # aynı hastalarda dondurulmuş vs FFPE entropi
  python scripts/proje.py cozunurluk-duyarlilik     # pilot DX'leri 0.5 µm/px'te de analiz et (kesit ↔ çözünürlük etkisini ayır)
  python scripts/proje.py monuseg              # uzman etiketli MoNuSeg böbrek görüntüleriyle doğrulama
  python scripts/proje.py dogrulama-hazirla    # QuPath'te çizilecek tile'lar
  python scripts/proje.py dogrulama-degerlendir   # F1 / Dice / entropi hatası, yöntem önerisi
  python scripts/proje.py ayarla segmenter stardist   # (gerekirse) ayar değiştir
  python scripts/proje.py toplu --onay         # 200 vaka: indir → analiz → sil
  python scripts/proje.py yedek-sec            # 20x vakalar yerine aynı grade'den rastgele yedek (kurala göre)
  python scripts/proje.py istatistik           # testler + figürler + klasörlere dağıtım
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import kirc_entropy_pipeline as kp  # noqa: E402

ROOT = HERE.parent
FOLDERS = ["01_literature", "02_raw_data", "02_raw_data/wsi", "03_preprocessed_images", "04_segmentation",
           "05_morphometry", "06_entropy", "07_statistics", "08_figures", "09_results", "scripts"]
CFG_FILE = ROOT / "proje_ayarlari.json"
RESULTS = ROOT / "09_results" / "pipeline"
FROZEN_PILOT = ROOT / "09_results" / "pilot_dondurulmus_TS_BS"
DX05 = ROOT / "09_results" / "duyarlilik_DX_0.5"
INVENTORY = ROOT / "02_raw_data" / "gdc_slayt_envanteri.xlsx"
MPP_40X = 0.275          # ≤ bu değer: 40x tarama (~0.25 µm/px)
MPP_RULE_SHARE = 0.90    # önceden belirlenen kural: DX slaytların ≥%90'ı 40x ise 0.25 µm/px, değilse 0.5
ANNOT = ROOT / "04_segmentation" / "annot"
DEFAULTS = {
    "excel": "02_raw_data/TCGA_KIRC_200_Case.xlsx",
    "wsi_klasoru": "02_raw_data/wsi",
    "gdc_client": "scripts/gdc-client.exe" if os.name == "nt" else "scripts/gdc-client",
    "indirdikten_sonra_sil": True,
    "min_bos_alan_gb": 5,
    "gdc_paralel_baglanti": 8,
    "segmenter": "classical",
    "target_mpp": 0.5,
    "parametre_dosyasi": None,
    "max_tile": None,              # None = slayttaki tüm uygun tile'lar; sayı = slayt başına rastgele örneklem
    "pilot_grade_basina": 3,
    "dogrulama_vaka": 10,
    "dogrulama_tile_vaka_basina": 2,
}


def ayarlar() -> dict:
    a = dict(DEFAULTS)
    if CFG_FILE.exists():
        a.update(json.loads(CFG_FILE.read_text(encoding="utf-8")))
    return a


def yol(rel: str | None) -> Path | None:
    if not rel:
        return None
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def gdc_yolu(a) -> str | None:
    g = yol(a["gdc_client"])
    if g and g.exists():
        return str(g)
    w = shutil.which("gdc-client")
    return w


def pipeline_cfg(a) -> kp.Config:
    over = {"segmenter": a["segmenter"], "target_mpp": float(a["target_mpp"]),
            "max_tiles": int(a["max_tile"]) if a.get("max_tile") else None}
    return kp.load_config(RESULTS, yol(a.get("parametre_dosyasi")), over)


def baslik(t):
    print("\n" + "=" * 78 + f"\n  {t}\n" + "=" * 78)


# -----------------------------------------------------------------------------
def kurulum(test: bool):
    baslik("KURULUM KONTROLÜ")
    for f in FOLDERS:
        (ROOT / f).mkdir(parents=True, exist_ok=True)
    print(f"✓ Klasör yapısı hazır: {ROOT}")
    if not CFG_FILE.exists():
        CFG_FILE.write_text(json.dumps(DEFAULTS, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Ayar dosyası oluşturuldu: {CFG_FILE}")
    a = ayarlar()
    ok = True
    print(f"{'✓' if sys.version_info >= (3, 10) else '✗'} Python {sys.version.split()[0]} (3.10+ gerekli)")
    for mod in ("numpy", "pandas", "scipy", "skimage", "cv2", "matplotlib", "openpyxl", "PIL", "openslide",
                "statsmodels", "sklearn"):
        try:
            m = __import__(mod)
            print(f"✓ {mod:12s} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"✗ {mod:12s} YOK → pip install -r scripts/requirements.txt   ({exc})")
    print(f"{'✓' if _has('notebook') else '·'} notebook     (Adım 5 defteri için) {'kurulu' if _has('notebook') else 'yok → pip install notebook'}")
    try:
        __import__("stardist")
        print("✓ stardist     (isteğe bağlı) kurulu")
    except Exception:  # noqa: BLE001
        print("· stardist     (isteğe bağlı) kurulu değil — yalnız klasik yöntem kullanılabilir")
    ex = yol(a["excel"])
    print(f"{'✓' if ex.exists() else '✗'} Excel: {ex}" + ("" if ex.exists() else "  ← Excel'i buraya koyun"))
    g = gdc_yolu(a)
    print(f"{'✓' if g else '✗'} gdc-client: {g or yol(a['gdc_client'])}" +
          ("" if g else "  ← GDC Data Transfer Tool'u indirip scripts/ içine koyun"))
    ok &= ex.exists()
    if test:
        baslik("SELF-TEST (birkaç dakika)")
        r = os.system(f'"{sys.executable}" "{HERE / "kirc_entropy_pipeline.py"}" selftest')
        ok &= r == 0
    print("\n" + ("HAZIR. Sıradaki: python scripts/proje.py liste" if ok else "Eksikleri giderip tekrar çalıştırın."))


def liste():
    a = ayarlar()
    baslik("EXCEL KONTROLÜ + GDC İNDİRME LİSTESİ")
    kp.setup_logging(RESULTS)
    ex, wsi = yol(a["excel"]), yol(a["wsi_klasoru"])
    cases, issues = kp.prepare_cases(ex, wsi)
    kp.write_validation(RESULTS, cases, issues)
    print(cases["prep_status"].value_counts().to_string())
    imp = issues[issues["level"].isin(["error", "warning"])]
    if len(imp):
        print(f"\n⚠ {len(imp)} uyarı/hata → {RESULTS / 'excel_validation_issues.csv'}")
        print(imp[["excel_row", "case_id_excel", "code", "message"]].head(15).to_string(index=False))
    kinds = cases["slide_barcode"].astype(str).str.extract(r"-([A-Z]{2})[A-Z0-9]$")[0].value_counts().to_dict()
    print(f"\nSlayt türleri: {kinds}  (TS/BS = dondurulmuş kesit, DX = FFPE tanısal)")
    kp.make_manifest(ex, ROOT / "02_raw_data")
    print("\nNOT: Elle indirmen GEREKMEZ. `pilot` ve `toplu` komutları slaytları gdc-client ile tek tek indirir "
          "(toplu analizde analizden sonra siler). Manifest yalnızca toplam boyutu görmek / elle indirmek içindir.")


def pilot(vakalar: list[str] | None, sil: bool):
    a = ayarlar()
    kp.setup_logging(RESULTS)
    cfg = pipeline_cfg(a)
    ex, wsi = yol(a["excel"]), yol(a["wsi_klasoru"])
    if not vakalar:
        df, _ = kp.load_case_table(ex)
        df = df[~df["is_duplicate"] & ~df["manual_excluded"]]
        vakalar = []
        for g in ("G1", "G2", "G3", "G4"):
            vakalar += df[df["grade"] == g].sample(frac=1, random_state=2209)["case_id_excel"].head(int(a["pilot_grade_basina"])).tolist()
    (RESULTS / "pilot_vakalar.json").parent.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pilot_vakalar.json").write_text(json.dumps(vakalar, indent=1), encoding="utf-8")
    baslik(f"PİLOT: {len(vakalar)} vaka ({', '.join(vakalar)})")
    print("Slaytlar doğrulama tile'ları için SAKLANIR (silinmez)." if not sil else "Slaytlar analizden sonra silinecek.")
    kp.run_batch(ex, wsi, RESULTS, cfg, gdc_client=gdc_yolu(a), delete_after=sil,
                 min_free_gb=float(a["min_bos_alan_gb"]), only_case_ids=vakalar,
                 gdc_processes=int(a["gdc_paralel_baglanti"]))
    for sid in _pilot_slaytlari():
        try:
            if (wsi.exists() and _slide_present(sid)):
                kp.calibrate_focus(RESULTS, sid, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  odak kalibrasyonu yapılamadı ({sid}): {exc}")
    topla()
    karar = mpp_kurali(RESULTS)
    if karar is not None and abs(karar - float(a["target_mpp"])) > 1e-9:
        print(f"⚠ ÖNCEDEN BELİRLENEN KURAL: kohort {karar} µm/px'te analiz edilmeli → "
              f"python scripts/proje.py ayarla target_mpp {karar}  ve pilotu tekrar çalıştırın")
    if FROZEN_PILOT.exists():
        print("Dondurulmuş-FFPE karşılaştırması için: python scripts/proje.py dondurulmus-karsilastir")
    print("\nKONTROL ET: 04_segmentation/ornekler/ ve 03_preprocessed_images/odak_kalibrasyonu/ içindeki görseller "
          "(bkz. YOL_HARITASI.md, Adım 5). Sonra: python scripts/proje.py dogrulama-hazirla")


def _pilot_slaytlari() -> list[str]:
    p = RESULTS / "pilot_vakalar.json"
    if not p.exists():
        return []
    ids = set(json.loads(p.read_text(encoding="utf-8")))
    out = []
    for rp in (RESULTS / "cases").glob("*/case_result.json"):
        r = json.loads(rp.read_text(encoding="utf-8"))
        if r.get("case_id") in ids and r.get("analysis_status") in kp.SUCCESS_STATUSES:
            out.append(rp.parent.name)
    return sorted(out)


def _slide_present(sid: str) -> bool:
    rp = RESULTS / "cases" / sid / "case_result.json"
    if not rp.exists():
        return False
    ip = json.loads(rp.read_text(encoding="utf-8")).get("image_path")
    return bool(ip) and Path(ip).exists()


def dx_gecis():
    """Önceden belirlenmiş kurala göre her hastanın DX slaytını seçer, DX Excel'ini yazar, ayarları günceller,
    dondurulmuş (TS/BS) pilot sonuçlarını karşılaştırma için arşivler. Orijinal Excel'e dokunulmaz."""
    a = ayarlar()
    baslik("DX (FFPE TANISAL) SLAYTLARA GEÇİŞ")
    ex = yol(a["excel"])
    if ex.stem.endswith("_DX"):
        raise SystemExit(f"Ayarlardaki Excel zaten DX sürümü: {ex.name}")
    if not INVENTORY.exists():
        print("GDC envanteri yok, oluşturuluyor (internet gerekir) ...")
        kp.make_manifest(ex, ROOT / "02_raw_data")
    if not INVENTORY.exists():
        raise SystemExit("GDC envanteri oluşturulamadı; internet bağlantısını kontrol edip tekrar deneyin.")
    dst = ex.with_name(ex.stem + "_DX.xlsx")
    sel = kp.make_dx_excel(ex, INVENTORY, dst)
    print(kp.DX_RULE)
    print("\n" + sel["durum"].value_counts().to_string())
    multi = sel[sel["n_DX"] > 1]
    print(f"\nBirden fazla DX'i olan hasta: {len(multi)} (seçilmeyenler DX_gecis sayfasında 'diger_DX' sütununda)")
    print(f"Seçilen DX slaytların toplam boyutu: {sel['dx_size_MB'].sum() / 1e3:.1f} GB "
          f"(ortalama {sel['dx_size_MB'].mean():.0f} MB, en büyük {sel['dx_size_MB'].max():.0f} MB)")
    print(f"✓ Yazıldı: {dst}   (orijinal {ex.name} değişmedi)")

    # Dondurulmuş pilotu arşivle (karşılaştırma için); slaytlarını sil (sonuçlar korunur)
    if (RESULTS / "cases").exists() and not FROZEN_PILOT.exists():
        shutil.move(str(RESULTS), str(FROZEN_PILOT))
        print(f"✓ TS/BS pilot sonuçları arşivlendi: {FROZEN_PILOT}")
        freed = 0
        for rp in (FROZEN_PILOT / "cases").glob("*/case_result.json"):
            ip = json.loads(rp.read_text(encoding="utf-8")).get("image_path")
            if ip and Path(ip).exists() and yol(a["wsi_klasoru"]) in Path(ip).parents:
                d = Path(ip).parent
                freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
        if freed:
            print(f"✓ Dondurulmuş pilot slaytları silindi ({freed / 1e9:.1f} GB açıldı; sonuçlar arşivde duruyor)")
    elif FROZEN_PILOT.exists():
        print(f"· Dondurulmuş pilot arşivi zaten var: {FROZEN_PILOT}")
    for k, v in (("excel", str(dst.relative_to(ROOT)) if ROOT in dst.parents else str(dst)), ("target_mpp", 0.25)):
        ayarla(k, str(v))
    print("\nSIRADAKİ: caffeinate -i python scripts/proje.py pilot     (aynı 12 hasta, bu kez DX slaytlarıyla)")


def _mpp_dagilimi(results: Path) -> pd.DataFrame | None:
    rows = []
    for rp in (results / "cases").glob("*/case_result.json"):
        r = json.loads(rp.read_text(encoding="utf-8"))
        m = r.get("image_resolution_or_mpp")
        if m:
            rows.append({"slide_id": rp.parent.name, "grade": r.get("grade"), "mpp": float(m),
                         "dx": "-DX" in rp.parent.name})
    return pd.DataFrame(rows) if rows else None


def mpp_kurali(results: Path = RESULTS, yazdir: bool = True) -> float | None:
    """Önceden belirlenmiş çözünürlük kuralı: DX slaytların ≥%90'ı 40x (≤0.275 µm/px) → 0.25, değilse 0.5."""
    d = _mpp_dagilimi(results)
    if d is None or not d["dx"].any():
        return None
    d = d[d["dx"]]
    share = float((d["mpp"] <= MPP_40X).mean())
    karar = 0.25 if share >= MPP_RULE_SHARE else 0.5
    if yazdir:
        print(f"Tarama çözünürlüğü (DX, n={len(d)}): 40x {int((d['mpp'] <= MPP_40X).sum())} · 20x {int((d['mpp'] > MPP_40X).sum())} "
              f"→ 40x oranı %{100 * share:.0f} → kural kararı: {karar} µm/px")
    return karar


def _basarili(d: Path) -> pd.DataFrame:
    p = d / "case_level_results.csv"
    if not p.exists():
        raise SystemExit(f"{p} yok")
    c = pd.read_csv(p)
    return c[c["analysis_status"].isin(kp.SUCCESS_STATUSES)].drop_duplicates("case_id").set_index("case_id")


def _eslestirilmis(da: Path, db: Path, ad_a: str, ad_b: str, out: Path, ad: str) -> pd.DataFrame:
    """İki sonuç klasöründeki aynı hastaları eşleştirip karşılaştırır (b − a). Tablo + saçılım grafiği."""
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fa, fb = _basarili(da), _basarili(db)
    ids = sorted(set(fa.index) & set(fb.index))
    if len(ids) < 4:
        raise SystemExit(f"{ad}: iki tarafta da başarılı ortak hasta {len(ids)} (<4).")
    cols = [f"{f}_entropy" for f in kp.MORPH_FEATURES if f"{f}_entropy" in fa and f"{f}_entropy" in fb]
    cols += [c for c in ("area_um2_median",) if c in fa and c in fb]
    rows = []
    fig, axs = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.8), dpi=150)
    for ax, c in zip(np.atleast_1d(axs), cols):
        x, y = fa.loc[ids, c].astype(float), fb.loc[ids, c].astype(float)
        rho, p_rho = stats.spearmanr(x, y)
        w = stats.wilcoxon(x, y) if len(ids) >= 6 else None
        rows.append({"karşılaştırma": f"{ad}  ({ad_a} → {ad_b})", "ölçüt": c, "n": len(ids),
                     "sol_medyan": x.median(), "sag_medyan": y.median(),
                     "fark_medyan (sağ−sol)": (y - x).median(), "spearman_rho": rho, "p_spearman": p_rho,
                     "p_wilcoxon": None if w is None else w.pvalue})
        g = fa.loc[ids, "grade"].astype(str)
        for gr, col in zip(("G1", "G2", "G3", "G4"), ("#4c78a8", "#72b7b2", "#f58518", "#e45756")):
            m = g == gr
            ax.scatter(x[m], y[m], s=28, color=col, label=gr)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--")
        ax.set(xlabel=ad_a, ylabel=ad_b, title=f"{c}\nρ={rho:.2f}")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle(ad, fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{ad}.png"); plt.close(fig)
    return pd.DataFrame(rows)


def dondurulmus_karsilastir():
    """Aynı hastalarda kesit türü ve çözünürlük etkisi (eşleştirilmiş, keşifsel).
    Varsa 0.5 µm/px DX duyarlılık sonuçlarıyla üç karşılaştırma yapılır."""
    out = ROOT / "07_statistics" / "dondurulmus_vs_ffpe"
    out.mkdir(parents=True, exist_ok=True)
    comps = [(FROZEN_PILOT, RESULTS, "dondurulmus_0.5", "FFPE_0.25", "A_birlikte_kesit+cozunurluk")]
    if (DX05 / "case_level_results.csv").exists():
        comps += [(FROZEN_PILOT, DX05, "dondurulmus_0.5", "FFPE_0.5", "B_saf_kesit_turu_etkisi"),
                  (DX05, RESULTS, "FFPE_0.5", "FFPE_0.25", "C_saf_cozunurluk_etkisi")]
    else:
        print("Not: 0.5 µm/px DX sonuçları yok → yalnız birleşik karşılaştırma. Ayırmak için: "
              "python scripts/proje.py cozunurluk-duyarlilik")
    baslik("KESİT TÜRÜ / ÇÖZÜNÜRLÜK KARŞILAŞTIRMASI (eşleştirilmiş, keşifsel)")
    t = pd.concat([_eslestirilmis(c[0], c[1], c[2], c[3], out, c[4])
                   for c in comps], ignore_index=True)
    with pd.ExcelWriter(out / "dondurulmus_vs_ffpe.xlsx", engine="openpyxl") as xw:
        t.to_excel(xw, sheet_name="ozet", index=False)
    for ad, g in t.groupby("karşılaştırma", sort=False):
        print(f"\n— {ad} —")
        print(g.drop(columns=["karşılaştırma"]).dropna(axis=1, how="all").round(3).to_string(index=False))
    print("\nOkuma: fark_medyan = sağdaki − soldaki (bit). p_wilcoxon < 0.05 → sistematik kayma; "
          "ρ yüksekse hastaların SIRALAMASI korunuyor (grade karşılaştırması için asıl önemli olan).")
    print(f"✓ {out}")


def cozunurluk_duyarlilik():
    """Pilot DX slaytlarını (diskte duranlar) 0.5 µm/px'te AYRI klasöre analiz eder; ana sonuçlara dokunmaz."""
    a = ayarlar()
    if not (RESULTS / "analysis_config.json").exists():
        raise SystemExit("Önce DX pilotunu çalıştırın.")
    ids = json.loads((RESULTS / "pilot_vakalar.json").read_text(encoding="utf-8"))
    kp.setup_logging(DX05)
    cfg = kp.load_config(DX05, RESULTS / "analysis_config.json", {"target_mpp": 0.5})
    baslik(f"ÇÖZÜNÜRLÜK DUYARLILIĞI: {len(ids)} pilot DX slaytı 0.5 µm/px'te → {DX05.relative_to(ROOT)}")
    print("Yalnız diskteki slaytlar kullanılır (indirme yok). Ana 0.25 µm/px sonuçları DEĞİŞMEZ.")
    kp.run_batch(yol(a["excel"]), yol(a["wsi_klasoru"]), DX05, cfg, gdc_client=None, delete_after=False,
                 only_case_ids=ids)
    dondurulmus_karsilastir()


def tile_incele(vaka: str, n: int = 6):
    """Bir vakanın EN YÜKSEK ve EN DÜŞÜK alan entropili tile'larını slayttan okuyup yan yana gösterir.
    Amaç: yüksek entropinin gerçek doku heterojenliğinden mi, artefakttan (kalem, kenar, kan, ezilme) mı geldiğini görmek."""
    import cv2
    cand = [d for d in (RESULTS / "cases").glob("*") if d.name.startswith(vaka) or vaka in d.name]
    if not cand:
        raise SystemExit(f"{vaka} için sonuç klasörü yok ({RESULTS / 'cases'})")
    cd = cand[0]
    r = json.loads((cd / "case_result.json").read_text(encoding="utf-8"))
    ip = r.get("image_path")
    gecici = None
    if not ip or not Path(ip).exists():
        a = ayarlar()
        g, fid = gdc_yolu(a), r.get("file_id")
        if not (g and fid):
            raise SystemExit(f"Slayt diskte yok ve indirilemiyor: {ip}")
        print(f"Slayt diskte yok → geçici olarak indiriliyor ({fid}) ...")
        path = kp.gdc_download(str(fid), yol(a["wsi_klasoru"]), g, n_processes=int(a["gdc_paralel_baglanti"]))
        if path is None:
            raise SystemExit("İndirme başarısız; tekrar deneyin.")
        ip, gecici = str(path), Path(path).parent
    tm = pd.read_csv(cd / "tile_metrics.csv")
    tm = tm[(tm["n_nuclei"] >= 30) & tm["nuc_area_entropy"].notna()].sort_values("nuc_area_entropy")
    src = kp.open_image(ip)
    def row(sub, renk):
        out = []
        for t in sub.itertuples():
            rgb = src.read_region_l0(int(t.x_l0), int(t.y_l0), int(t.tile_l0), int(t.tile_l0))
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue, sat = hsv[..., 0] / 180, hsv[..., 1] / 255
            kalem = float(((hue > 0.15) & (hue < 0.60) & (sat > 0.25)).mean())
            small = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
            out.append((small, f"H={t.nuc_area_entropy:.2f}  n={int(t.n_nuclei)}  kalem %{100 * kalem:.0f}", renk))
        return out
    rows = [row(tm.tail(n).iloc[::-1], (200, 30, 30)), row(tm.head(n), (30, 90, 200))]
    dst = ROOT / "03_preprocessed_images" / "tile_inceleme" / f"{cd.name}_uc_tilelar.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    kp.pixel_montage(rows, dst, title=f"{cd.name} — üst satır: EN YÜKSEK alan entropili {n} tile (kırmızı), alt satır: EN DÜŞÜK "
                                       f"(mavi). Her tile {int(tm['tile_l0'].iloc[0]) * (src.mpp or 0):.0f} µm, 512 px'e küçültüldü",
                     font_size=14)
    src.close()
    if gecici is not None:
        shutil.rmtree(gecici, ignore_errors=True)
        print("Geçici slayt silindi.")
    print(f"✓ {dst}")


def hiz_testi(n: int = 12):
    """Pilot DX slaytlarından n tile üzerinde klasik ve StarDist süresini ölçer; 200 vakalık süreyi tahmin eder."""
    import time as _t
    import kirc_validate as kv
    a = ayarlar()
    cfg = pipeline_cfg(a)
    sids = [s_ for s_ in _pilot_slaytlari() if _slide_present(s_)]
    if not sids:
        raise SystemExit("Diskte pilot slaytı yok.")
    segs = ["classical"] + (["stardist"] if _has("stardist") else [])
    baslik(f"HIZ TESTİ: {len(sids[:3])} slayt × {n // 3} tile · {', '.join(segs)}")
    sure = {s_: [] for s_ in segs}
    n_tiles = []
    for sid in sids:
        r = json.loads((RESULTS / "cases" / sid / "case_result.json").read_text(encoding="utf-8"))
        n_tiles.append(r.get("n_tiles_analyzed") or 0)
    for sid in sids[:3]:
        cd = RESULTS / "cases" / sid
        r = json.loads((cd / "case_result.json").read_text(encoding="utf-8"))
        tm = pd.read_csv(cd / "tile_metrics.csv")
        tm = tm[tm["selected"].astype(bool)].sample(n=min(n // 3, int(tm["selected"].sum())), random_state=1)
        src = kp.open_image(r["image_path"])
        for t in tm.itertuples():
            rgb = kp.read_tile(src, int(t.x_l0), int(t.y_l0), int(t.tile_l0), cfg.tile_size_px)
            for s_ in segs:
                c = kp.Config(**{**cfg.__dict__, "segmenter": s_})
                ctx = kv.load_case_ctx(RESULTS, sid, c, rgb, r["effective_mpp"])
                h, _ = kp.deconvolve(rgb, ctx["io"], ctx["pinv"], ctx["scale"])
                if s_ == "stardist" and not sure[s_]:
                    kp.segment_nuclei(rgb, h, np.ones(h.shape, bool), ctx, c)   # model yükleme (ısınma) sayılmaz
                t0 = _t.time()
                kp.segment_nuclei(rgb, h, np.ones(h.shape, bool), ctx, c)
                sure[s_].append(_t.time() - t0)
        src.close()
    med_tiles = float(np.median(n_tiles))
    rows = []
    for s_ in segs:
        ek = float(np.median(sure[s_]))
        rows.append({"yöntem": s_, "sn/tile (yalnız segmentasyon)": round(ek, 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    # Pilot süresinden segmentasyon dışı payı: klasik pilot süresi − klasik segmentasyon
    pilot_sec = []
    for sid in sids:
        r = json.loads((RESULTS / "cases" / sid / "case_result.json").read_text(encoding="utf-8"))
        if r.get("runtime_sec"):
            pilot_sec.append(float(r["runtime_sec"]))
    base = float(np.median(pilot_sec)) if pilot_sec else 600.0
    cl = float(np.median(sure["classical"]))
    print(f"\nPilot: slayt başına medyan {med_tiles:.0f} tile, medyan süre {base / 60:.1f} dk (klasik)")
    for s_ in segs:
        ek = float(np.median(sure[s_])) - cl
        for mt in (None, 1000, 600, 400):
            tiles = med_tiles if mt is None else min(med_tiles, mt)
            per = (base * tiles / med_tiles) + ek * tiles
            print(f"  {s_:9s} max_tile={str(mt):5s} → vaka başına ~{per / 60:5.1f} dk · 200 vaka ~{per * 200 / 3600:5.1f} saat")


YEDEK_KAYDI = "yedek_kaydi.json"
YEDEK_SEED = 2209


def _gdc_kirc_grades(query=None) -> dict:
    """GDC: TCGA-KIRC hastalarının tümör grade'i (diagnoses.tumor_grade). Döner: {case_id: grade|None}."""
    import urllib.request
    if query is not None:
        return query()
    body = json.dumps({"filters": {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-KIRC"}},
                       "fields": "submitter_id,diagnoses.tumor_grade", "format": "JSON", "size": "2000"}).encode()
    req = urllib.request.Request(f"{kp.GDC_API}/cases", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        hits = json.loads(r.read().decode())["data"]["hits"]
    out = {}
    for h in hits:
        gs = {d.get("tumor_grade") for d in (h.get("diagnoses") or []) if d.get("tumor_grade") in ("G1", "G2", "G3", "G4")}
        out[h["submitter_id"]] = gs.pop() if len(gs) == 1 else None
    return out


def _gdc_kirc_dx(query=kp._gdc_files_query) -> pd.DataFrame:
    hits = query({"op": "and", "content": [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-KIRC"}},
        {"op": "=", "content": {"field": "data_format", "value": "SVS"}}]})
    rows = [{"file_id": h.get("file_id") or h.get("id"), "file_name": h.get("file_name"), "file_size": h.get("file_size"),
             "state": h.get("state"), "case_id": kp._hit_case(h)} for h in hits]
    d = pd.DataFrame(rows)
    d = d[d["file_name"].astype(str).str.contains(r"-DX[0-9A-Z]*\.", regex=True) & (d["state"] == "released")]
    return d.sort_values(["case_id", "file_size", "file_name"], ascending=[True, False, True])


def yedek_sec(grades_query=None, files_query=kp._gdc_files_query, downloader=None):
    """20x tarandığı için analiz edilemeyen vakaların yerine, AYNI grade'den, kohortta olmayan TCGA-KIRC hastaları
    arasından ÖNCEDEN BELİRLENMİŞ kuralla yedek seçer: GDC grade'i, rastgele sıra (seed 2209), ilk 40x DX slayt.
    Seçim entropi sonuçlarına bakılmadan yapılır. Orijinal Excel değişmez; yeni kopya yazılır."""
    from openpyxl import load_workbook
    from openpyxl.styles import PatternFill
    a = ayarlar()
    ex, wsi = yol(a["excel"]), yol(a["wsi_klasoru"])
    g = gdc_yolu(a) if downloader is None else "test"
    if not g:
        raise SystemExit("gdc-client bulunamadı.")
    cl = pd.read_csv(RESULTS / "case_level_results.csv")
    bad = cl[(cl["analysis_status"] == "failed_qc") & cl["qc_flags"].astype(str).str.contains("RESOLUTION_INSUFFICIENT")]
    if bad.empty:
        print("20x nedeniyle dışlanan vaka yok; yedek gerekmez.")
        return
    df, _ = kp.load_case_table(ex)
    need = bad["grade"].value_counts().sort_index().to_dict()
    baslik(f"YEDEK VAKA SEÇİMİ — gereken: {', '.join(f'{k} {v}' for k, v in need.items())}")
    print(f"Kural: kohortta olmayan TCGA-KIRC hastaları · GDC grade'i aynı · rastgele sıra (seed {YEDEK_SEED}) · "
          f"en büyük DX slayt · 40x (≤{MPP_40X} µm/px) olan ilk aday. Entropi sonuçlarına bakılmaz.")
    grades = _gdc_kirc_grades(grades_query)
    comp = df[df["case_id_excel"].isin(grades)]
    agree = (comp["grade"] == comp["case_id_excel"].map(grades)).mean() if len(comp) else float("nan")
    print(f"Excel grade'i ile GDC grade'i uyumu (mevcut kohort): %{100 * agree:.0f} ({len(comp)} hasta)")
    dx = _gdc_kirc_dx(files_query)
    in_excel = set(df["case_id_excel"].dropna())
    used = set()
    chosen, rejected = [], []
    for gr, n in need.items():
        pool = sorted(c for c, v in grades.items() if v == gr and c not in in_excel and c in set(dx["case_id"]))
        order = list(np.random.default_rng(YEDEK_SEED + int(gr[1])).permutation(pool))
        print(f"\n{gr}: {len(pool)} aday, {n} yedek gerekiyor")
        for cid in order:
            if sum(1 for c in chosen if c["grade"] == gr) >= n:
                break
            f = dx[dx["case_id"] == cid].iloc[0]
            print(f"  aday {cid} → {f['file_name']} ({f['file_size'] / 1e6:.0f} MB) indiriliyor ...")
            path = (downloader or (lambda fid: kp.gdc_download(fid, wsi, g, n_processes=int(a["gdc_paralel_baglanti"]))))(f["file_id"])
            if path is None:
                rejected.append({"case_id": cid, "grade": gr, "neden": "indirilemedi"})
                print("    indirilemedi → sıradaki aday")
                continue
            try:
                src = kp.open_image(path)
                mpp = src.mpp
                src.close()
            except Exception as exc:  # noqa: BLE001
                mpp = None
                print(f"    açılamadı: {exc}")
            if mpp is None or mpp > MPP_40X:
                rejected.append({"case_id": cid, "grade": gr, "neden": f"çözünürlük {mpp}"})
                print(f"    {mpp} µm/px (20x/bilinmiyor) → reddedildi, siliniyor")
                shutil.rmtree(Path(path).parent, ignore_errors=True)
                continue
            bc = Path(path).name.split(".")
            chosen.append({"case_id": cid, "grade": gr, "file_id": f["file_id"], "submitter_id": bc[0],
                           "slide_uuid": bc[1].lower() if len(bc) > 2 else None, "mpp": mpp,
                           "size_MB": round(float(f["file_size"]) / 1e6, 1)})
            print(f"    ✓ {mpp:.3f} µm/px → seçildi")
    missing = {gr: n - sum(1 for c in chosen if c["grade"] == gr) for gr, n in need.items()}
    if any(v > 0 for v in missing.values()):
        print(f"⚠ Yeterli aday bulunamadı: {missing}")

    # Yeni Excel kopyası: dışlananlar 'Selected? = No' + gerekçe; yedekler sona eklenir
    base = ex.stem.replace("_YEDEKLI", "")
    dst = ex.with_name(base + "_YEDEKLI.xlsx")
    wb = load_workbook(ex)
    ws = wb["Case_Selection"]
    head = {str(c.value).strip(): c.column for c in ws[1] if c.value}
    yellow, green = PatternFill("solid", fgColor="FFF2CC"), PatternFill("solid", fgColor="E2EFDA")
    out_ids = set(bad["case_id"])
    for r in range(2, ws.max_row + 1):
        if str(ws.cell(row=r, column=head["Case ID"]).value or "").strip() in out_ids:
            ws.cell(row=r, column=head["Selected?"], value="No").fill = yellow
            if "Exclusion Reason" in head:
                ws.cell(row=r, column=head["Exclusion Reason"], value="20x tarama (0.25 µm/px için yetersiz çözünürlük) — yedekle değiştirildi").fill = yellow
    no = max([int(v) for v in (ws.cell(row=r, column=head["No."]).value for r in range(2, ws.max_row + 1))
              if str(v).isdigit()] or [ws.max_row - 1])
    repl = list(bad.sort_values("grade")["case_id"])
    for k, c in enumerate(chosen):
        r = ws.max_row + 1
        vals = {"No.": no + k + 1, "Case ID": c["case_id"], "Grade": c["grade"], "Selected?": "Yes",
                "Selected Slide ID": c["slide_uuid"], "File ID": c["file_id"], "Submitter ID": c["submitter_id"],
                "Exclusion Reason": "_", "DX dosya boyutu (MB)": c["size_MB"],
                "Notes": f"YEDEK (aynı grade, rastgele seed {YEDEK_SEED}, GDC grade) — 20x vaka yerine"}
        for k_, v in vals.items():
            if k_ in head:
                ws.cell(row=r, column=head[k_], value=v).fill = green
    wb.save(dst)
    rec = {"tarih": pd.Timestamp.now().isoformat(timespec="seconds"), "kural": "aynı grade (GDC diagnoses.tumor_grade), kohort dışı TCGA-KIRC, "
           f"rastgele sıra seed {YEDEK_SEED}+grade, en büyük DX, ilk 40x aday", "dislanan": bad[["case_id", "grade"]].to_dict("records"),
           "yedekler": chosen, "reddedilen_adaylar": rejected, "excel_gdc_grade_uyumu": None if agree != agree else float(agree),
           "excel": str(dst.name)}
    (RESULTS / YEDEK_KAYDI).write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n✓ {len(chosen)} yedek seçildi; reddedilen aday {len(rejected)}. Excel: {dst.name} (orijinal değişmedi)")
    print(pd.DataFrame(chosen)[["case_id", "grade", "submitter_id", "mpp"]].to_string(index=False) if chosen else "")
    ayarla("excel", str(dst.relative_to(ROOT)) if ROOT in dst.parents else str(dst))
    print("\nSIRADAKİ: caffeinate -i python scripts/proje.py toplu --onay   (yalnız yedekler analiz edilir, ~1 saat)")


def stardist_ayar():
    """MoNuSeg böbrek görüntülerinde StarDist karar eşiklerini ayarlar (hasta bazında ayar/test bölmesi)."""
    import kirc_validate as kv
    if not _has("stardist"):
        raise SystemExit("StarDist kurulu değil.")
    if not (MONUSEG_DIR / "tiles_manifest.csv").exists():
        raise SystemExit("Önce: python scripts/proje.py monuseg")
    a = ayarlar()
    cfg = pipeline_cfg(a)
    baslik("StarDist EŞİK AYARI — MoNuSeg böbrek (24 kombinasyon, ~15–30 dk)")
    t = kv.tune_stardist(MONUSEG_DIR, None, cfg)
    b, d = t["best_row"], t["default_row"]
    print(f"\nAyar hastaları: {', '.join(t['tune_cases'])}\nTest hastaları: {', '.join(t['test_cases'])}  (seçimde kullanılmadı)")
    tab = pd.DataFrame([
        {"ayar": "varsayılan (prob 0.69, nms 0.30)", "test F1": d["test_F1"], "test Dice": d["test_dice"],
         "test PQ": d["test_PQ"], "test entropi hatası (bit)": d["test_entropy_MAE"]},
        {"ayar": f"en iyi (prob {b['prob_thresh']:.2f}, nms {b['nms_thresh']:.2f})", "test F1": b["test_F1"],
         "test Dice": b["test_dice"], "test PQ": b["test_PQ"], "test entropi hatası (bit)": b["test_entropy_MAE"]}])
    print("\n" + tab.round(3).to_string(index=False))
    gain = b["test_F1"] - d["test_F1"]
    worse_ent = b["test_entropy_MAE"] > d["test_entropy_MAE"] + 0.02
    if gain >= 0.03 and not worse_ent:
        print(f"\nKARAR: test F1 +{gain:.3f} → ANLAMLI KAZANÇ. 200 vakayı yeni eşiklerle yeniden analiz etmek önerilir:\n"
              f"  python scripts/proje.py ayarla parametre_dosyasi 04_segmentation/monuseg_bobrek/evaluation/stardist_best_params.json\n"
              f"  caffeinate -i python scripts/proje.py toplu --onay      (~1,5 gün)")
    else:
        print(f"\nKARAR: test F1 farkı {gain:+.3f} (eşik +0.03) → anlamlı kazanç YOK. Varsayılan eşiklerle devam; "
              "yeniden analiz gerekmez. Raporda: 'eşik ayarı denendi, bağımsız test hastalarında anlamlı iyileşme sağlamadı'.")
    print(f"Ayrıntı: {MONUSEG_DIR / 'evaluation' / 'stardist_tuning_grid.csv'}")


def yogunluk_duyarlilik(esikler=(100, 150), cikar=("TCGA-CZ-5455",)):
    """Duyarlılık: hücreden fakir tile'lar (kanama/nekroz/fibrin) dışlanınca bulgular değişiyor mu?
    Kayıtlı çekirdek tablolarından yeniden hesaplanır (slayt indirilmez). Ayrıca uç vaka(lar) çıkarılarak tekrar."""
    import kirc_stats as ks
    a = ayarlar()
    cfg = pipeline_cfg(a)
    ana, _, _ = ks.load_analysis_set(RESULTS, yol(a["excel"]))
    baslik(f"DUYARLILIK — hücreden fakir tile dışlama (eşik: {', '.join(map(str, esikler))} çekirdek/tile) + uç vaka")
    rows = []
    for k, r in enumerate(ana.itertuples(), 1):
        p = RESULTS / "cases" / str(r.slide_id) / "nuclei_features.csv.gz"
        if not p.exists():
            continue
        nuc = pd.read_csv(p)
        if "in_focus" in nuc and cfg.focus_exclude:
            nuc = nuc[nuc["in_focus"].astype(bool)]
        cnt = nuc.groupby("tile_id").size()
        rec = {"case_id": r.case_id, "n_tile": len(cnt)}
        for t in esikler:
            keep = cnt[cnt >= t].index
            sub = nuc[nuc["tile_id"].isin(keep)]
            rec[f"dislanan_tile_orani_{t}"] = 1 - len(keep) / max(len(cnt), 1)
            if len(sub) >= cfg.min_nuclei_case:
                e = kp._entropy_block(sub, cfg, suffix=f"_d{t}", full=False)
                rec.update({k_: v for k_, v in e.items() if any(k_.startswith(f) for f in kp.MORPH_FEATURES)})
        rows.append(rec)
        if k % 25 == 0:
            print(f"  {k}/{len(ana)} vaka")
    d = ana.merge(pd.DataFrame(rows), on="case_id", how="left")
    rng = np.random.default_rng(ks.SEED)
    out = []
    def ekle(df, etiket, suffix):
        for f in kp.MORPH_FEATURES:
            c = f"{f}_entropy{suffix}"
            if c not in df:
                continue
            mw = ks.mann_whitney(df, c, rng)
            sp = ks.spearman_grade(df, c, rng)
            out.append({"analiz": etiket, "ölçüt": ks.LABEL.get(f"{f}_entropy", f), "n": int(df[c].notna().sum()),
                        "r (High−Low)": mw["rank_biserial_r"] if mw else None, "p_MWU": mw["p"] if mw else None,
                        "rho_grade": sp["rho"] if sp else None, "p_spearman": sp["p"] if sp else None})
    ekle(d, "ANA ANALİZ (tüm tile'lar)", "")
    for t in esikler:
        ekle(d, f"<{t} çekirdekli tile'lar dışlandı", f"_d{t}")
    ekle(d[~d["case_id"].isin(cikar)], f"Uç vaka çıkarıldı ({', '.join(cikar)})", "")
    t_ = pd.DataFrame(out)
    print("\n" + t_.round(3).to_string(index=False))
    print("\nDışlanan tile oranı (medyan) — gruplar arasında farklı mı?")
    for t in esikler:
        c = f"dislanan_tile_orani_{t}"
        lo_, hi_ = d.loc[d.y_high == 0, c].median(), d.loc[d.y_high == 1, c].median()
        mw = ks.mann_whitney(d, c, rng)
        print(f"  <{t}: Low %{100 * lo_:.1f} · High %{100 * hi_:.1f}  (Mann–Whitney {ks.fmt_p(mw['p']) if mw else '-'})")
    top = d.sort_values(f"dislanan_tile_orani_{esikler[-1]}", ascending=False)[["case_id", "grade", f"dislanan_tile_orani_{esikler[-1]}", "area_um2_entropy"]].head(8)
    print(f"\nEn çok tile'ı dışlanan vakalar (<{esikler[-1]}):\n" + top.round(3).to_string(index=False))
    dst = ROOT / "07_statistics" / "duyarlilik_hucre_yogunlugu.xlsx"
    with pd.ExcelWriter(dst, engine="openpyxl") as xw:
        t_.to_excel(xw, sheet_name="ozet", index=False)
        d[["case_id", "grade"] + [c for c in d.columns if c.startswith("dislanan_") or "_entropy_d" in c]].to_excel(xw, sheet_name="vaka_bazinda", index=False)
    print(f"\n✓ {dst}")


def dogrulama_hazirla():
    a = ayarlar()
    import kirc_validate as kv
    present = [s for s in _pilot_slaytlari() if _slide_present(s)]
    if not present:
        raise SystemExit("Slaytı diskte duran başarılı pilot vaka yok. Önce: python scripts/proje.py pilot")
    baslik("DOĞRULAMA TILE'LARI")
    kv.export_tiles(RESULTS, ANNOT, min(int(a["dogrulama_vaka"]), len(present)), int(a["dogrulama_tile_vaka_basina"]),
                    only_slides=present)
    print(f"\nŞİMDİ: QuPath'te {ANNOT} içindeki PNG'lerde çekirdekleri çizin "
          f"(rehber: {ANNOT / 'QUPATH_ETIKETLEME_REHBERI.md'}). Bitince: python scripts/proje.py dogrulama-degerlendir")


MONUSEG_DIR = ROOT / "04_segmentation" / "monuseg_bobrek"


def _degerlendir(annot: Path, ad: str, ayar: bool = True, results: Path | None = RESULTS):
    """Etiketli tile'larda klasik (+ kuruluysa StarDist) değerlendirme + (klasik) parametre ayarı + öneri."""
    import kirc_validate as kv
    a = ayarlar()
    segs = ["classical"] + (["stardist"] if _has("stardist") else [])
    cfg = pipeline_cfg(a)
    baslik(f"DOĞRULAMA — {ad} (yöntemler: {', '.join(segs)})")
    if "stardist" not in segs:
        print("Not: StarDist kurulu değil → yalnız klasik yöntem. Karşılaştırma için: pip install stardist tensorflow")
    per, summ = kv.evaluate(annot, results, segs, cfg, quiet=True)
    n_gt = int(per.drop_duplicates("png")["n_gt"].sum()) if "n_gt" in per else None
    tuned = None
    if ayar:
        try:
            tuned = kv.tune(annot, results, cfg)
        except SystemExit as exc:
            print(f"Parametre ayarı atlandı: {exc}")
    ent_cols = [c for c in (f"entropy_MAE_{f}" for f in kp.MORPH_FEATURES) if c in summ]
    summ["ort_entropi_hatasi"] = summ[ent_cols].mean(axis=1)
    ok = summ[summ["F1 (mikro)"] >= 0.70]
    best = (ok if len(ok) else summ).sort_values("ort_entropi_hatasi").iloc[0]
    print(f"\n{per['png'].nunique()} görüntü" + (f", {n_gt} etiketli çekirdek" if n_gt else ""))
    print(summ[["segmenter", "F1 (mikro)", "PQ", "pixel_dice", "area_bias_%", "ort_entropi_hatasi"]].round(3).to_string(index=False))
    print(f"\nÖNERİ: '{best['segmenter']}' (ortalama entropi hatası {best['ort_entropi_hatasi']:.3f} bit, F1 {best['F1 (mikro)']:.2f}).")
    if tuned and best["segmenter"] == "classical":
        print(f"  Klasik parametre önerisi (test yarısı F1 {tuned['best_row']['test_F1']:.2f} vs varsayılan "
              f"{tuned['default_row']['test_F1']:.2f}) → {annot / 'evaluation' / 'best_params.json'}")
    print(f"  Ayrıntı ve görseller: {annot / 'evaluation'}")
    return per, summ, tuned


def monuseg():
    """Uzman etiketli MoNuSeg böbrek görüntüleriyle (TCGA, 40x) segmentasyon doğrulaması — elle çizim gerekmez."""
    import kirc_validate as kv
    if not (MONUSEG_DIR / "tiles_manifest.csv").exists():
        baslik("MoNuSeg İNDİRME (bir kez, ~100 MB)")
        kv.prepare_monuseg(MONUSEG_DIR, ROOT / "02_raw_data" / "monuseg")
        ortak = set(pd.read_csv(MONUSEG_DIR / "tiles_manifest.csv")["case_id"]) & set(
            kp.load_case_table(yol(ayarlar()["excel"]))[0]["case_id_excel"])
        if ortak:
            print(f"Not: MoNuSeg'deki {', '.join(sorted(ortak))} senin kohortunda da var (yöntem eğitilmediği için sorun değil).")
    _degerlendir(MONUSEG_DIR, "MoNuSeg böbrek (uzman etiketli)", results=None)   # boya parametreleri görüntünün kendisinden


def dogrulama_degerlendir():
    n = len(list(ANNOT.glob("*.geojson")))
    if n == 0:
        raise SystemExit(f"{ANNOT} içinde .geojson yok. Önce QuPath'te çizim yapıp dışa aktarın.")
    _degerlendir(ANNOT, f"kendi tile'ların ({n} etiketli)", ayar=n >= 4)


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def ayarla(anahtar: str, deger: str):
    a = ayarlar()
    if anahtar not in DEFAULTS:
        raise SystemExit(f"Bilinmeyen ayar: {anahtar}. Geçerli: {', '.join(DEFAULTS)}")
    v: object = deger
    if deger.lower() in ("true", "false"):
        v = deger.lower() == "true"
    elif deger.lower() in ("none", "null", ""):
        v = None
    else:
        try:
            v = float(deger) if "." in deger else int(deger)
        except ValueError:
            pass
    old = a.get(anahtar)
    a[anahtar] = v
    CFG_FILE.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{anahtar}: {old!r} → {v!r}  ({CFG_FILE})")
    if anahtar in ("segmenter", "target_mpp", "parametre_dosyasi"):
        print("Not: segmentasyon parametresi değişti → toplu analizde vakalar bu ayarla (yeniden) işlenecek.")


def toplu(onay: bool, limit: int | None):
    a = ayarlar()
    if not onay:
        raise SystemExit("200 vakalık analiz için --onay gerekli. Önce pilot ve doğrulama adımlarını tamamlayın.")
    kp.setup_logging(RESULTS)
    cfg = pipeline_cfg(a)
    g = gdc_yolu(a)
    baslik(f"TOPLU ANALİZ — yöntem {cfg.segmenter}, {cfg.target_mpp} µm/px, "
           f"{'indir→analiz→sil' if a['indirdikten_sonra_sil'] else 'indirilen slaytlar saklanır'}")
    if not g:
        print("⚠ gdc-client bulunamadı: yalnız diskte olan slaytlar analiz edilecek.")
    kp.run_batch(yol(a["excel"]), yol(a["wsi_klasoru"]), RESULTS, cfg, limit=limit, gdc_client=g,
                 delete_after=bool(a["indirdikten_sonra_sil"]), min_free_gb=float(a["min_bos_alan_gb"]),
                 gdc_processes=int(a["gdc_paralel_baglanti"]))
    topla()
    durum()


def istatistik():
    a = ayarlar()
    import kirc_stats as ks
    ana, allc, notes = ks.load_analysis_set(RESULTS, yol(a["excel"]))
    baslik(f"İSTATİSTİK ({len(ana)} vaka)")
    ks.run(ana, allc, RESULTS, ROOT / "07_statistics", demo=False, notes=notes)
    topla()


def topla():
    """Ham pipeline çıktılarından senin klasör yapına özet dosyaları kopyalar (büyük dosyalar kopyalanmaz)."""
    clp = RESULTS / "case_level_results.csv"
    if clp.exists():
        cl = pd.read_csv(clp)
        ok = cl[cl["analysis_status"].isin(kp.SUCCESS_STATUSES)]
        base = [c for c in ("case_id", "slide_id", "grade", "grade_group", "segmenter", "n_nuclei") if c in cl]
        morph = base + [c for c in cl.columns if any(c.startswith(f"{f}_") for f in kp.MORPH_FEATURES)
                        and c.split("_")[-1] in ("mean", "std", "median", "iqr", "cv")]
        ok[morph].to_csv(ROOT / "05_morphometry" / "morfometri_vaka_ozeti.csv", index=False)
        ent = base + [c for c in cl.columns if "entropy" in c and not c.startswith(("patch_entropy", "gray", "red", "green", "blue"))]
        with pd.ExcelWriter(ROOT / "06_entropy" / "entropi_tablosu.xlsx", engine="openpyxl") as xw:
            ok[ent].to_excel(xw, sheet_name="entropi", index=False)
            pd.DataFrame(kp.COLUMN_DICTIONARY, columns=["sütun", "birim", "açıklama"]).to_excel(xw, sheet_name="aciklama", index=False)
        ok[ent].to_csv(ROOT / "06_entropy" / "entropi_tablosu.csv", index=False)
        for f in ("case_level_results.xlsx", "quality_control_report.xlsx", "aksiyon_listesi.xlsx", "aksiyon_listesi_galeri.png"):
            if (RESULTS / f).exists():
                shutil.copy2(RESULTS / f, ROOT / "09_results" / f)
    if (RESULTS / "qc_gallery").exists():
        shutil.copytree(RESULTS / "qc_gallery", ROOT / "03_preprocessed_images" / "qc_galerisi", dirs_exist_ok=True)
    for sid in _pilot_slaytlari():
        cd = RESULTS / "cases" / sid
        for src, dst in ((cd / "qc_segmentation_examples.png", ROOT / "04_segmentation" / "ornekler" / f"{sid}_segmentasyon.png"),
                         (cd / "focus_calibration.png", ROOT / "03_preprocessed_images" / "odak_kalibrasyonu" / f"{sid}.png"),
                         (cd / "qc_feature_histograms.png", ROOT / "05_morphometry" / "pilot_histogramlar" / f"{sid}.png")):
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    st = ROOT / "07_statistics"
    if st.exists():
        (ROOT / "08_figures").mkdir(exist_ok=True)
        for f in st.glob("fig*.png"):
            shutil.copy2(f, ROOT / "08_figures" / f.name)


def durum():
    a = ayarlar()
    baslik("PROJE DURUMU")
    adim = None
    if not CFG_FILE.exists():
        adim = "python scripts/proje.py kurulum"
    ex = yol(a["excel"])
    print(f"Excel: {'✓' if ex.exists() else '✗'} {ex}")
    print(f"Yöntem: {a['segmenter']} · çözünürlük {a['target_mpp']} µm/px · indir-sil: {a['indirdikten_sonra_sil']}")
    wsi = yol(a["wsi_klasoru"])
    if wsi.exists():
        n_svs = sum(1 for p in wsi.rglob("*") if p.suffix.lower() in kp.SUPPORTED_WSI_EXT)
        print(f"Diskteki slayt: {n_svs} · boş alan {shutil.disk_usage(wsi).free / 1e9:.0f} GB")
    clp = RESULTS / "case_level_results.csv"
    n_ok = 0
    if clp.exists():
        cl = pd.read_csv(clp)
        vc = cl["analysis_status"].value_counts()
        n_ok = int(vc.reindex(list(kp.SUCCESS_STATUSES)).fillna(0).sum())
        print("\nAnaliz durumları:\n" + vc.to_string())
        if "grade" in cl:
            okg = cl[cl["analysis_status"].isin(kp.SUCCESS_STATUSES)]["grade"].value_counts().reindex(["G1", "G2", "G3", "G4"]).fillna(0).astype(int)
            print("Başarılı vakalar grade'e göre: " + ", ".join(f"{g}={n}" for g, n in okg.items()))
        if (RESULTS / "aksiyon_listesi.xlsx").exists():
            act = pd.read_excel(RESULTS / "aksiyon_listesi.xlsx")
            if len(act):
                print(f"⚠ Aksiyon listesinde {int(act['zorunlu'].sum())} zorunlu, {int((~act['zorunlu']).sum())} kontrol önerisi "
                      f"→ 09_results/aksiyon_listesi.xlsx")
    if FROZEN_PILOT.exists():
        print(f"Dondurulmuş (TS/BS) pilot arşivi: ✓ {FROZEN_PILOT.relative_to(ROOT)}")
    karar = mpp_kurali(RESULTS)
    if karar is not None and abs(karar - float(a["target_mpp"])) > 1e-9:
        print(f"⚠ Kural {karar} µm/px diyor, ayar {a['target_mpp']} → python scripts/proje.py ayarla target_mpp {karar}")
    pilot_done = bool(_pilot_slaytlari())
    n_gj = len(list(ANNOT.glob("*.geojson"))) if ANNOT.exists() else 0
    evaluated = (ANNOT / "evaluation" / "validation_report.xlsx").exists()
    print(f"\nPilot: {'✓' if pilot_done else '✗'} · doğrulama tile'ı: {len(list(ANNOT.glob('*.png'))) if ANNOT.exists() else 0} "
          f"· çizilmiş (geojson): {n_gj} · değerlendirme: {'✓' if evaluated else '✗'} · başarılı vaka: {n_ok}")
    if adim is None:
        if not (RESULTS / "case_file_mapping.csv").exists():
            adim = "python scripts/proje.py liste"
        elif not pilot_done:
            adim = "python scripts/proje.py pilot"
        elif not ANNOT.exists() or not any(ANNOT.glob("*.png")):
            adim = "python scripts/proje.py dogrulama-hazirla"
        elif n_gj == 0:
            adim = f"QuPath'te {ANNOT} içindeki tile'ları çiz (kod dışı iş)"
        elif not evaluated:
            adim = "python scripts/proje.py dogrulama-degerlendir"
        elif n_ok < 150:
            adim = "python scripts/proje.py toplu --onay"
        elif not (ROOT / "07_statistics" / "statistics_report.xlsx").exists():
            adim = "python scripts/proje.py istatistik"
        else:
            adim = "Bulguları yorumla ve raporu yaz (YOL_HARITASI.md, Adım 9-11)"
    print(f"\n➜ SIRADAKİ ADIM: {adim}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="TÜBİTAK 2209-A nükleer entropi projesi — tek giriş noktası")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kurulum"); p.add_argument("--test", action="store_true", help="Self-test de çalıştır")
    sub.add_parser("durum")
    sub.add_parser("liste")
    p = sub.add_parser("pilot"); p.add_argument("--vakalar", nargs="*"); p.add_argument("--sil", action="store_true")
    sub.add_parser("dogrulama-hazirla")
    sub.add_parser("dogrulama-degerlendir")
    p = sub.add_parser("ayarla"); p.add_argument("anahtar"); p.add_argument("deger")
    p = sub.add_parser("toplu"); p.add_argument("--onay", action="store_true"); p.add_argument("--limit", type=int)
    sub.add_parser("istatistik")
    sub.add_parser("topla")
    sub.add_parser("dx-gecis")
    sub.add_parser("dondurulmus-karsilastir")
    sub.add_parser("cozunurluk-duyarlilik")
    sub.add_parser("monuseg")
    sub.add_parser("hiz-testi")
    sub.add_parser("yedek-sec")
    sub.add_parser("stardist-ayar")
    sub.add_parser("yogunluk-duyarlilik")
    p = sub.add_parser("tile-incele"); p.add_argument("vaka"); p.add_argument("--n", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd != "kurulum" and not CFG_FILE.exists():
        raise SystemExit("Önce: python scripts/proje.py kurulum")
    {"kurulum": lambda: kurulum(a.test), "durum": durum, "liste": liste,
     "pilot": lambda: pilot(a.vakalar, a.sil), "dogrulama-hazirla": dogrulama_hazirla,
     "dogrulama-degerlendir": dogrulama_degerlendir, "ayarla": lambda: ayarla(a.anahtar, a.deger),
     "toplu": lambda: toplu(a.onay, a.limit), "istatistik": istatistik, "topla": topla,
     "dx-gecis": dx_gecis, "dondurulmus-karsilastir": dondurulmus_karsilastir,
     "cozunurluk-duyarlilik": cozunurluk_duyarlilik, "monuseg": monuseg, "hiz-testi": hiz_testi, "yedek-sec": yedek_sec, "stardist-ayar": stardist_ayar, "yogunluk-duyarlilik": yogunluk_duyarlilik,
     "tile-incele": lambda: tile_incele(a.vaka, a.n)}[a.cmd]()


if __name__ == "__main__":
    main()
