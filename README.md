# H&E Boyalı Tümör Dokularında Nükleer Morfometrik Entropi Analizi

TÜBİTAK 2209-A Üniversite Öğrencileri Araştırma Projeleri Destek Programı kapsamında yürütülen projenin analiz kodları.

**Proje yürütücüsü:** Gülnihal Kelkit
**Danışman:** Dr. Öğr. Üyesi Cüneyd Parlayan
**Kurum:** Biruni Üniversitesi

Proje, TCGA-KIRC (berrak hücreli böbrek karsinomu) kohortundaki 200 hastanın FFPE tanısal tüm slayt görüntülerinde çekirdekleri
segmente eder. Ardından her hastada çekirdek alanı, çevresi, daireselliği ve eksantrisitesinin dağılımlarından Shannon entropisini
hesaplar ve bu entropiyi tümör derecesiyle (grade) karşılaştırır. Yöntem ve bulgular proje sonuç raporunda ayrıntılı olarak anlatılmıştır.

> Bu yazılım araştırma amaçlıdır, tanı koymaz.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `scripts/proje.py` | Tek giriş noktası. Projedeki tüm adımlar bu dosyanın komutlarıyla sırayla çalıştırılır. |
| `scripts/kirc_entropy_pipeline.py` | Görüntü analizi: Excel/ID doğrulama, GDC'den indirme, doku maskesi, kalem ve odak kontrolü, boya taraması, Macenko normalizasyonu, segmentasyon (klasik veya StarDist), morfometri, entropi. |
| `scripts/kirc_validate.py` | Segmentasyon doğrulaması: MoNuSeg ve QuPath etiketleriyle F1, Dice, PQ ve entropi hatası, eşik ayarı. |
| `scripts/kirc_stats.py` | İstatistik ve figürler: Mann-Whitney U, Spearman, Kruskal-Wallis, lojistik regresyon, merkez etkisi (van Elteren), ROC ve çapraz doğrulamalı AUC, artımsal değer, duyarlılık analizleri. |
| `scripts/requirements.txt` | Gerekli Python paketleri. |
| `proje_ayarlari.json` | Rapordaki ana analizde kullanılan ayarlar (StarDist, 0,25 µm/piksel, slayt başına en fazla 500 tile). |
| `02_raw_data/` | Vaka listesi Excel'inin konacağı klasör (bkz. `02_raw_data/README.md`). |

Kodlar çalıştıkça proje klasöründe şu klasörler oluşur (GitHub'a yüklenmez):

```
02_raw_data/wsi/            slaytlar (indir, analiz et, sil)
03_preprocessed_images/     kalite kontrol galerisi, odak kalibrasyonu, tile incelemeleri
04_segmentation/            segmentasyon örnekleri, MoNuSeg doğrulaması, QuPath etiketleri
05_morphometry/             vaka düzeyi morfometri özeti
06_entropy/                 entropi tablosu
07_statistics/              istatistik raporu (Excel) ve figürler
08_figures/                 tüm figürlerin kopyası
09_results/pipeline/        ham analiz çıktıları (vaka başına klasörler)
```

## 1. Kurulum

Python 3.10 veya üstü gerekir (raporda 3.13 kullanıldı).

```bash
git clone <bu deponun adresi>
cd nukleer-entropi-ccrcc
python -m venv .venv
# macOS/Linux: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r scripts/requirements.txt
```

Slaytları indirmek için GDC Data Transfer Tool (`gdc-client`) gerekir:
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
İndirilen programı `scripts/` klasörüne koyun. Windows'ta dosya adı `gdc-client.exe` olacağı için
`python scripts/proje.py ayarla gdc_client scripts/gdc-client.exe` komutunu çalıştırın.

Kurulumu ve kodların doğru çalıştığını kontrol etmek için:

```bash
python scripts/proje.py kurulum --test     # klasörleri kurar, paketleri ve Excel'i kontrol eder, self-test çalıştırır
python scripts/kirc_validate.py selftest   # segmentasyon doğrulama testleri
python scripts/kirc_stats.py --demo --out demo_istatistik   # SİMÜLE veriyle örnek figürler (sonuç değildir)
```

Pipeline self-testi 47 testten oluşur. StarDist kuruluysa StarDist testi de çalışır.

`python scripts/proje.py durum` komutu her an nerede olduğunuzu ve sıradaki adımı gösterir.

## 2. Rapordaki analizi tekrarlamak

Bu yol, rapordaki son 200 hastalık analiz setini (`02_raw_data/TCGA_KIRC_200_Case_DX_YEDEKLI.xlsx`) ve
`proje_ayarlari.json` içindeki ayarları kullanır. Uzun süren komutları macOS'ta `caffeinate -i` ile başlatmak
bilgisayarın uykuya geçmesini önler.

| Sıra | Komut | Ne yapar | Rapordaki karşılığı |
|---|---|---|---|
| 1 | `python scripts/proje.py liste` | Excel'deki ID'leri kontrol eder, GDC indirme listesini oluşturur. | Bölüm 2.1.2 |
| 2 | `python scripts/proje.py monuseg` | Uzman etiketli MoNuSeg böbrek görüntülerini indirir (~100 MB), klasik yöntemi ve StarDist'i karşılaştırır. | Bölüm 2.1.4, Tablo 1 |
| 3 | `python scripts/proje.py stardist-ayar` | StarDist karar eşiklerini hasta bazında ayrılmış ayar yarısında seçer ve test yarısında raporlar. `04_segmentation/monuseg_bobrek/evaluation/stardist_best_params.json` dosyasını yazar. | Tablo 1 (olasılık eşiği 0,69 → 0,50) |
| 4 | `python scripts/proje.py pilot` | Her grade'den 3 vaka (12 hasta) analiz eder ve slaytları saklar. | Bölüm 2.1.2, 2.2.6 |
| 5 | `python scripts/proje.py cozunurluk-duyarlilik` | Pilot slaytlarını 0,5 µm/piksel'de de analiz eder. | Bölüm 2.2.6 (kesit türü ve çözünürlük) |
| 6 | `python scripts/proje.py toplu --onay` | 200 vakayı sırayla indirir, analiz eder ve siler. Kesilirse aynı komutla kaldığı yerden devam eder. | Bölüm 2.1.3-2.1.6 |
| 7 | `python scripts/proje.py istatistik` | Tüm testleri, figürleri ve Excel raporunu üretir. | Bölüm 2.2.1-2.2.5, 2.2.7 |
| 8 | `python scripts/proje.py yogunluk-duyarlilik` | Hücreden fakir tile'ları (<100 ve <150 çekirdek) ve uç değerli vakayı çıkararak analizi tekrarlar. | Tablo 7, Bölüm 2.2.6 |
| 9 | `python scripts/proje.py tile-incele TCGA-CZ-5455` | Bir vakanın en yüksek ve en düşük entropili tile'larını yan yana gösterir. | Şekil 13 |

3\. adım bittiğinde `proje_ayarlari.json` içindeki `parametre_dosyasi` alanının gösterdiği dosya oluşmuş olur ve
sonraki tüm analizler bu eşiklerle yapılır. 6. adım bu dosya olmadan başlatılmamalıdır.

İsteğe bağlı adımlar:

* `python scripts/proje.py dogrulama-hazirla` ve `dogrulama-degerlendir`: Kohorttan tile'ları QuPath'te elle etiketleyip
  segmentasyonu kendi verinizde değerlendirmek için. Raporda bu etiketler eksik kaldığı için nicel doğrulamada kullanılmamıştır.
* `python scripts/proje.py hiz-testi`: Klasik yöntem ve StarDist için tile başına süreyi ölçer, 200 vakanın süresini tahmin eder.

## 3. Projeyi baştan kurmak (vaka seçiminden itibaren)

Projede vaka listesi önce dondurulmuş kesitlerle (TS/BS) hazırlanmış, sonra FFPE tanısal (DX) slaytlara geçilmiş ve
20x taranmış 8 vaka yedeklerle değiştirilmiştir. Bu süreci aynen izlemek için ilk Excel'i (`TCGA_KIRC_200_Case.xlsx`)
`02_raw_data/` içine koyun ve `proje_ayarlari.json` dosyasını silin. `kurulum` komutu varsayılan ayarlarla yeni bir dosya yazar.
Sonra komutları şu sırayla çalıştırın:

```bash
python scripts/proje.py kurulum
python scripts/proje.py liste
python scripts/proje.py pilot                    # dondurulmuş kesitlerle pilot (12 hasta)
python scripts/proje.py dx-gecis                 # her hasta için DX slaytını seçer, _DX.xlsx yazar, çözünürlüğü 0,25 yapar
python scripts/proje.py pilot                    # aynı 12 hasta, DX slaytlarla; 40x kuralını kontrol eder
python scripts/proje.py cozunurluk-duyarlilik    # DX pilotu 0,5 µm/piksel'de
python scripts/proje.py dondurulmus-karsilastir  # dondurulmuş ve FFPE kesitlerin karşılaştırması
python scripts/proje.py monuseg
python scripts/proje.py stardist-ayar
python scripts/proje.py ayarla segmenter stardist
python scripts/proje.py ayarla parametre_dosyasi 04_segmentation/monuseg_bobrek/evaluation/stardist_best_params.json
python scripts/proje.py ayarla max_tile 500
python scripts/proje.py toplu --onay
python scripts/proje.py yedek-sec                # 20x vakaların yerine aynı grade'den yedek (seed 2209); _YEDEKLI.xlsx yazar
python scripts/proje.py toplu --onay             # yalnız yedekler analiz edilir
python scripts/proje.py istatistik
python scripts/proje.py yogunluk-duyarlilik
```

## 4. Önceden belirlenmiş kurallar

Kodda sabitlenmiş ve sonuçlara bakılmadan uygulanan kurallar:

* **Slayt türü:** Her hasta için FFPE tanısal (DX) slayt. Birden fazla DX varsa en büyük dosya seçilir (`dx-gecis`).
* **Çözünürlük:** DX slaytların en az %90'ı 40x (≤ 0,275 µm/piksel) ise analiz 0,25 µm/piksel'de yapılır, değilse 0,5'te.
  Hedeften kaba taranmış slaytlar büyütülmez, `RESOLUTION_INSUFFICIENT` olarak işaretlenir.
* **Yedek vaka:** Kohort dışındaki TCGA-KIRC hastaları arasından aynı GDC grade'i, rastgele sıra (seed 2209 + grade numarası),
  en büyük DX slayt ve 40x taranmış ilk aday (`yedek-sec`).
* **Segmentasyon eşiği:** Yalnızca MoNuSeg doğrulama verisinde, hasta bazında ayrılmış ayar yarısında seçilir.
  Grade sonuçlarına bakılmaz.

## 5. Ana parametreler

| Parametre | Değer |
|---|---|
| Çözünürlük | 0,25 µm/piksel |
| Tile boyutu | 256 µm (1024 × 1024 piksel) |
| Slayt başına tile | En fazla 500 (sabit tohumla rastgele seçim) |
| Segmentasyon | StarDist `2D_versatile_he`, olasılık eşiği MoNuSeg'de ayarlı (0,50) |
| Çekirdek filtreleri | 8-400 µm², katılık ≥ 0,75, görüntü kenarına değenler hariç |
| Odak kontrolü | 64 µm'lik bloklarda çok ölçekli bant oranı; bulanık bloklardaki çekirdekler dışlanır |
| Entropi bin'leri | Alan 0-200 µm² (40), çevre 0-80 µm (40), dairesellik 0-1 (20), eksantrisite 0-1 (20), log₂ |
| Tile entropisi | En az 30 çekirdek; uzamsal heterojenlik = tile entropilerinin SD'si ve 90. persentili |

Parametrelerin tamamı `scripts/kirc_entropy_pipeline.py` içindeki `Config` sınıfında, açıklamalarıyla birlikte yer alır.
Her analizde kullanılan ayarlar `09_results/pipeline/analysis_config.json` dosyasına yazılır. Bir segmentasyon
parametresi değişirse vakalar yeniden analiz edilir. `kirc_stats.py` farklı parametrelerle analiz edilmiş vakaları
aynı istatistikte birleştirmeyi reddeder.

## 6. Figürlerin rapordaki karşılıkları

`07_statistics/` (ve `08_figures/`) içindeki dosyalar:

| Dosya | Rapor |
|---|---|
| `fig01_workflow.png` | Şekil 1 |
| `fig02_cohort_flow.png` | Şekil 2 |
| `fig03_boxplot_low_high.png` | Şekil 5 |
| `fig04_boxplot_by_grade.png` | Şekil 6 |
| `fig07_forest_logistic.png` | Şekil 7 |
| `fig14_site_by_grade.png` | Şekil 8 |
| `fig06_roc.png` | Şekil 9 |
| `fig08_correlation_heatmap.png` | Şekil 10 |
| `fig13_incremental_value.png` | Şekil 11 |
| `fig11_entropy_heatmap_gallery.png` | Şekil 12 |
| `fig15_stain_normalization.png` | Şekil 14 |
| `fig05_histograms.png` | Ek Şekil A1 |
| `fig12_pooled_feature_distributions.png` | Ek Şekil A2 |
| `fig09_bin_sensitivity.png` | Ek Şekil A3 |
| `fig10_confounders.png` | Ek Şekil A4 |
| `statistics_report.xlsx` | Tablo 3-6, Ek Tablo B1 |

Vaka düzeyindeki kalite kontrol özetleri (Şekil 3, Ek Şekil A5) `09_results/pipeline/cases/<slayt>/qc_overview.png`
dosyalarındadır. Segmentasyon örnekleri her vaka için `qc_segmentation_examples.png` olarak üretilir.

## 7. Süre ve disk

* Toplu analiz slaytları tek tek indirir, analiz eder ve siler. Bir DX slaytı yaklaşık 1 GB'tır, en az 5 GB boş alan yeterlidir.
* StarDist ile 200 vaka CPU'da yaklaşık 1,5 gün sürer. `hiz-testi` komutu kendi bilgisayarınız için tahmin verir.
* Analiz kesilirse aynı komut tamamlanan vakaları atlayarak devam eder.

## 8. Notlar

* Aynı Excel, aynı ayarlar ve aynı tohumlarla sonuçlar tekrarlanabilir. TensorFlow/StarDist sürümleri farklıysa
  çekirdek sınırlarında küçük farklar olabilir. Kullanılan paket sürümleri `kurulum` çıktısında listelenir.
* Doku maskesi tümör, stroma ve nekrozu ayırmaz. Lenfosit ve endotel çekirdekleri de ölçüme girer (raporda kısıtlılık olarak belirtilmiştir).
* TCGA görüntüleri GDC Veri Portalı'ndan açık erişimle indirilir. MoNuSeg verisi CC BY-NC-SA 4.0 lisanslıdır.

## Kaynak veri ve yöntemler

* Cancer Genome Atlas Research Network (2013). Comprehensive molecular characterization of clear cell renal cell carcinoma. *Nature*, 499, 43-49.
* Kumar ve ark. (2017). A dataset and a technique for generalized nuclear segmentation for computational pathology. *IEEE TMI*, 36(7), 1550-1560.
* Macenko ve ark. (2009). A method for normalizing histology slides for quantitative analysis. *ISBI 2009*, 1107-1110.
* Schmidt, Weigert, Broaddus ve Myers (2018). Cell detection with star-convex polygons. *MICCAI 2018*, 265-273.
