# 02_raw_data

Bu klasöre vaka listesini içeren Excel dosyası konur:

`TCGA_KIRC_200_Case_DX_YEDEKLI.xlsx`: rapordaki 200 hastalık son analiz seti
(FFPE tanısal slaytlar, 20x vakaların yerine seçilen yedeklerle birlikte).

Excel'de `Case_Selection` adlı bir sayfa olmalı. Zorunlu sütunlar: `Case ID`, `Grade`, `Selected?`,
`Selected Slide ID`, `File ID`, `Submitter ID`. İsteğe bağlı: `No.`, `Exclusion Reason`, `Notes`,
`No. of Slides`. `Selected?` değeri `No` olan ya da `Exclusion Reason` alanı dolu olan satırlar analize alınmaz.

`wsi/` alt klasörü slaytların geçici olarak indirildiği yerdir. Bu klasör GitHub'a yüklenmez.
