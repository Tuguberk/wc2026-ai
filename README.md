# ⚽ WC 2026 — Yapay Zeka Tahmin Sistemi

Dünya Kupası 2026 maçları için istatistiksel olasılık tahminleri.  
Hiyerarşik Bayesian Poisson modeli, LightGBM ve piyasa verilerinin ağırlıklı birleşimi.
---

## Nasıl Çalışır?

49.000+ uluslararası maç verisi (1872–2025) üzerinde eğitilmiş üç farklı model, optimize edilmiş ağırlıklarla birleştirilir:

```
Bayesian Poisson  ──┐
LightGBM (23 özellik) ──┼──► Ağırlıklı Birleştirme ──► Olasılık Kalibrasyonu ──► 25.000 Simülasyon
Piyasa Verileri   ──┘
```

**Pipeline:**
1. Maç sonuçları football-data.org'dan çekilir
2. 49k+ tarihsel maçla Bayesian modeli güncellenir  
3. LightGBM 23 sinyalle (Elo, form, FIFA sıralaması) yeniden eğitilir
4. Piyasa tahminleri Shin de-vig yöntemiyle arındırılır
5. Üç model birleştirilir, isotonic regresyonla kalibre edilir
6. 32 takımlı turnuva 25.000 kez simüle edilir
7. Sonuçlar `data/outputs/`'a yazılır, Streamlit uygulaması güncellenir

---

## Model Performansı

Son 1 yıllık tarihsel veri üzerinde (n = 957 maç, modelin hiç görmediği test seti):

| Model | Brier Skoru ↓ | Doğruluk ↑ |
|---|---|---|
| **Birleşik Model** | **0.541** | **%55.2** |
| Bayesian Poisson | 0.543 | %56.1 |
| LightGBM | 0.552 | %55.8 |
| Kıyaslama: eşit şans (⅓) | 0.667 | %33.3 |
| Kıyaslama: hep ev sahibi | 0.648 | %44.6 |

> Futbolda %55+ doğruluk, büyük veri altyapılarına sahip profesyonel sistemlerle aynı aralıktadır.
> Önemli olan doğruluktan çok **kalibrasyondur** — %70 dediğimizde gerçekten ~%70 olmalı.

---

## Kullanılan Teknolojiler

| Katman | Teknoloji |
|---|---|
| Bayesian model | PyMC 5 — hiyerarşik Poisson, zaman-decay ağırlıkları |
| Makine öğrenmesi | LightGBM — çok sınıflı (H/D/A), 23 özellik |
| Özellik mühendisliği | Elo puanları, son form, FIFA dünya sıralaması (1992–2024) |
| Kalibrasyon | Isotonic regresyon |
| Simülasyon | Monte Carlo, 25.000 iterasyon, Wilson %95 güven aralıkları |
| Veri kaynakları | football-data.org · Kaggle · FIFA ranking CSV |
| Arayüz | Streamlit + Plotly |

---

## Kurulum

```bash
git clone https://github.com/tugberkakbulut/wc2026-ai.git
cd wc2026-ai

python -m venv .venv && source .venv/bin/activate
make setup

cp .env.example .env
# .env dosyasına API anahtarlarını ekle (en az FOOTBALL_DATA_API_KEY)

make update   # ~7 dakika
make app      # http://localhost:8501
```

---

## Güncelleme Akışı

```bash
make update
```

Her çalıştırmada şunlar olur:
- Yeni maç sonuçları çekilir
- Model verileri güncellenerek yeniden eğitilir
- Kalibre edilmiş tahminler üretilir
- Tüm takımlar için bracket simülasyonu çalışır
- Tarihli snapshot `data/snapshots/{ts}/` altına kaydedilir
- `data/outputs/` dosyaları Streamlit için güncellenir

**Ne zaman çalıştırmalısın:** Maç günlerinin sonunda, Türkiye maçlarından hemen sonra.

---

## Yapı

```
wc2026-ai/
├── app/
│   └── streamlit_app.py        # Türkçe arayüz — sadece data/outputs/ okur
├── src/wc26/
│   ├── config.py               # pydantic-settings (.env)
│   ├── update.py               # Ana pipeline orkestrasyonu
│   ├── fetchers/
│   │   ├── kaggle_historical.py  # 49k+ tarihsel maç
│   │   ├── footballdata.py       # WC2026 fikstür + sonuçlar
│   │   ├── fifa_ranking.py       # FIFA dünya sıralaması (1992–2024)
│   │   └── odds_live.py          # Piyasa tahmin verileri
│   ├── features/
│   │   ├── elo.py               # Elo puanı hesaplama
│   │   ├── form.py              # Son N maç form metrikleri
│   │   └── pipeline.py          # 23 özelliğin birleşimi
│   ├── models/
│   │   ├── bayesian_poisson.py  # PyMC hiyerarşik model
│   │   ├── lgbm.py              # LightGBM çok sınıflı
│   │   ├── calibration.py       # Isotonic regresyon
│   │   ├── ensemble.py          # Ağırlıklı birleştirme
│   │   └── market_model.py      # Shin de-vig
│   └── sim/
│       ├── bracket_sim.py       # 32 takım tam bracket, Wilson CI
│       ├── group_sim.py         # Grup aşaması Monte Carlo
│       └── turkey.py            # Türkiye özet istatistikleri
├── data/
│   ├── outputs/                 # Streamlit'in okuduğu dosyalar (git'te)
│   └── snapshots/               # Tarihli tahmin arşivi (gitignore)
├── tests/                       # Fetcher, model, idempotency, leakage testleri
├── .streamlit/config.toml       # Karanlık tema konfigürasyonu
├── requirements.txt             # Streamlit Cloud için minimal bağımlılıklar
└── Makefile                     # setup · update · app · test · lint
```

---

## Konfigürasyon

`.env.example` dosyasını kopyalayarak `.env` oluştur:

| Değişken | Zorunlu | Açıklama |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | Evet | WC2026 fikstür + sonuçları |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` | Önerilen | Tarihsel maç verisi (GitHub fallback var) |
| `API_FOOTBALL_KEY` | Opsiyonel | Yedek fikstür kaynağı |
| `ODDS_API_KEY` | Opsiyonel | Piyasa tahmin verileri (the-odds-api.com) |
| `MCMC_DRAWS` | Opsiyonel | Varsayılan 1000 |
| `MONTE_CARLO_ITERATIONS` | Opsiyonel | Varsayılan 10000 |

---

## Dürüst Kısıtlamalar

- **Sakatlık ve kadro haberi yok.** Maç öncesi önemli bir oyuncu eksikse model bunu bilemez.
- **Knockout bracket yaklaşıktır.** FIFA'nın resmi çekiliş tablosu henüz yayınlanmamıştı; çeyrek final ve sonrası olasılıklar tahmindir.
- **Futbol öngörülemezdir.** En iyi sistemler bile %55 civarında doğruluk elde eder — sürprizler bu sporun özündedir.

---

## Testler

```bash
make test   # fetcher · model smoke · idempotency · leakage
make lint   # ruff
```

---

*Turnuva boyunca aktif olarak güncellenmektedir — son güncelleme `data/outputs/` commit tarihine bakınız.*
