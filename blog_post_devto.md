# WC 2026 Maç Sonuçlarını AI ile Nasıl Tahmin Ettim?

Geçen dönem Probability & Statistics dersini alırken aklımda sürekli şu soru vardı: *"Bunları gerçek hayatta nasıl kullanacağım?"* Formüller anlamlıydı, sınavlar güzeldi ama uygulaması nerede?

Cevap beklenmedik bir yerden geldi: Dünya Kupası 2026.

Bu yazıda sıfırdan kurduğum bir futbol tahmin sistemini anlatacağım. Sadece "şu kütüphaneyi kullan" değil, **neden bu modeli seçtim, rakipleriyle nasıl karşılaştırdım, matematiği nasıl çalışıyor** bunların hepsini adım adım açıklayacağım.

Spoiler: Model şu an %55 doğrulukla çalışıyor. Bu iyi mi? Kötü mü? Yazının sonunda anlayacaksınız.

---

## Önce Problem: Futbol Neden Bu Kadar Zor?

Futbol tahmin etmek kulağa kolay gelir. Ama şunu düşünün:

- Bir maçta ortalama **2.5 gol** atılır. Poisson dağılımı için bu düşük bir lambda demek, yani rastgelelik çok yüksek.
- Favoriler her zaman kazanmıyor. Tarihsel veriye bakarsanız **ev sahibi takımlar ancak %45 oranında** kazanıyor.
- 90 dakikada tek bir hata maçı değiştirebilir.

![Futbol maç istatistikleri](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Football_in_Bloomington%2C_Indiana%2C_1995.jpg/320px-Football_in_Bloomington%2C_Indiana%2C_1995.jpg)

Akademisyenler bunu "düşük sinyalli yüksek gürültülü ortam" olarak tanımlar. Hava durumu tahmini ya da hisse senedi tahmininden bile zordur, çünkü insan faktörü (motivasyon, formun o günkü hali, taktik tercih) ölçülmesi neredeyse imkânsız değişkenler içerir.

Bu yüzden modeli kurmadan önce kendime şunu sordum: **Ne kadar iyi olmak mümkün?**

Cevap: Dünyanın en iyi sistemleri **%55-58** doğruluk elde ediyor. Bu bizim hedefimizdi.

---

## Neden LLM Kullanmadım?

İlk düşünce herkesin düşündüğü şey: *"ChatGPT'ye sorarım, o tahmin eder."*

Denedim. Claude'a "Türkiye - ABD maçında kim kazanır, olasılık ver" dedim. Cevap geldi: "%40 Türkiye favori."

Ama bu sayı **kalibre değil.** Yani Claude'un söylediği %40, gerçek hayatta %40 gerçekleşme olasılığını temsil etmiyor. Aynı soruyu 10 kez sor, 10 farklı sayı alırsın. LLM'ler tutarlı, kalibre olasılık üretemez.

Daha da önemlisi: **geçmiş veriden öğrenmiyor.** 49.000 tarihsel maçı "biliyor" ama bu bilgi içselleştirilmiş, matematiksel bir model değil. Hallucination riski yüksek.

```
LLM: "Türkiye güçlü bir takım, %40 şansları var"
İstatistiksel model: "49.000 maç, Elo farkı, son form, FIFA sıralaması → %33"
```

İkincisi **açıklanabilir, tekrarlanabilir ve kalibre edilebilir.**

LLM'nin gerçekten işe yarayabileceği tek alan: yapısal olmayan bilgi sakatlık haberleri, basın toplantıları, taktik analizler. Bunları gelecekte bir "agent" olarak sisteme eklemek istedim. Ama temel model istatistik olmalıydı.

---

## Mimari: Üç Katman

Sistemin genel yapısını görelim:

```
┌─────────────────────────────────────────────────────────────┐
│  VERİ KATMANI                                               │
│  49k+ tarihsel maç · FIFA sıralaması · WC2026 fikstür       │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
   │  Bayesian   │ │  LightGBM   │ │   Piyasa     │
   │  Poisson    │ │  23 özellik │ │   Verileri   │
   └──────┬──────┘ └──────┬──────┘ └──────┬───────┘
          └───────────────┼───────────────┘
                          ▼
               ┌──────────────────┐
               │ Ağırlıklı Birleş │
               │ + Kalibrasyon    │
               └────────┬─────────┘
                        ▼
            ┌───────────────────────┐
            │  25.000 × MC Sim      │
            │  Bracket Olasılıkları │
            └───────────────────────┘
```

Her katmanı sırayla anlatalım.

---

## Katman 1: Hierarchical Bayesian Poisson Modeli

### Neden Poisson?

Gol sayısı ayrık (0, 1, 2, 3...) ve nadiren büyük değerler alıyor. Bu tam Poisson dağılımının tanımı:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Ama hangi lambda? İşte burada Bayesian modeli devreye giriyor.

### Hiyerarşik Yapı

Her takımın gizli (latent) **atak** ve **savunma** gücü var:

```python
# Matematiksel gösterim (PyMC ile implement edildi)

attack[t]  ~ Normal(μ_att, σ_att)   # tüm takımlar aynı prior'ı paylaşır
defense[t] ~ Normal(μ_def, σ_def)

λ_home = exp(intercept + home_adv × (1 - neutral) + attack[home] - defense[away])
λ_away = exp(intercept + attack[away] - defense[home])

home_goals ~ Poisson(λ_home)
away_goals ~ Poisson(λ_away)
```

**Neden hiyerarşik?** Küçük takımlar hakkında az veri var. Hiyerarşik yapı, az veri olan takımların tahminlerini genel ortalamayla "düzleştirir" (shrinkage). Bu klasik istatistikte **Stein paradoksu**nun pratik uygulaması.

### Zaman Ağırlıkları

2010'daki Türkiye ile 2025'teki Türkiye aynı takım değil. 8 yıl önceki maçların bu anki tahmine etkisi ne olmalı?

**Yarı-ömür weighting:**

```python
# t gün önceki maçın ağırlığı
weight = exp(-log(2) * days_ago / 730)
# 730 gün (2 yıl) önce → ağırlık = 0.5
# 4 yıl önce → ağırlık = 0.25
```

Bu exponential decay, fizikteki radyoaktif bozunma formülüyle aynı. Derste öğrendiğimiz bir şeyi burada görebiliyorum,güzel bir detay.

![Bayesian model şeması](https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Bayes%27_Theorem_MMB_01.jpg/320px-Bayes%27_Theorem_MMB_01.jpg)

### Neden Bayesian, Frekantist Değil?

Klasik (frekantist) yaklaşım: maksimum likelihood ile parametre tahmin et, tek bir değer al.

Bayesian yaklaşım: **posterior dağılımı** hesapla, yani parametrenin olası değerlerinin tüm dağılımını elde et.

```
P(parametreler | veri) ∝ P(veri | parametreler) × P(parametreler)
     posterior              likelihood               prior
```

Bunun pratikte anlamı: **belirsizliği sayısallaştırabiliyorsun.** "Türkiye'nin atak gücü 0.3" demek yerine "Türkiye'nin atak gücü %95 olasılıkla [0.1, 0.5] arasında" diyebiliyorsun.

PyMC ile implement:

```python
import pymc as pm
import numpy as np

with pm.Model() as football_model:
    # Hyperpriors
    mu_att  = pm.Normal("mu_att", mu=0, sigma=1)
    sigma_att = pm.HalfNormal("sigma_att", sigma=1)
    
    # Team-level attack and defense
    attack  = pm.Normal("attack",  mu=mu_att,  sigma=sigma_att, shape=n_teams)
    defense = pm.Normal("defense", mu=0, sigma=1, shape=n_teams)
    
    # Home advantage
    home_adv = pm.Normal("home_adv", mu=0.3, sigma=0.2)
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    
    # Expected goals
    log_lambda_home = intercept + home_adv * (1 - neutral) + attack[home_idx] - defense[away_idx]
    log_lambda_away = intercept + attack[away_idx] - defense[home_idx]
    
    # Likelihood (weighted)
    home_goals = pm.Poisson("home_goals", mu=pm.math.exp(log_lambda_home),
                             observed=home_goals_obs)
    away_goals = pm.Poisson("away_goals", mu=pm.math.exp(log_lambda_away),
                             observed=away_goals_obs)
    
    # MCMC sampling
    trace = pm.sample(draws=1000, tune=500, chains=2, target_accept=0.9)
```

MCMC (Markov Chain Monte Carlo) bu posterior'u örnekleyerek hesaplıyor. Her sampling ~90 saniye sürüyor. Deterministik bir çözüm yok, örnekleme var.

---

## Katman 2: LightGBM ile Makine Öğrenmesi

Bayesian model güçlü ama sınırlı: sadece gol verisi kullanıyor. **Elo puanları, son form, FIFA sıralaması** gibi özellikler eklemek için ikinci bir model kurdum.

### Özellik Mühendisliği: 23 Sinyal

```python
FEATURE_COLS = [
    # Elo
    "elo_diff", "home_elo", "away_elo",
    # Son form (son 5 maç)
    "home_win_rate", "away_win_rate",
    "home_gd", "away_gd",           # averaj
    "home_gf", "away_gf",           # atılan gol
    "home_ga", "away_ga",           # yenilen gol
    # FIFA sıralaması
    "home_fifa_rank", "away_fifa_rank",
    "fifa_rank_diff",
    "home_fifa_pts", "away_fifa_pts",
    "fifa_pts_diff",
    # Bağlam
    "tournament_importance",
    "is_neutral",
    "home_rest", "away_rest",        # dinlenme günü
    "home_draw_rate", "away_draw_rate",
]
```

### Elo Puanı Nedir?

Satranç dünyasından gelen bir sıralama sistemi. Basit fikir: kazandığında rakibinden puan alırsın, kaybettiğinde verirsin. Kazanma ihtimalin ne kadar düşükse, kazandığında o kadar çok puan alırsın.

```python
def update_elo(rating_a, rating_b, result, k=30):
    """result: 1=A kazandı, 0.5=beraberlik, 0=B kazandı"""
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    delta = k * (result - expected_a)
    return rating_a + delta, rating_b - delta
```

49.000 tarihsel maçı baştan sona geçirerek her takım için güncel Elo puanı hesapladım. Bu özellik, LightGBM'in en önemli girdilerinden biri oldu.

### FIFA Sıralaması: Veri Sızıntısına Dikkat

Burada kritik bir hata yaptım ve düzelttim.

İlk versiyonda FIFA sıralaması olarak "bugünkü sıralamayı" kullanıyordum. Ama 2018 maçını tahmin ederken 2024 sıralaması kullanmak **veri sızıntısı** (data leakage). Gelecek bilgisini geçmiş tahminlerine sokmuş oluyorsun.

Çözüm: Her maç için o maçın oynandığı tarihteki sıralamayı çek.

```python
def fifa_rank_at(lookup: dict, team: str, date: datetime) -> tuple[float, float]:
    """Maç tarihindeki FIFA sıralamasını döndür (leakage-free)."""
    if team not in lookup:
        return 150.0, 0.0  # bilinmeyen takım → konservatif fallback
    
    ts = lookup[team]  # takımın tüm sıralama geçmişi
    past = ts[ts["date"] <= date]
    
    if past.empty:
        return 150.0, 0.0
    
    row = past.iloc[-1]  # maç tarihinden önceki en son sıralama
    return float(row["rank"]), float(row["points"])
```

Bu detay küçük görünüyor ama model doğruluğu açısından kritik. Leakage'ı keşfetmek için ayrı bir test yazdım:

```python
def test_no_future_leakage():
    """Hiçbir özellik maç tarihinden sonraki veriyi kullanmamalı."""
    for _, row in features.iterrows():
        match_date = row["date"]
        fifa_date  = row["fifa_rank_lookup_date"]
        assert fifa_date <= match_date, f"LEAKAGE: {row['match_id']}"
```

### 3-Way Train/Val/Test Split

```
Tarih sırası:  [──── TRAIN ────][─ VAL ─][─ TEST ─]
                2017-2023        2023-24   2024-25
                  5754 maç        1062      981
```

Neden böyle? Finansal zaman serilerinde de aynı kural: **geçmişle geleceği tahmin edersin, geleceğin bilgisiyle geçmişi eğitmezsin.**

Random split kullanmak burada yanıltıcı olurdu, model geleceği öğrenmiş gibi görünürdü.

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    objective="multiclass",   # H / D / A
    num_class=3,
    class_weight="balanced",  # maçların %46'sı ev sahibi galibiyeti dengesiz
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(50)]
)
```

---

## Katman 3: Kalibrasyon

İki modeli birleştirdim, güzel olasılıklar ürettim. Ama bunlara güvenebilir miyim?

**Kalibrasyon testi:** Modelin %70 dediği maçlarda, takım gerçekten %70 kez kazanıyor mu?

![Kalibrasyon diyagramı açıklaması](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Calibration_curve.svg/400px-Calibration_curve.svg.png)

Çoğu model bu testte başarısız olur. Özellikle tree-based modeller sistematik olarak aşırı güvenir (over-confident).

**Isotonic Regresyon ile Düzeltme:**

```python
from sklearn.isotonic import IsotonicRegression

# Her sınıf için ayrı kalibratör
calibrators = {}
for outcome, col in [("H", "p_home"), ("D", "p_draw"), ("A", "p_away")]:
    cal = IsotonicRegression(out_of_bounds="clip")
    y_bin = (val_df["actual_outcome"] == outcome).astype(int)
    cal.fit(val_df[col], y_bin)
    calibrators[outcome] = cal
```

Isotonic regresyon monoton (düz artan) bir dönüşüm uygular. Model %70 diyorsa ama gerçekte %65 oluyor, bunu %65'e çeker. Basit ama etkili.

**Önce/Sonra Brier skoru:**
- Kalibrasyonsuz: 0.561
- Kalibrasyonlu: 0.538

---

## Monte Carlo Turnuva Simülasyonu

Artık her maç için kalibre olasılıklar var. 25.000 kez turnuvayı baştan sona simüle ettim.

```python
def simulate_full_tournament(all_groups, played, model, n=25_000):
    reach = {}  # her takım için kaç kez hangi aşamaya geldi
    
    for _ in range(n):
        # 1. Grup aşamasını simüle et
        group_results = {}
        for grp, teams in all_groups.items():
            group_results[grp] = simulate_group(teams, played[grp], model)
        
        # 2. İlk 2 + en iyi 8 üçüncüyü belirle
        qualifiers = determine_qualifiers(group_results)
        
        # 3. Eleme turlarını simüle et
        bracket = seed_bracket(qualifiers)
        simulate_knockouts(bracket, model, reach)
    
    # Olasılık = kaç kez ulaştı / toplam simülasyon
    return {team: counts / n for team, counts in reach.items()}
```

Her simülasyon 32 takımın tüm yolculuğunu oynar. 25.000 × ~100 maç = 2.5 milyon sanal maç!

### Wilson Güven Aralıkları

25.000 simülasyondan Türkiye'nin final oynama olasılığı %0.2 çıktı. Bu tahmine ne kadar güvenelim?

**Wilson score interval** (Wilson, 1927):

```python
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """k başarı, n deneme için %95 güven aralığı."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half = (z * (p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return max(0, centre - half), min(1, centre + half)

# Türkiye final: 50/25000
lo, hi = wilson_ci(50, 25_000)
# → (%0.15, %0.26)
```

Normal approximation (ders kitabındaki yöntem) burada işe yaramaz çünkü oran çok küçük. Wilson interval küçük oranlar için çok daha güvenilir.

---

## Sonuçlar: Dürüst Bir Değerlendirme

| Model | Brier Skoru ↓ | Doğruluk ↑ |
|---|---|---|
| **Birleşik Model** | **0.541** | **%55.2** |
| Bayesian Poisson | 0.543 | %56.1 |
| LightGBM | 0.552 | %55.8 |
| Kıyaslama: eşit şans | 0.667 | %33.3 |
| Kıyaslama: hep ev sahibi | 0.648 | %44.6 |

**Brier skoru nedir?** Tahmin vektörü ile gerçek vektör arasındaki ortalama karesel hata. 0 mükemmel, 0.667 tamamen rastgele.

Modelimiz rastgeleden %19 daha iyi. "Hep ev sahibi kazanır" de ki naif stratejiyi de geride bırakıyor.

%55 doğruluk ilk bakışta düşük görünebilir. Ama şunu düşünün:

> *Dünyanın en büyük organizasyonları, milyarlarca veri noktası ve yüzlerce analist ile bu işe giren sistemler bile %57'nin üzerine nadiren çıkıyor.*

Futbol bu. Rastgelelik bu sporun özünde var. Modelimizin değeri kesin kazananı bulmak değil, **olasılıkları doğru kalibre etmek.**

### Türkiye'nin Görünümü

WC2026 başladığında (15 Haziran 2026 itibarıyla) modelimizin hesapları:

| Aşama | Olasılık |
|---|---|
| Grup aşamasından çıkma | %33 |
| Son 16 | %10 |
| Çeyrek Final | %2.6 |
| Yarı Final | %0.7 |
| Şampiyon | %0.05 |

Türkiye Grup D'de (Paraguay, Avustralya, ABD ile) %1 ihtimalle birinci, %9 ihtimalle ikinci bitirebilir. Ama %46 ihtimalle üçüncü bitirecek ve bu grubun en iyi üçüncüsü olursa yine tutarlı. Toplam eleme şansı: **%33**.

---

## Öğrendiklerim

**1. Kalibrasyon doğruluktan önemli.**  
Model doğruluğu sizi yanıltabilir. %55 doğru tahmin eden ama kalibre olmayan model, %52 doğru tahmin eden ama kalibre olan modelden daha az kullanışlıdır. Özellikle olasılıklarla karar alıyorsanız.

**2. Veri sızıntısı (leakage) gizli kalmayı sever.**  
FIFA sıralaması örneği gibi, leakage bazen çok açık değil. Test setinde fazla iyi sonuç görüyorsanız iki kez kontrol edin.

**3. Baseline model kurmadan başlamayın.**  
"Hep ev sahibi kazanır" modelim %44.6 doğrulukla çalışıyor. Bunu geçemeyen bir model geliştirmek zaman kaybı. Her zaman naive baseline'ı önce kurun.

**4. Bayesian düşünmek farklı bir zihin açıyor.**  
Frekantist: "Parametre nedir?" → tek cevap  
Bayesian: "Parametre ne olabilir?" → dağılım  
Bu fark gerçek verilerle çalışırken devasa bir avantaj.

---

## Mimari: Teknik Genel Bakış

```mermaid
flowchart TD
    A[make update] --> B[football-data.org\nWC2026 sonuçları]
    A --> C[Kaggle/GitHub\n49k+ tarihsel maç]
    A --> D[FIFA Ranking\n1992-2024]
    B --> E[data/build.py\ntemizleme · dedup · validate]
    C --> E
    D --> F[ranking_lookup\nleakage-free]
    E --> G[BayesianPoissonModel\nPyMC · MCMC · time-decay]
    E --> H[FeaturePipeline\n23 özellik · Elo · form · FIFA]
    F --> H
    G --> I[EnsembleModel\nBayesian 0.84 + LightGBM 0.16]
    H --> I
    I --> J[IsotonicCalibration]
    J --> K[simulate_full_tournament\n25.000 × MC]
    K --> L[data/outputs/\nStreamlit için]
    L --> M[Streamlit Dashboard\nTürkçe · dark theme]
```

---

## Kodu Görmek İster misiniz?

Tüm kaynak kodu GitHub'da açık:  
**[github.com/tugberkakbulut/wc2026-ai](https://github.com/tugberkakbulut/wc2026-ai)**

Canlı demo:  
**[wc2026-ai.streamlit.app](https://wc2026-ai.streamlit.app)**
---

## Sıradaki Adım: LLM Hibrit Mimarisi

Bu sistem iyi çalışıyor ama kritik bir eksik var: **sakatlık ve kadro bilgisi yok.**

Maç öncesi "Çalhanoğlu yok" haberi modeli tamamen değiştirmeli. Bunu istatistiksel modelle yapamazsın, yapısal olmayan metinden bilgi çıkarman lazım.


İstatistiksel backbone kalır (kalibre, tekrarlanabilir, açıklanabilir).  
LLM agent yapısal olmayan dünyayı sayıya dönüştürür.  
İki güçlü yaklaşım bir arada.

---

## Kaynaklar

- **Dixon & Coles (1997)** — *Modelling Association Football Scores and Inefficiencies in the Football Betting Market* — bu alanın klasiği, temel formülasyonu buradan aldım
- **Maher (1982)** — *Modelling Association Football Scores* — Poisson bağımsızlık varsayımının kökeni
- **Karlis & Ntzoufras (2003)** — *Analysis of sports data by using bivariate Poisson models* — çift Poisson modeli
- **Wilson (1927)** — *Probable inference, the law of succession, and statistical inference* — Wilson CI'nın orijinal makalesi
- **Gelman et al.** — *Bayesian Data Analysis* — Bayesian düşüncenin İncil'i
- PyMC documentation — [docs.pymc.io](https://docs.pymc.io)
- LightGBM paper — Ke et al., NeurIPS 2017

---

*Bu projeyi WC2026 boyunca aktif olarak güncellemeye çalışacağım. Her maçtan sonra `make update` → `git push` → Streamlit otomatik güncelleniyor. 

*Sorularınız veya önerileriniz için yorumları kullanabilirsiniz.*
