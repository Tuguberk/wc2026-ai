"""WC2026 Yapay Zeka Tahmin Sistemi — Streamlit Uygulaması.

Sadece data/outputs/ klasöründen okur, model çalıştırmaz.
Sekmeler: Şampiyonluk | Türkiye 🇹🇷 | Tüm Maçlar | Nasıl Çalışır
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Yollar ───────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
OUTPUTS      = PROJECT_ROOT / "data" / "outputs"

# ── Renk paleti — gece stadyumu teması ────────────────────────────────────────
C_BG      = "#05080f"   # çok koyu lacivert
C_SURFACE = "#0c1321"   # koyu mavi yüzey
C_BORDER  = "#1c2d4a"   # çelik mavi kenar
C_ACCENT  = "#05f26c"   # çim yeşili (canlı)
C_AMBER   = "#ff5722"   # turuncu (Türkiye vurgusu)
C_GOLD    = "#ffc107"   # altın sarısı (şampiyon)
C_TEXT    = "#f0f6ff"   # neredeyse beyaz, hafif mavi ton
C_MUTED   = "#5e7aa8"   # gri-mavi
C_HOME    = "#29b6f6"   # gökyüzü mavisi
C_DRAW    = "#78909c"   # gri
C_AWAY    = "#f44336"   # kırmızı
C_PURPLE  = "#7c4dff"   # mor (grafik varyasyonu)

FONT_DISPLAY = "'Oswald', sans-serif"
FONT_BODY    = "'Inter', sans-serif"
FONT_MONO    = "'JetBrains Mono', monospace"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] {{
    background-color: {C_BG};
    color: {C_TEXT};
    font-family: {FONT_BODY};
}}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px; }}

/* Streamlit native bileşen renkleri */
.stRadio > label, .stSelectbox > label {{ color: {C_MUTED} !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }}
div[data-baseweb="select"] > div {{ background-color: {C_SURFACE} !important; border-color: {C_BORDER} !important; color: {C_TEXT} !important; }}
div[data-baseweb="select"] span {{ color: {C_TEXT} !important; }}
.stRadio div[role="radiogroup"] label span {{ color: {C_TEXT} !important; }}
div[data-testid="stMetricValue"] {{ color: {C_ACCENT}; font-family: {FONT_DISPLAY}; }}
div[data-testid="stMetricLabel"] {{ color: {C_MUTED} !important; font-size: 0.75rem !important; }}

/* Hero kart — üst kenarda canlı degrade çizgi */
.hero-card {{
    background: linear-gradient(160deg, {C_SURFACE} 0%, #0a1730 100%);
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}}
.hero-number {{
    font-family: {FONT_DISPLAY};
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
}}
.hero-label {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {C_MUTED};
    margin-top: 0.4rem;
}}

/* Küçük kart */
.mini-card {{
    background: linear-gradient(160deg, {C_SURFACE} 0%, #091428 100%);
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}}
.mini-number {{
    font-family: {FONT_DISPLAY};
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
    color: {C_ACCENT};
}}
.mini-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {C_MUTED};
    margin-top: 0.35rem;
}}

/* Maç kartı */
.match-card {{
    background: linear-gradient(160deg, {C_SURFACE} 0%, #091428 100%);
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    transition: border-color 0.2s;
}}
.match-card:hover {{ border-color: rgba(5,242,108,0.3); }}
.match-teams {{
    font-family: {FONT_DISPLAY};
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: {C_TEXT};
}}
.match-date {{
    font-size: 0.75rem;
    color: {C_MUTED};
    margin-bottom: 0.5rem;
}}

/* Olasılık çubuğu */
.prob-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}}
.prob-label {{
    width: 130px;
    text-align: right;
    color: {C_MUTED};
    font-size: 0.78rem;
    flex-shrink: 0;
}}
.prob-bar-bg {{
    flex: 1;
    background: rgba(28,45,74,0.6);
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}}
.prob-bar {{ height: 100%; border-radius: 4px; }}
.prob-pct {{
    width: 40px;
    font-family: {FONT_MONO};
    font-size: 0.8rem;
    text-align: right;
    color: {C_TEXT};
    flex-shrink: 0;
}}

/* Bölüm başlığı */
.sec-head {{
    font-family: {FONT_DISPLAY};
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C_ACCENT};
    border-bottom: 1px solid {C_BORDER};
    padding-bottom: 0.35rem;
    margin: 1.6rem 0 0.8rem;
}}

/* Not kutusu */
.note-box {{
    background: rgba(12,19,33,0.8);
    border-left: 3px solid rgba(5,242,108,0.35);
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: {C_MUTED};
    margin: 0.3rem 0 0.8rem;
    line-height: 1.6;
}}

/* Sekmeler */
.stTabs [data-baseweb="tab-list"] {{ background: transparent; gap: 0.2rem; border-bottom: 1px solid {C_BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: {FONT_DISPLAY};
    font-size: 0.88rem;
    letter-spacing: 0.06em;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    color: {C_MUTED};
    padding: 0.5rem 1.1rem;
    transition: color 0.15s;
}}
.stTabs [aria-selected="true"] {{
    background: transparent;
    color: {C_ACCENT};
    border-bottom: 2px solid {C_ACCENT};
}}

/* Dataframe */
.stDataFrame {{ font-family: {FONT_MONO}; font-size: 0.8rem; }}
[data-testid="stDataFrameResizable"] {{ background: {C_SURFACE} !important; }}

#MainMenu, footer, header {{ visibility: hidden; }}
</style>
"""


# ── Veri yükleyiciler ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame | None:
    p = OUTPUTS / "latest_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_fixtures() -> pd.DataFrame | None:
    p = OUTPUTS / "wc2026_fixtures.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_calibration() -> dict:
    p = OUTPUTS / "calibration_latest.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data(ttl=300)
def load_turkey_path() -> pd.DataFrame | None:
    p = OUTPUTS / "turkey_path.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_timeseries() -> pd.DataFrame | None:
    p = OUTPUTS / "probability_timeseries.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_model_metrics() -> dict:
    p = OUTPUTS / "model_metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}

@st.cache_data(ttl=300)
def load_feature_importance() -> pd.DataFrame | None:
    p = OUTPUTS / "feature_importance.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_bracket_results() -> pd.DataFrame | None:
    p = OUTPUTS / "bracket_results.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data(ttl=300)
def load_calib_records() -> pd.DataFrame | None:
    snaps_dir = PROJECT_ROOT / "data" / "snapshots"
    if not snaps_dir.exists():
        return None
    frames = [
        pd.read_parquet(snap / "calibration.parquet")
        for snap in sorted(snaps_dir.iterdir())
        if (snap / "calibration.parquet").exists()
    ]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    # Her maç için en erken snapshot'ı tut (yinelenen satırları kaldır)
    return combined.sort_values("snapshot_ts").drop_duplicates(subset=["match_id"]).reset_index(drop=True)


# ── HTML yardımcıları ─────────────────────────────────────────────────────────

def not_kutusu(metin: str) -> None:
    st.markdown(f'<div class="note-box">{metin}</div>', unsafe_allow_html=True)

def bolum(baslik: str) -> None:
    st.markdown(f'<div class="sec-head">{baslik}</div>', unsafe_allow_html=True)

def olasilik_cubugu(p_ev: float, p_ber: float, p_dep: float, ev: str, dep: str) -> str:
    def satir(p, renk, etiket):
        return (
            f'<div class="prob-row">'
            f'<span class="prob-label">{etiket[:18]}</span>'
            f'<div class="prob-bar-bg">'
            f'<div class="prob-bar" style="width:{p*100:.1f}%;background:{renk}"></div>'
            f'</div>'
            f'<span class="prob-pct">{p:.0%}</span>'
            f'</div>'
        )
    return (
        satir(p_ev,  C_HOME, ev)
        + satir(p_ber, C_DRAW, "Beraberlik")
        + satir(p_dep, C_AWAY, dep)
    )

def plotly_tema(**kw) -> dict:
    ax = dict(
        gridcolor="rgba(28,45,74,0.6)",
        zerolinecolor="rgba(28,45,74,0.8)",
        tickfont=dict(color=C_MUTED, size=11),
    )
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,15,0.4)",
        font=dict(family=FONT_BODY, color=C_TEXT, size=12),
        xaxis=dict(**ax),
        yaxis=dict(**ax),
        margin=dict(l=10, r=20, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT)),
    )
    for k in ("xaxis", "yaxis"):
        if k in kw:
            base[k] = {**base[k], **kw.pop(k)}
    base.update(kw)
    return base


# ── Sayfa başlığı ─────────────────────────────────────────────────────────────

def baslik_goster() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0c1a3a 0%, #071020 60%, {C_BG} 100%);
            border: 1px solid {C_BORDER};
            border-radius: 14px;
            padding: 1.4rem 1.8rem 1.2rem;
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        ">
            <!-- parlak nokta efekti -->
            <div style="
                position:absolute;top:-40px;right:-40px;
                width:180px;height:180px;
                background:radial-gradient(circle, rgba(5,242,108,0.12) 0%, transparent 70%);
                border-radius:50%;
            "></div>
            <div style="display:flex;align-items:center;gap:0.7rem;position:relative">
                <span style="font-size:2rem">⚽</span>
                <div>
                    <div style="display:flex;align-items:baseline;gap:0.4rem">
                        <span style="font-family:{FONT_DISPLAY};font-size:2rem;font-weight:700;
                                     letter-spacing:0.03em;color:{C_TEXT}">WC 2026</span>
                        <span style="font-family:{FONT_DISPLAY};font-size:1.6rem;font-weight:400;
                                     color:{C_ACCENT}">YAPAY ZEKA TAHMİNLERİ</span>
                    </div>
                    <p style="color:{C_MUTED};font-size:0.76rem;margin:0.15rem 0 0">
                        Bayesian + Makine Öğrenmesi + Piyasa Tahminleri &nbsp;·&nbsp; Her maçtan sonra güncellenir
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sekme 1: Şampiyonluk Yarışı ───────────────────────────────────────────────

def sekme_sampiyonluk(bracket: pd.DataFrame | None) -> None:

    if bracket is None or bracket.empty:
        st.info("Şampiyonluk olasılıkları için `make update` çalıştırın.")
        return

    bolum("Dünya Kupası'nı kim kazanacak?")
    not_kutusu(
        "Bu yüzdeler, 32 takımlı turnuvanın 25.000 kez simüle edilmesinden elde edilmektedir. "
        "Her simülasyon grup aşamasından finale kadar tüm maçları oynar. "
        "Sayılar, her takımın bu simülasyonlarda kaç kez şampiyon olduğunu gösterir."
    )

    ilk10 = bracket.head(10).copy()

    # Sıralamaya göre renk: altın → yeşil → mavi → gri
    renk_skala = [C_GOLD, "#c8f500", C_ACCENT, C_HOME, C_HOME, C_HOME,
                  "#29b6f6", "#29b6f6", "#5e7aa8", "#5e7aa8"]
    renkler = [renk_skala[min(i, len(renk_skala)-1)] for i in range(len(ilk10))]

    fig = go.Figure(go.Bar(
        x=ilk10["p_champion"] * 100,
        y=ilk10["team"],
        orientation="h",
        marker=dict(
            color=renkler,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=[f"  %{p*100:.1f}" for p in ilk10["p_champion"]],
        textfont=dict(color=C_TEXT, size=13, family=FONT_DISPLAY),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Şampiyonluk şansı: %%{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="🏆  Şampiyonluk Olasılığı — İlk 10 Takım",
            font=dict(color=C_TEXT, size=14, family=FONT_DISPLAY),
        ),
        **plotly_tema(
            xaxis=dict(
                title="Turnuvayı kazanma ihtimali (%)",
                range=[0, ilk10["p_champion"].max() * 155],
            ),
            yaxis=dict(autorange="reversed", tickfont=dict(size=13, family=FONT_DISPLAY)),
            height=400,
            bargap=0.28,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    bolum("Tüm Takımlar — Tam Turnuva Olasılıkları")
    not_kutusu(
        "Her sütun, o aşamaya ulaşma olasılığını gösterir. "
        "'Grup' = grup aşamasından çıkma. Sütunlar toplanarak 100 yapmaz — "
        "birçok takım aynı aşamaya ulaşabilir."
    )

    ci_var = "p_champion_lo" in bracket.columns
    satirlar = []
    for _, r in bracket.iterrows():
        def yuzde(col):
            v = r.get(col, float("nan"))
            return f"%{v*100:.1f}" if pd.notna(v) and v > 0.0001 else "—"

        def yuzde_ci(col):
            v = r.get(col, float("nan"))
            if pd.isna(v) or v < 0.0001:
                return "—"
            if ci_var:
                lo = r.get(f"{col}_lo", v)
                hi = r.get(f"{col}_hi", v)
                return f"%{v*100:.1f}  (%{lo*100:.1f}–%{hi*100:.1f})"
            return f"%{v*100:.1f}"

        satirlar.append({
            "Takım":        r["team"],
            "Grup":         yuzde("p_advance"),
            "Son 16":       yuzde("p_r16"),
            "Çeyrek Final": yuzde("p_qf"),
            "Yarı Final":   yuzde("p_sf"),
            "Final":        yuzde("p_final"),
            "Şampiyon":     yuzde_ci("p_champion"),
        })

    st.dataframe(pd.DataFrame(satirlar), use_container_width=True, hide_index=True, height=500)
    if ci_var:
        not_kutusu(
            "Şampiyon sütunundaki parantez içindeki aralık, gerçek olasılığın "
            "neredeyse kesinlikle içinde yer aldığı %95 güven aralığıdır."
        )


# ── Sekme 2: Türkiye ──────────────────────────────────────────────────────────

def sekme_turkiye(
    calib: dict,
    turkiye_yolu: pd.DataFrame | None,
    zaman_serisi: pd.DataFrame | None,
    tahminler: pd.DataFrame | None,
    bracket: pd.DataFrame | None,
) -> None:
    turkey_probs = calib.get("turkey_probs", {})

    # Bracket simülasyonundan Türkiye satırını bul (en doğru kaynak)
    tr_bracket_row = None
    if bracket is not None and not bracket.empty:
        tr_rows = bracket[bracket["team"].str.contains("Turk", case=False, na=False)]
        if not tr_rows.empty:
            tr_bracket_row = tr_rows.iloc[0]

    # Eleme ihtimali: bracket p_advance (en iyi 3. dahil) → JSON p_advance_full → JSON p_advance
    if tr_bracket_row is not None:
        ilerle = tr_bracket_row.get("p_advance")
    else:
        ilerle = turkey_probs.get("p_advance_full") or turkey_probs.get("p_advance")

    # ── Eleme hesabı ─────────────────────────────────────────────────────
    p1       = turkey_probs.get("p_1st", 0) or 0
    p2       = turkey_probs.get("p_2nd", 0) or 0
    p3       = turkey_probs.get("p_3rd", 0) or 0
    p_ilerle = float(ilerle) if ilerle else 0
    p_en_iyi_3 = max(0.0, p_ilerle - p1 - p2)

    # ── Satır 1: Hero kart + 4 pozisyon kartı yan yana ───────────────────
    col_hero, col_k1, col_k2, col_k3, col_k4 = st.columns([2, 1, 1, 1, 1])

    with col_hero:
        deger = f"%{p_ilerle*100:.0f}" if p_ilerle else "—"
        st.markdown(
            f'<div class="hero-card" style="border-top:4px solid {C_AMBER};height:100%">'
            f'<div class="hero-number" style="color:{C_AMBER}">{deger}</div>'
            f'<div class="hero-label">Grup aşamasından<br>çıkma şansı</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    for kol, (anahtar, etiket, renk, alt) in zip(
        [col_k1, col_k2, col_k3, col_k4],
        [
            ("p_1st", "1. Sıra", C_ACCENT, "otomatik çıkar ✓"),
            ("p_2nd", "2. Sıra", C_ACCENT, "otomatik çıkar ✓"),
            ("p_3rd", "3. Sıra", C_GOLD,   "en iyi 8'e girerse çıkar ~"),
            ("p_4th", "4. Sıra", C_AWAY,   "elenir ✗"),
        ],
    ):
        v = turkey_probs.get(anahtar, float("nan"))
        s = f"%{v*100:.0f}" if not np.isnan(v) else "—"
        kol.markdown(
            f'<div class="mini-card">'
            f'<div class="mini-number" style="color:{renk}">{s}</div>'
            f'<div class="mini-label">{etiket}</div>'
            f'<div style="font-size:0.63rem;color:{C_MUTED};margin-top:0.25rem;line-height:1.4">{alt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Satır 2: Eleme dökümü (tam genişlik) ─────────────────────────────
    oran_3 = (p_en_iyi_3 / p3 * 100) if p3 else 0
    st.markdown(
        f"""
        <div style="background:rgba(12,19,33,0.7);border:1px solid {C_BORDER};
                    border-radius:8px;padding:0.8rem 1.2rem;margin-top:0.6rem;
                    display:flex;align-items:center;gap:2rem;flex-wrap:wrap;
                    font-size:0.8rem">
            <div style="color:{C_MUTED};font-size:0.7rem;text-transform:uppercase;
                        letter-spacing:0.08em;flex-shrink:0">Eleme dökümü</div>
            <div>
                <span style="color:{C_ACCENT};font-family:{FONT_MONO}">%{p1*100:.0f}</span>
                <span style="color:{C_MUTED}"> 1. +</span>
                <span style="color:{C_ACCENT};font-family:{FONT_MONO}"> %{p2*100:.0f}</span>
                <span style="color:{C_MUTED}"> 2. = </span>
                <span style="color:{C_TEXT}"><b>%{(p1+p2)*100:.0f} otomatik</b></span>
            </div>
            <div>
                <span style="color:{C_GOLD};font-family:{FONT_MONO}">+%{p_en_iyi_3*100:.0f}</span>
                <span style="color:{C_MUTED}"> en iyi 3. olarak &nbsp;(3. bitirenlerin ~%{oran_3:.0f}'i)</span>
            </div>
            <div style="margin-left:auto;font-family:{FONT_DISPLAY};font-size:1.1rem">
                <span style="color:{C_MUTED}">= </span>
                <span style="color:{C_AMBER};font-weight:700">%{p_ilerle*100:.0f} toplam</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Turnuva yolu ──────────────────────────────────────────────────────
    bolum("Türkiye ne kadar ileri gidebilir?")
    not_kutusu(
        "Bu olasılıklar 25.000 simülasyonun ortalamasıdır. "
        "Türkiye'nin 1., 2. veya en iyi 3. olarak ilerlemesi dahil tüm yollar hesaba katılmıştır."
    )

    if tr_bracket_row is not None:
        yol_map: dict[str, float | None] = {
            "Grup Aşaması": tr_bracket_row.get("p_advance"),
            "Son 16":       tr_bracket_row.get("p_r16"),
            "Çeyrek Final": tr_bracket_row.get("p_qf"),
            "Yarı Final":   tr_bracket_row.get("p_sf"),
            "Final":        tr_bracket_row.get("p_final"),
            "Şampiyon":     tr_bracket_row.get("p_champion"),
        }
    else:
        yol_map = {
            "Grup Aşaması": turkey_probs.get("p_advance_full") or turkey_probs.get("p_advance"),
            "Son 16":       turkey_probs.get("round_of_16"),
            "Çeyrek Final": turkey_probs.get("quarter_final"),
            "Yarı Final":   turkey_probs.get("semi_final"),
            "Final":        turkey_probs.get("final"),
            "Şampiyon":     turkey_probs.get("champion"),
        }

    asamalar = [s for s, p in yol_map.items() if p is not None and not np.isnan(p)]
    olasiliklar = [yol_map[s] for s in asamalar]

    if asamalar:
        # Aşama renklerini ilerledikçe altına doğru ısıt
        _asama_renkler = ["#29b6f6", "#05f26c", "#ffc107", "#ff9800", C_AMBER, "#f44336"]
        _bar_renkler = [_asama_renkler[min(i, len(_asama_renkler)-1)]
                        for i, p in enumerate(olasiliklar)]
        fig = go.Figure(go.Bar(
            x=asamalar,
            y=[p * 100 for p in olasiliklar],
            marker=dict(color=_bar_renkler, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"%{p*100:.1f}" for p in olasiliklar],
            textfont=dict(color=C_TEXT, size=13, family=FONT_DISPLAY),
            textposition="outside",
            hovertemplate="%{x}: %%{y:.1f}<extra></extra>",
        ))
        fig.update_layout(
            **plotly_tema(
                yaxis=dict(
                    title="Olasılık (%)",
                    range=[0, max(olasiliklar) * 155 if olasiliklar else 1],
                ),
                height=340,
                bargap=0.3,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sonraki maç ───────────────────────────────────────────────────────
    if tahminler is not None and not tahminler.empty:
        tr_tahmin = tahminler[
            tahminler["home_team"].str.contains("Turk", case=False, na=False)
            | tahminler["away_team"].str.contains("Turk", case=False, na=False)
        ]
        if not tr_tahmin.empty:
            bolum("Sonraki Maç")
            nxt = tr_tahmin.iloc[0]
            ev, dep = nxt["home_team"], nxt["away_team"]
            ph = float(nxt["p_home"])
            pd_ = float(nxt["p_draw"])
            pa = float(nxt["p_away"])
            eh = float(nxt.get("exp_home_goals", 0))
            ea = float(nxt.get("exp_away_goals", 0))

            favori = ev if ph > pa else dep
            fav_p = max(ph, pa)

            st.markdown(
                f'<div class="match-card">'
                f'<div class="match-teams">{ev} <span style="color:{C_MUTED}">-</span> {dep}</div>'
                f'<div style="margin:0.5rem 0">',
                unsafe_allow_html=True,
            )
            st.markdown(olasilik_cubugu(ph, pd_, pa, ev, dep), unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-size:0.78rem;color:{C_MUTED};margin-top:0.4rem">'
                f'Beklenen skor: {eh:.1f} – {ea:.1f} &nbsp;·&nbsp; '
                f'Model favorisi: <b style="color:{C_TEXT}">{favori}</b> (%{fav_p*100:.0f})'
                f'</p></div>',
                unsafe_allow_html=True,
            )
            not_kutusu(
                "Bu olasılıklar, 8 yıllık uluslararası maç verisiyle eğitilmiş "
                "istatistiksel bir modeli ve piyasa tahminlerini birleştirmektedir. "
                "Son güncellemeden sonraki sakatlık veya kadro haberleri yansıtılmamıştır."
            )

    # ── Grup D karşılaştırması ────────────────────────────────────────────
    if turkiye_yolu is not None and not turkiye_yolu.empty:
        bolum("Grup D — Tüm Takımlar")
        not_kutusu("25.000 simüle edilmiş grup aşamasının ortalaması.")
        sutunlar = ["team", "p_1st", "p_2nd", "p_advance", "avg_points", "avg_gd"]
        mevcut = [c for c in sutunlar if c in turkiye_yolu.columns]
        yeniad = {
            "team": "Takım", "p_1st": "1. Bitirme", "p_2nd": "2. Bitirme",
            "p_advance": "Çıkma Şansı", "avg_points": "Ort. Puan", "avg_gd": "Ort. Averaj",
        }
        goster = turkiye_yolu[mevcut].copy()
        for c in ["p_1st", "p_2nd", "p_advance"]:
            if c in goster:
                goster[c] = goster[c].apply(lambda x: f"%{x*100:.0f}")
        for c in ["avg_points", "avg_gd"]:
            if c in goster:
                goster[c] = goster[c].apply(lambda x: f"{x:.1f}")
        st.dataframe(goster.rename(columns=yeniad), use_container_width=True, hide_index=True)

    # ── Zaman serisi ──────────────────────────────────────────────────────
    if (zaman_serisi is not None and not zaman_serisi.empty
            and "turkey_advance" in zaman_serisi.columns
            and zaman_serisi["turkey_advance"].notna().sum() > 1):
        bolum("Türkiye'nin görünümü zaman içinde nasıl değişti?")
        not_kutusu("Her nokta bir model güncellemesidir. Maç sonuçları geldikçe olasılıklar güncellenir.")
        fig2 = go.Figure(go.Scatter(
            x=zaman_serisi["snapshot_ts"],
            y=zaman_serisi["turkey_advance"] * 100,
            mode="lines+markers",
            line=dict(color=C_AMBER, width=2),
            marker=dict(size=7, color=C_AMBER),
            hovertemplate="%{x}<br>Çıkma: %%{y:.1f}<extra></extra>",
        ))
        fig2.update_layout(
            **plotly_tema(yaxis=dict(title="Çıkma olasılığı (%)"))
        )
        st.plotly_chart(fig2, use_container_width=True)


# ── Sekme 3: Tüm Maçlar ──────────────────────────────────────────────────────

def sekme_maclar(tahminler: pd.DataFrame | None, fikstür: pd.DataFrame | None) -> None:
    if tahminler is None or tahminler.empty:
        st.info("Henüz tahmin bulunmuyor. `make update` çalıştırın.")
        return

    df = tahminler.copy()
    if fikstür is not None and not fikstür.empty:
        fix = fikstür[["match_id", "date", "stage", "group", "status",
                        "home_score", "away_score"]].copy()
        df = df.merge(fix, on="match_id", how="left")

    df = df.sort_values("date") if "date" in df.columns else df

    # Filtreler
    f1, f2 = st.columns([1, 2])
    with f1:
        durum_filtre = st.radio("Göster", ["Tümü", "Yaklaşan", "Tamamlanan"], horizontal=True)
    with f2:
        gruplar = (["Tüm gruplar"] + sorted(df["group"].dropna().unique().tolist())
                   if "group" in df.columns else ["Tüm gruplar"])
        grup_filtre = st.selectbox("Grup", gruplar)

    if durum_filtre == "Yaklaşan" and "status" in df.columns:
        df = df[df["status"] != "FINISHED"]
    elif durum_filtre == "Tamamlanan" and "status" in df.columns:
        df = df[df["status"] == "FINISHED"]

    if grup_filtre != "Tüm gruplar" and "group" in df.columns:
        df = df[df["group"] == grup_filtre]

    if df.empty:
        st.info("Bu filtre için maç bulunamadı.")
        return

    not_kutusu(
        "Yüzdeler modelin maç öncesi tahminini gösterir. "
        "Tamamlanan maçlarda gerçek sonuç ve modelin doğru tahmin edip etmediği görünür."
    )

    for _, satir in df.iterrows():
        ev  = satir["home_team"]
        dep = satir["away_team"]
        ph  = float(satir["p_home"])
        pd_ = float(satir["p_draw"])
        pa  = float(satir["p_away"])
        durum = str(satir.get("status", "SCHEDULED"))
        tarih = str(satir.get("date", ""))[:10]
        asama = str(satir.get("stage", satir.get("group", "")))

        model_tahmini = ("Ev kazanır" if ph > max(pd_, pa)
                         else ("Beraberlik" if pd_ > max(ph, pa) else "Deplasman kazanır"))
        favori = ev if ph > max(pd_, pa) else (None if pd_ > max(ph, pa) else dep)
        fav_p  = max(ph, pd_, pa)

        if durum == "FINISHED":
            hs  = int(satir.get("home_score", 0))
            as_ = int(satir.get("away_score", 0))
            gercek = ("Ev kazandı" if hs > as_
                      else ("Beraberlik" if hs == as_ else "Deplasman kazandı"))
            dogru  = model_tahmini.split()[0] in gercek
            rozet  = (
                f'<span style="color:{C_ACCENT};font-size:0.78rem">✓ Doğru tahmin</span>'
                if dogru else
                f'<span style="color:{C_AWAY};font-size:0.78rem">✗ Tahmin: {model_tahmini}</span>'
            )
            baslik = (
                f'<div class="match-card">'
                f'<div class="match-date">{tarih} · {asama}</div>'
                f'<div class="match-teams">'
                f'{ev} <span style="color:{C_ACCENT};font-size:1rem;font-weight:700">{hs} – {as_}</span> {dep}'
                f'</div>'
                f'<div style="margin:0.3rem 0 0.6rem;font-size:0.8rem">{rozet}</div>'
            )
        else:
            fav_str = (f"{favori} favorisi (%{fav_p*100:.0f})" if favori
                       else f"Beraberlik olası (%{pd_*100:.0f})")
            baslik = (
                f'<div class="match-card">'
                f'<div class="match-date">{tarih} · {asama}</div>'
                f'<div class="match-teams">{ev} <span style="color:{C_MUTED}">-</span> {dep}</div>'
                f'<div style="margin:0.3rem 0 0.6rem;font-size:0.78rem;color:{C_MUTED}">{fav_str}</div>'
            )

        st.markdown(
            baslik + olasilik_cubugu(ph, pd_, pa, ev, dep) + "</div>",
            unsafe_allow_html=True,
        )


# ── Sekme 4: Nasıl Çalışır ────────────────────────────────────────────────────

def sekme_teknik(
    model_metrikleri: dict,
    onem_sirasi: pd.DataFrame | None,
    kalibrasyon_kayitlari: pd.DataFrame | None,
    bracket: pd.DataFrame | None,
) -> None:

    # ── Sade dille doğruluk özeti ─────────────────────────────────────────
    bolum("Model ne kadar doğru?")

    if model_metrikleri:
        ens    = model_metrikleri.get("ensemble", {})
        acc    = ens.get("accuracy")
        n      = model_metrikleri.get("n_test", 0)
        brier  = ens.get("brier")
        b_rand = model_metrikleri.get("baselines", {}).get("equal_odds", {}).get("brier", 0.667)
        iyilesme = (b_rand - brier) / b_rand * 100 if brier else 0

        if acc:
            c1, c2, c3 = st.columns(3)
            n_calib = len(kalibrasyon_kayitlari) if kalibrasyon_kayitlari is not None else 0
            for kol, deger, etiket, altetiket in [
                (c1, f"%{acc*100:.0f}", "Doğru tahmin oranı", f"{n:,} test maçında"),
                (c2, f"%{iyilesme:.0f}", "Rastgeleden daha iyi", "Brier skoru iyileşmesi"),
                (c3, str(n_calib), "Değerlendirilen WC2026 maçı", "Gerçek sonuçla karşılaştırıldı"),
            ]:
                kol.markdown(
                    f'<div class="mini-card">'
                    f'<div class="mini-number">{deger}</div>'
                    f'<div class="mini-label">{etiket}<br>'
                    f'<span style="font-size:0.65rem">{altetiket}</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    not_kutusu(
        "Futbol doğası gereği öngörülemezdir. Dünyanın en iyi modelleri bile kazanan/beraberlik/kaybeden "
        "tahmininde ancak %55–58 oranında doğru olabilmektedir. Bizim modelimiz bu aralıktadır. "
        "Rastgeleden (%33) ve her zaman ev sahibini seçmekten (%45) belirgin biçimde iyidir. "
        "En önemli şey olasılıkların iyi kalibre edilmiş olmasıdır — %60 dediğimizde gerçekten ~%60 olmalı."
    )

    # ── Model karşılaştırması ──────────────────────────────────────────────
    bolum("Model karşılaştırması")
    not_kutusu(
        "Brier skoru olasılık tahminlerinin ne kadar yanlış olduğunu ölçer — düşük daha iyi, 0 mükemmel. "
        "Doğruluk ise modelin en yüksek olasılık verdiği sonucun gerçekten gerçekleşme oranıdır. "
        "Ölçümler modelin hiç görmediği son 1 yıllık tarihsel veriye dayanır."
    )

    if model_metrikleri:
        model_sirasi    = ["bayesian", "lgbm", "ensemble"]
        baseline_sirasi = ["equal_odds", "always_home", "base_rate"]
        model_etiket = {
            "bayesian": "Bayesian Poisson",
            "lgbm":     "Makine Öğrenmesi (LightGBM)",
            "ensemble": "Birleşik Model (son tahmin)",
        }
        baseline_etiket = {
            "equal_odds":  "Kıyaslama: eşit şans (⅓)",
            "always_home": "Kıyaslama: hep ev sahibi",
            "base_rate":   "Kıyaslama: tarihsel oran",
        }

        satirlar = []
        for anahtar in model_sirasi:
            m = model_metrikleri.get(anahtar)
            if m:
                satirlar.append({
                    "Model": model_etiket.get(anahtar, anahtar),
                    "Brier Skoru ↓": f"{m['brier']:.4f}",
                    "Doğruluk ↑":    f"%{m['accuracy']*100:.1f}",
                    "_b": m["brier"], "_model": True,
                })
        for anahtar in baseline_sirasi:
            m = (model_metrikleri.get("baselines") or {}).get(anahtar)
            if m:
                satirlar.append({
                    "Model": baseline_etiket.get(anahtar, anahtar),
                    "Brier Skoru ↓": f"{m['brier']:.4f}",
                    "Doğruluk ↑":    f"%{m['accuracy']*100:.1f}",
                    "_b": m["brier"], "_model": False,
                })

        if satirlar:
            renkler = [
                C_ACCENT if r["_model"] and "Birleşik" in r["Model"] else
                (C_HOME if r["_model"] else C_MUTED)
                for r in satirlar
            ]
            fig = go.Figure(go.Bar(
                x=[r["Model"] for r in satirlar],
                y=[r["_b"] for r in satirlar],
                marker_color=renkler,
                text=[r["Brier Skoru ↓"] for r in satirlar],
                textfont=dict(color=C_TEXT),
                textposition="outside",
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text="Brier Skoru — düşük daha iyi (0 = mükemmel, 0.667 = rastgele)",
                    font=dict(color=C_TEXT),
                ),
                **plotly_tema(
                    yaxis=dict(range=[0, max(r["_b"] for r in satirlar) * 1.3]),
                    xaxis=dict(tickangle=-15),
                    height=340,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            df_m = pd.DataFrame(
                [{k: v for k, v in r.items() if not k.startswith("_")} for r in satirlar]
            )
            st.dataframe(df_m, use_container_width=True, hide_index=True)

    # ── Kalibrasyon diyagramı ──────────────────────────────────────────────
    bolum("Olasılıklara güvenebilir miyiz?")
    not_kutusu(
        "Bu grafik modelin dürüstlüğünü test eder. "
        "Bir takıma %70 şans verdiğimizde, o takım gerçekten ~%70 kez mi kazanıyor? "
        "Mükemmel model noktalı çizgiyi izler. Üstte = hafife almış, altta = abartmış demektir."
    )

    n_calib = len(kalibrasyon_kayitlari) if kalibrasyon_kayitlari is not None else 0
    MIN_CALIB = 20  # diyagram için gereken minimum maç

    if n_calib < MIN_CALIB:
        tamamlandi = n_calib
        kalan = MIN_CALIB - tamamlandi
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {C_SURFACE} 0%, #091428 100%);
                border: 1px solid {C_BORDER};
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin: 0.5rem 0 1rem;
            ">
                <div style="font-size:2.5rem;margin-bottom:0.6rem">⏳</div>
                <div style="font-family:{FONT_DISPLAY};font-size:1.1rem;color:{C_TEXT};margin-bottom:0.4rem">
                    Turnuva devam ediyor — grafik dolmaya başlıyor
                </div>
                <div style="font-size:0.82rem;color:{C_MUTED};max-width:480px;margin:0 auto;line-height:1.7">
                    Şu ana kadar <b style="color:{C_ACCENT}">{tamamlandi}</b> WC2026 maçı tamamlandı ve tahminlerimizle karşılaştırıldı.
                    Anlamlı bir diyagram için en az <b style="color:{C_TEXT}">{MIN_CALIB}</b> maça ihtiyaç var
                    (~{kalan} maç daha).
                    Grup aşaması ilerledikçe bu grafik otomatik olarak güncellenecek.
                </div>
                <div style="margin-top:1rem">
                    <div style="display:inline-block;background:{C_BORDER};border-radius:100px;
                                height:8px;width:240px;overflow:hidden">
                        <div style="height:100%;width:{min(100, tamamlandi/MIN_CALIB*100):.0f}%;
                                    background:linear-gradient(90deg,{C_HOME},{C_ACCENT});
                                    border-radius:100px"></div>
                    </div>
                    <div style="font-size:0.72rem;color:{C_MUTED};margin-top:0.4rem">
                        {tamamlandi} / {MIN_CALIB} maç
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        n_aralik = 10
        sinirlar = np.linspace(0, 1, n_aralik + 1)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=C_MUTED, dash="dash", width=1),
            name="Mükemmel kalibrasyon",
        ))
        for sonuc, sutun, renk, etiket in [
            ("H", "p_home", C_HOME, "Ev sahibi kazanır"),
            ("D", "p_draw", C_DRAW, "Beraberlik"),
            ("A", "p_away", C_AWAY, "Deplasman kazanır"),
        ]:
            tahmin_pts, gercek_pts, sayilar = [], [], []
            for i in range(n_aralik):
                maske = (
                    (kalibrasyon_kayitlari[sutun] >= sinirlar[i])
                    & (kalibrasyon_kayitlari[sutun] < sinirlar[i + 1])
                )
                alt = kalibrasyon_kayitlari[maske]
                if len(alt) < 3:
                    continue
                tahmin_pts.append(float(alt[sutun].mean()))
                gercek_pts.append(float((alt["actual_outcome"] == sonuc).mean()))
                sayilar.append(len(alt))

            fig_r.add_trace(go.Scatter(
                x=tahmin_pts, y=gercek_pts,
                mode="lines+markers",
                line=dict(color=renk, width=2),
                marker=dict(size=[max(5, min(18, c * 2)) for c in sayilar], color=renk),
                name=etiket,
                customdata=sayilar,
                hovertemplate="Tahmin: %{x:.0%}<br>Gerçek: %{y:.0%}<br>n=%{customdata}<extra></extra>",
            ))

        fig_r.update_layout(
            title=dict(text="Kalibrasyon Diyagramı — tahminler gerçeği ne kadar yansıtıyor?", font=dict(color=C_TEXT)),
            **plotly_tema(
                xaxis=dict(title="Tahmin edilen olasılık", tickformat=".0%", range=[0, 1]),
                yaxis=dict(title="Gerçekleşme oranı",     tickformat=".0%", range=[0, 1]),
                height=380,
            ),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        not_kutusu(f"{n_calib} WC2026 maç tahmini gerçek sonuçlarla karşılaştırılmıştır.")

    # ── Özellik önem sırası ───────────────────────────────────────────────
    if onem_sirasi is not None and not onem_sirasi.empty:
        bolum("Model hangi sinyalleri kullanıyor?")
        not_kutusu(
            "Makine öğrenmesi modeli 23 farklı sinyalden öğrenir. "
            "'Kazanım', her sinyalin tahmin hatasını ne kadar azalttığını gösterir — "
            "yüksek = modelin kararları için daha önemli."
        )
        tr_etiketler = {
            "elo_diff":             "Elo puan farkı",
            "home_elo":             "Ev sahibi Elo puanı",
            "away_elo":             "Deplasman Elo puanı",
            "home_win_rate":        "Ev sahibi son form (galibiyet oranı)",
            "away_win_rate":        "Deplasman son form (galibiyet oranı)",
            "home_gd":              "Ev sahibi averaj (son 5 maç)",
            "away_gd":              "Deplasman averaj (son 5 maç)",
            "home_gf":              "Ev sahibi atılan gol (son 5 maç)",
            "away_gf":              "Deplasman atılan gol (son 5 maç)",
            "home_ga":              "Ev sahibi yenilen gol (son 5 maç)",
            "away_ga":              "Deplasman yenilen gol (son 5 maç)",
            "fifa_rank_diff":       "FIFA sıralama farkı",
            "home_fifa_rank":       "Ev sahibi FIFA sırası",
            "away_fifa_rank":       "Deplasman FIFA sırası",
            "home_fifa_pts":        "Ev sahibi FIFA puanı",
            "away_fifa_pts":        "Deplasman FIFA puanı",
            "fifa_pts_diff":        "FIFA puan farkı",
            "tournament_importance":"Turnuva önemi",
            "is_neutral":           "Tarafsız saha",
            "home_rest":            "Ev sahibi dinlenme günü",
            "away_rest":            "Deplasman dinlenme günü",
            "home_draw_rate":       "Ev sahibi beraberlik oranı",
            "away_draw_rate":       "Deplasman beraberlik oranı",
        }
        ilk15 = onem_sirasi.head(15).copy()
        ilk15["feature"] = ilk15["feature"].map(lambda f: tr_etiketler.get(f, f))
        fig_fi = go.Figure(go.Bar(
            x=ilk15["importance"],
            y=ilk15["feature"],
            orientation="h",
            marker_color=C_ACCENT,
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        ))
        fig_fi.update_layout(
            title=dict(text="Özellik Önem Sırası (Kazanım)", font=dict(color=C_TEXT)),
            **plotly_tema(
                xaxis=dict(title="Önem (kazanım)"),
                yaxis=dict(autorange="reversed"),
                height=max(300, len(ilk15) * 30),
            ),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Dürüst kısıtlamalar ───────────────────────────────────────────────
    bolum("Bu modelin yapamadıkları")
    st.markdown(
        f"""
<div style="font-family:{FONT_BODY};line-height:1.85;font-size:0.88rem;max-width:700px;color:{C_TEXT}">

<p><b style="color:{C_AMBER}">Sakatlık ve kadro haberi bilgisi yok.</b><br>
Bir maç öncesinde önemli bir oyuncu sakatlanır ya da cezalı olursa model bunu bilmez.
Bu, her istatistiksel tahmin sisteminin en büyük hata kaynağıdır.</p>

<p><b style="color:{C_AMBER}">Knockout bracket eşleşmeleri yaklaşıktır.</b><br>
FIFA'nın 32 takımlı eleme aşaması için resmi çekiliş tablosu bu modelin yapım aşamasında
henüz yayınlanmamıştı. Çeyrek final, yarı final ve final olasılıkları kaba bir tahmin
olarak değerlendirilmelidir.</p>

<p><b style="color:{C_AMBER}">Futbol doğası gereği öngörülemezdir.</b><br>
Büyük veri altyapısına sahip profesyonel tahmin sistemleri bile
ancak %55 oranında doğru tahmin yapabilmektedir. Sürprizler olur — bu futboldur.</p>

<p style="color:{C_MUTED}"><b style="color:{C_MUTED}">Modelin iyi yaptıkları:</b>
Favorileri doğru sıralamak, grup maçları için iyi kalibre edilmiş olasılık aralıkları
vermek ve sonuçlar geldikçe hızla güncellemek.</p>

</div>
""",
        unsafe_allow_html=True,
    )

    # ── Veri ve yöntem ────────────────────────────────────────────────────
    bolum("Veri ve yöntem")
    st.markdown(
        f"""
<div style="font-family:{FONT_BODY};line-height:1.8;font-size:0.82rem;max-width:700px;color:{C_MUTED}">

<b style="color:{C_TEXT}">Veri kaynakları</b><br>
49.000+ uluslararası maç (1872–2025) &nbsp;·&nbsp; FIFA dünya sıralaması (1992–2024)
&nbsp;·&nbsp; Piyasa tahmin verileri &nbsp;·&nbsp; WC2026 fikstürü (football-data.org)

<br><br>
<b style="color:{C_TEXT}">Nasıl çalışır</b><br>
1. <b style="color:{C_TEXT}">Bayesian Poisson modeli</b> — Her takımın son 8 yıldaki
maçlardan tahmin edilen atak ve savunma güçleri vardır. Yakın maçlar daha fazla ağırlık taşır
(yarılanma ömrü: 2 yıl).<br>
2. <b style="color:{C_TEXT}">Makine öğrenmesi (LightGBM)</b> — Elo puanları, son form,
FIFA sıralaması ve saha tipi gibi sinyalleri kullanarak kazanan/beraberlik/kaybeden tahmin eder.<br>
3. <b style="color:{C_TEXT}">Piyasa verileri</b> — Finansal piyasaların maç tahminleri istatistiksel olarak
arındırılır ve modele üçüncü sinyal olarak eklenir.<br>
4. Üç sinyal optimize edilmiş ağırlıklarla birleştirilir ve olasılıklar dürüst olacak şekilde kalibre edilir.

<br><br>
<b style="color:{C_TEXT}">Simülasyon</b><br>
Tam 32 takımlı turnuva, mevcut durumdan itibaren 25.000 kez simüle edilir.
Her simülasyon kalan tüm grup maçlarını ve eleme turlarını oynar.

</div>
""",
        unsafe_allow_html=True,
    )


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="WC 2026 · Yapay Zeka Tahminleri",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    baslik_goster()

    tahminler           = load_predictions()
    fikstür             = load_fixtures()
    calib               = load_calibration()
    turkiye_yolu        = load_turkey_path()
    zaman_serisi        = load_timeseries()
    model_metrikleri    = load_model_metrics()
    onem_sirasi         = load_feature_importance()
    kalibrasyon_kayit   = load_calib_records()
    bracket             = load_bracket_results()

    sekmeler = st.tabs(["🏆  Şampiyonluk", "🇹🇷  Türkiye", "📅  Tüm Maçlar", "🔬  Nasıl Çalışır"])

    with sekmeler[0]:
        sekme_sampiyonluk(bracket)

    with sekmeler[1]:
        sekme_turkiye(calib, turkiye_yolu, zaman_serisi, tahminler, bracket)

    with sekmeler[2]:
        sekme_maclar(tahminler, fikstür)

    with sekmeler[3]:
        sekme_teknik(model_metrikleri, onem_sirasi, kalibrasyon_kayit, bracket)


if __name__ == "__main__":
    main()
