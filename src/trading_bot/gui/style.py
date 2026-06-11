"""Apple-style design system for the TradingBot dashboard.

iOS dark-mode palette, SF system font stack, glass cards, minimal chrome.
Shared by all views: inject_css(), plotly_layout(), metric_card(), section().
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ── iOS dark palette ─────────────────────────────────────────────────────────
BG       = "#000000"
CARD     = "#1c1c1e"
CARD2    = "#2c2c2e"
BORDER   = "#3a3a3c"
TEXT     = "#f5f5f7"
TEXT2    = "#98989d"
BLUE     = "#0a84ff"
GREEN    = "#30d158"
RED      = "#ff453a"
GOLD     = "#ffd60a"
PURPLE   = "#bf5af2"
TEAL     = "#64d2ff"
ORANGE   = "#ff9f0a"

PALETTE = [BLUE, GREEN, PURPLE, ORANGE, TEAL]

FONT = ("-apple-system, BlinkMacSystemFont, 'SF Pro Display', "
        "'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif")


def inject_css() -> None:
    st.markdown(f"""
    <style>
    /* ── chrome ──────────────────────────────────────────── */
    /* Nascondi SOLO menu, footer e toolbar — MAI l'header intero: dentro
       c'è il chevron che apre/chiude la sidebar (bug Safari: visibility
       sull'header lo eliminava). */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
    }}
    /* Chevron sidebar SEMPRE visibile (Streamlit lo mostra solo in hover:
       era il motivo per cui "la freccia non c'è") e leggibile sul nero. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {{
        visibility: visible !important;
    }}
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] * {{
        color: {TEXT} !important;
        fill: {TEXT} !important;
    }}
    .block-container {{padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}}

    html, body, [class*="css"] {{
        font-family: {FONT};
        -webkit-font-smoothing: antialiased;
    }}
    .stApp {{ background: {BG}; }}

    /* ── sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {BORDER};
    }}
    /* NIENTE font-family su tutti i discendenti della sidebar: le icone
       Material Symbols (chevron, nav) sono legature e si rompono in testo
       grezzo se gli sovrascrivi il font. Il testo eredita già SF Pro. */

    /* ── headings ────────────────────────────────────────── */
    h1 {{
        font-weight: 700 !important; letter-spacing: -0.02em !important;
        font-size: 2rem !important; color: {TEXT} !important;
    }}
    h2, h3 {{
        font-weight: 600 !important; letter-spacing: -0.01em !important;
        color: {TEXT} !important;
    }}

    /* ── apple metric cards ──────────────────────────────── */
    .apl-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }}
    .apl-card {{
        flex: 1; min-width: 150px;
        background: linear-gradient(180deg, {CARD} 0%, #161618 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,.5);
    }}
    .apl-label {{
        font-size: 12px; font-weight: 500; color: {TEXT2};
        text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px;
    }}
    .apl-value {{
        font-size: 22px; font-weight: 700; letter-spacing: -0.02em;
        color: {TEXT}; line-height: 1.15; font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }}
    .apl-delta {{ font-size: 13px; font-weight: 600; margin-top: 2px; }}
    .apl-green {{ color: {GREEN}; }}
    .apl-red   {{ color: {RED}; }}
    .apl-dim   {{ color: {TEXT2}; }}
    .apl-gold  {{ color: {GOLD}; }}
    .apl-blue  {{ color: {BLUE}; }}

    /* ── section headers ─────────────────────────────────── */
    .apl-section {{
        font-size: 13px; font-weight: 600; color: {TEXT2};
        text-transform: uppercase; letter-spacing: .08em;
        margin: 18px 0 8px 2px;
    }}

    /* ── pills ───────────────────────────────────────────── */
    .apl-pill {{
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 12px; font-weight: 600;
    }}
    .pill-green {{ background: rgba(48,209,88,.15); color: {GREEN}; }}
    .pill-red   {{ background: rgba(255,69,58,.15);  color: {RED}; }}
    .pill-gold  {{ background: rgba(255,214,10,.15); color: {GOLD}; }}
    .pill-blue  {{ background: rgba(10,132,255,.15); color: {BLUE}; }}
    .pill-dim   {{ background: rgba(152,152,157,.15); color: {TEXT2}; }}

    /* ── tables ──────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden;
    }}

    /* ── tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background: {CARD}; border-radius: 12px; padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 9px; font-weight: 600; color: {TEXT2};
    }}
    .stTabs [aria-selected="true"] {{
        background: {CARD2} !important; color: {TEXT} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def plotly_layout(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    # Margine sinistro 60px + tick color TEXT: con l=8 e grigio scuro le
    # label dell'asse Y uscivano dal canvas ("0k", "00") ed erano illeggibili.
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=TEXT2, size=12),
        height=height,
        margin=dict(l=60, r=16, t=16, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER,
                        font=dict(family=FONT, color=TEXT)),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)) if legend
        else dict(visible=False),
        xaxis=dict(gridcolor=CARD2, zeroline=False, showline=False,
                   tickfont=dict(size=12, color=TEXT)),
        yaxis=dict(gridcolor=CARD2, zeroline=False, showline=False,
                   tickfont=dict(size=12, color=TEXT)),
    )
    return fig


def cards(items: list[tuple[str, str, str | None, str]]) -> None:
    """Render a row of Apple metric cards.

    items: (label, value, delta_text|None, delta_class in
            {green,red,dim,gold,blue})
    """
    html = '<div class="apl-row">'
    for label, value, delta, klass in items:
        d = (f'<div class="apl-delta apl-{klass}">{delta}</div>' if delta else "")
        html += (f'<div class="apl-card"><div class="apl-label">{label}</div>'
                 f'<div class="apl-value">{value}</div>{d}</div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="apl-section">{title}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "dim") -> str:
    return f'<span class="apl-pill pill-{kind}">{text}</span>'
