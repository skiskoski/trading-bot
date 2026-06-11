"""TradingBot — router multi-pagina (st.navigation, API ufficiale Streamlit).

Sidebar di navigazione a 8 sezioni, sempre visibile (initial_sidebar_state
="expanded"). Ogni vista è uno script in views/ eseguito dal router; il
set_page_config e il CSS vengono iniettati UNA volta qui.
"""

from __future__ import annotations

import streamlit as st

from trading_bot.gui.style import inject_css

st.set_page_config(page_title="TradingBot", page_icon="◉", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

pages = [
    st.Page("views/panoramica.py", title="Panoramica", icon="📊", default=True),
    st.Page("views/ricerca.py", title="Ricerca", icon="🔬"),
    st.Page("views/strategie.py", title="Strategie", icon="📈"),
    st.Page("views/portafoglio.py", title="Portafoglio", icon="🎯"),
    st.Page("views/validazione.py", title="Validazione", icon="✅"),
    st.Page("views/live.py", title="Live", icon="💼"),
    st.Page("views/storico.py", title="Storico", icon="📅"),
    st.Page("views/guida.py", title="Guida", icon="ℹ️"),
]

nav = st.navigation(pages)

with st.sidebar:
    st.markdown("---")
    st.caption("◉ TradingBot — quant onesto\n\n"
               "Gate: PBO<0.4 · OOS attivo≥0.5 · path+≥65%")

nav.run()
