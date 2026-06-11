"""Page 07 — Live Trading Dashboard.

Shows:
- Current Alpaca paper positions with P&L
- Today's signals from best promoted strategy
- Portfolio equity curve (Alpaca history)
- Rebalance calendar
- Quick-action buttons (generate signals, dry-run rebalance)
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Live Trading", page_icon="📡", layout="wide")
st.title("📡 Live Trading Dashboard")

# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(ttl=60)
def _get_alpaca():
    from trading_bot.live.alpaca import get_alpaca_client
    return get_alpaca_client()


@st.cache_data(ttl=300)
def _load_signals():
    from trading_bot.live.signals import generate_signals
    try:
        return generate_signals(), None
    except Exception as e:
        return None, str(e)


# ── Connection status ─────────────────────────────────────────────────────────

alpaca = _get_alpaca()

col_a, col_b, col_c = st.columns(3)

with col_a:
    if alpaca:
        try:
            account = alpaca.get_account()
            market_open = alpaca.is_market_open()
            st.metric("Portfolio Value", f"${account.portfolio_value:,.0f}")
            st.caption(f"Cash: ${account.cash:,.0f}  |  Market: {'🟢 Open' if market_open else '🔴 Closed'}")
        except Exception as e:
            st.error(f"Alpaca error: {e}")
    else:
        st.warning("🔑 Alpaca not configured")
        st.caption("Add `TRADEBOT_ALPACA_API_KEY` and `TRADEBOT_ALPACA_SECRET_KEY` to `.env`")

with col_b:
    from trading_bot.registry import list_strategies
    strategies = list_strategies()
    promoted = [s for s in strategies if s.get("status") == "promoted"]
    st.metric("Promoted Strategies", len(promoted))
    if promoted:
        st.caption(f"Best: {promoted[0]['name']}")

with col_c:
    from trading_bot.live.runner import is_rebalance_day
    today = pd.Timestamp.today().normalize()
    rebalance_today = is_rebalance_day(today)
    if rebalance_today:
        st.metric("Rebalance Day", "✅ TODAY")
    else:
        # Find next rebalance day
        for i in range(1, 32):
            candidate = today + pd.Timedelta(days=i)
            if is_rebalance_day(candidate):
                days_left = i
                break
        else:
            days_left = "?"
        st.metric("Next Rebalance", f"In {days_left} days")
    st.caption(f"Today: {today.strftime('%Y-%m-%d %A')}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_signals, tab_positions, tab_equity, tab_holdout = st.tabs([
    "📊 Today's Signals",
    "💼 Positions",
    "📈 Equity Curve",
    "🧪 Holdout Test",
])

# ── Tab 1: Signals ────────────────────────────────────────────────────────────

with tab_signals:
    st.subheader("Today's Trading Signals")

    if st.button("🔄 Refresh Signals", type="primary"):
        st.cache_data.clear()
        st.rerun()

    signals, err = _load_signals()

    if err:
        if "No strategies" in err or "not found" in err.lower():
            st.warning(
                "No promoted strategy found. Run the research loop first:\n"
                "```\ntradebot research --rounds 10 --candidates 5\n```"
            )
        else:
            st.error(f"Signal generation error: {err}")
    elif signals:
        col1, col2, col3 = st.columns(3)
        col1.metric("Strategy", signals.strategy_name)
        col2.metric("Positions", len(signals.positions))
        col3.metric(
            "Regime",
            "Active ✅" if signals.regime_active else ("Off ⚠️" if signals.regime_active is False else "N/A"),
        )

        st.caption(f"Signals as of: {signals.asof}")

        if signals.positions:
            df = pd.DataFrame([
                {"Symbol": sym, "Weight %": round(w * 100, 1), "Allocation ($100k)": round(w * 100_000)}
                for sym, w in sorted(signals.positions.items(), key=lambda x: -x[1])
            ])

            # Bar chart
            fig = go.Figure(go.Bar(
                x=df["Symbol"],
                y=df["Weight %"],
                marker_color="#00B4D8",
                text=df["Weight %"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            ))
            fig.update_layout(
                title="Target Portfolio Weights",
                xaxis_title="Symbol",
                yaxis_title="Weight (%)",
                height=350,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

            total_w = sum(signals.positions.values())
            st.caption(f"Total allocated: {total_w*100:.1f}%  |  Cash buffer: {(1-total_w)*100:.1f}%")
        else:
            st.info("No positions — regime filter is blocking all trades (bearish regime).")

        # Dry-run rebalance button
        if alpaca and rebalance_today:
            st.warning("⚠️ Today is a rebalance day!")
            col_dry, col_live = st.columns(2)
            with col_dry:
                if st.button("🔍 Dry-run rebalance (simulate)", use_container_width=True):
                    from trading_bot.live.runner import run_daily
                    with st.spinner("Simulating..."):
                        result = run_daily(dry_run=True, force_rebalance=True, skip_ingest=True)
                    st.success(f"Dry run complete. {result.orders_count} orders simulated.")
            with col_live:
                if st.button("🚀 Execute rebalance (LIVE)", type="primary", use_container_width=True):
                    from trading_bot.live.runner import run_daily
                    with st.spinner("Executing..."):
                        result = run_daily(dry_run=False, force_rebalance=True, skip_ingest=True)
                    if result.errors:
                        st.error(f"Errors: {result.errors}")
                    else:
                        st.success(f"✅ {result.orders_count} orders submitted to Alpaca!")

# ── Tab 2: Current Positions ──────────────────────────────────────────────────

with tab_positions:
    st.subheader("Current Paper Positions")

    if alpaca is None:
        st.info("Configure Alpaca API keys to see live positions.")
    else:
        if st.button("🔄 Refresh Positions"):
            st.cache_resource.clear()
            st.rerun()

        try:
            positions = alpaca.get_positions()
            if not positions:
                st.info("No open positions. Run a rebalance to open positions.")
            else:
                rows = []
                for p in sorted(positions, key=lambda x: -x.market_value):
                    rows.append({
                        "Symbol": p.symbol,
                        "Qty": round(p.qty, 3),
                        "Current Price": f"${p.current_price:,.2f}",
                        "Market Value": f"${p.market_value:,.2f}",
                        "Cost Basis": f"${p.cost_basis:,.2f}",
                        "Unrealized P&L": f"${p.unrealized_pl:+,.2f}",
                        "P&L %": f"{p.unrealized_plpc*100:+.2f}%",
                    })
                df = pd.DataFrame(rows)

                # Color P&L column
                total_pl = sum(p.unrealized_pl for p in positions)
                total_value = sum(p.market_value for p in positions)

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Market Value", f"${total_value:,.0f}")
                m2.metric(
                    "Total Unrealized P&L",
                    f"${total_pl:+,.0f}",
                    delta_color="normal",
                )
                m3.metric("Positions", len(positions))

                st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading positions: {e}")

# ── Tab 3: Equity Curve ───────────────────────────────────────────────────────

with tab_equity:
    st.subheader("Paper Portfolio Equity Curve")

    if alpaca is None:
        st.info("Configure Alpaca API keys to see equity curve.")
    else:
        period = st.select_slider("Period", options=["1M", "3M", "6M", "1A", "all"], value="3M")

        try:
            history = alpaca.get_portfolio_history(period=period, timeframe="1D")
            timestamps = history.get("timestamp", [])
            equity = history.get("equity", [])

            if timestamps and equity:
                dates = pd.to_datetime(timestamps, unit="s")
                eq_series = pd.Series(equity, index=dates).dropna()

                if len(eq_series) > 1:
                    total_return = (eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100
                    peak = eq_series.max()
                    current_dd = (peak - eq_series.iloc[-1]) / peak * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Return", f"{total_return:+.2f}%")
                    m2.metric("Peak Value", f"${peak:,.0f}")
                    m3.metric("Current Drawdown", f"{current_dd:.1f}%")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq_series.index,
                        y=eq_series.values,
                        mode="lines",
                        name="Portfolio",
                        line=dict(color="#00B4D8", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(0,180,216,0.1)",
                    ))
                    fig.update_layout(
                        title="Paper Portfolio Equity",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough history yet. Check back after the first rebalance.")
            else:
                st.info("No portfolio history available yet.")

        except Exception as e:
            st.error(f"Error loading portfolio history: {e}")

# ── Tab 4: Holdout Test ───────────────────────────────────────────────────────

with tab_holdout:
    st.subheader("Final Holdout OOS Test")
    st.info(
        "**What is the holdout?** The period 2024-01-01 → today was never used during "
        "training or the research loop. This is the final, one-time statistical test before "
        "deploying a strategy to live paper trading.\n\n"
        "⚠️ Run this only once per strategy — looking at holdout data 'consumes' it."
    )

    strategies = list_strategies()
    promoted = [s for s in strategies if s.get("status") == "promoted"]

    if not promoted:
        st.warning("No promoted strategies. Run the research loop first.")
    else:
        strategy_names = [s["name"] for s in promoted]
        selected = st.selectbox("Select promoted strategy", strategy_names)
        holdout_start = st.date_input("Holdout start", value=pd.Timestamp("2024-01-01").date())

        if st.button("🧪 Run Holdout Test", type="primary"):
            with st.spinner(f"Running holdout test on {selected}..."):
                from trading_bot.validation.holdout import run_holdout_test
                try:
                    result = run_holdout_test(
                        strategy_name=selected,
                        holdout_start=str(holdout_start),
                    )

                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Holdout Sharpe", f"{result.sharpe:.3f}",
                                delta=f"IS: {result.is_sharpe:.3f}")
                    col2.metric("CAGR", f"{result.cagr*100:.1f}%")
                    col3.metric("Max Drawdown", f"{result.max_drawdown*100:.1f}%")
                    col4.metric("Months", result.n_months)

                    if result.passed:
                        st.success(f"✅ {result.summary()}")
                    else:
                        st.warning(f"⚠️ {result.summary()}")

                    # Equity curve
                    if not result.equity_curve.empty:
                        fig = go.Figure(go.Scatter(
                            x=result.equity_curve.index,
                            y=result.equity_curve.values,
                            mode="lines",
                            line=dict(color="#2ECC71" if result.passed else "#F39C12", width=2),
                            fill="tozeroy",
                        ))
                        fig.update_layout(
                            title=f"Holdout Equity — {selected}",
                            height=350,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Monthly returns heatmap
                    if not result.monthly_returns.empty:
                        st.subheader("Monthly Returns")
                        monthly_df = pd.DataFrame({
                            "Month": result.monthly_returns.index.strftime("%Y-%m"),
                            "Return %": (result.monthly_returns * 100).round(2),
                        })
                        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Holdout test failed: {e}")
