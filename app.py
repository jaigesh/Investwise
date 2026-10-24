import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="InvestWise AI — Research Lab", page_icon="📊", layout="wide")

st.title("📊 InvestWise AI — Investment Research Lab")
st.caption("Educational prototype • Indian & international stocks • historical strategy tests • mutual-fund NAV analysis")
st.warning(
    "Research and learning only — not financial advice. Market data may be delayed, incomplete, or unavailable. "
    "This app does not connect to a broker, execute trades, or predict guaranteed returns. Use paper analysis only."
)

with st.sidebar:
    st.header("Stock data settings")
    period_label = st.selectbox("Historical lookback", ["6 months", "1 year", "2 years", "5 years", "10 years"], index=2)
    period_map = {"6 months": "6mo", "1 year": "1y", "2 years": "2y", "5 years": "5y", "10 years": "10y"}
    period = period_map[period_label]
    st.caption("Yahoo Finance data is accessed through yfinance for personal research/education. Check provider terms before redistribution or production use.")


def clean_symbol(value: str) -> str:
    return value.strip().upper().replace(" ", "")


@st.cache_data(ttl=900, show_spinner=False)
def load_history(symbol: str, lookback: str) -> pd.DataFrame:
    """Download daily adjusted historical OHLCV through yfinance."""
    hist = yf.Ticker(symbol).history(period=lookback, interval="1d", auto_adjust=True, actions=False, timeout=20)
    if hist is None or hist.empty:
        raise ValueError("No historical data returned. Check the ticker and try again.")
    hist = hist.copy()
    if "Close" not in hist.columns:
        raise ValueError("The data provider did not return a Close price column.")
    hist = hist[~hist.index.duplicated(keep="last")].sort_index()
    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 3:
        raise ValueError("Not enough usable historical prices to analyse this symbol.")
    return hist


def drawdown_series(prices_or_equity: pd.Series) -> pd.Series:
    peak = prices_or_equity.cummax()
    return prices_or_equity / peak - 1.0


def annualized_volatility(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(252))


def max_drawdown(prices_or_equity: pd.Series) -> float:
    dd = drawdown_series(prices_or_equity.dropna())
    return float(dd.min()) if not dd.empty else float("nan")


def fmt_pct(value, digits=2):
    if pd.isna(value) or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_price_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    out = hist.copy()
    close = out["Close"]
    out["SMA 20"] = close.rolling(20, min_periods=20).mean()
    out["SMA 50"] = close.rolling(50, min_periods=50).mean()
    out["RSI 14"] = calculate_rsi(close)
    return out


def show_stock_analysis(symbol: str, lookback: str, show_backtest: bool = False):
    symbol = clean_symbol(symbol)
    if not symbol:
        st.info("Enter a ticker symbol to begin.")
        return
    try:
        with st.spinner(f"Loading historical prices for {symbol}…"):
            hist = load_history(symbol, lookback)
    except Exception as exc:
        st.error(f"Could not load {symbol}: {exc}")
        st.caption("Examples: RELIANCE.NS or TCS.NS (NSE), a BSE symbol ending in .BO, or AAPL/MSFT (US). Availability varies by provider.")
        return

    data = add_price_indicators(hist)
    close = data["Close"].dropna()
    returns = close.pct_change().dropna()
    total_return = close.iloc[-1] / close.iloc[0] - 1
    vol = annualized_volatility(returns)
    mdd = max_drawdown(close)
    if len(close) > 1:
        elapsed_years = max((close.index[-1] - close.index[0]).days / 365.25, 1 / 365.25)
        cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / elapsed_years) - 1
    else:
        cagr = float("nan")

    latest_date = pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d")
    st.caption(f"Ticker: **{symbol}** · Latest observation returned: **{latest_date}** · {len(close):,} daily observations")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest adjusted close", f"{close.iloc[-1]:,.2f}")
    c2.metric("Return over selected period", fmt_pct(total_return))
    c3.metric("Annualised volatility", fmt_pct(vol))
    c4.metric("Historical max drawdown", fmt_pct(mdd))
    st.caption(f"Approx. annualised growth rate (CAGR): {fmt_pct(cagr)}. Historical measurements do not forecast future returns.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Adjusted close", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA 20"], name="20-day SMA", line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA 50"], name="50-day SMA", line=dict(width=1, dash="dash")))
    fig.update_layout(title=f"{symbol} — historical adjusted price", xaxis_title="Date", yaxis_title="Price (local currency; provider units)", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    rsi = data["RSI 14"].dropna()
    if not rsi.empty:
        st.write(f"**Latest 14-day RSI:** {rsi.iloc[-1]:.1f} / 100. RSI is a momentum indicator, not a standalone buy/sell signal.")
    with st.expander("View recent data"):
        st.dataframe(data.tail(15).reset_index(), use_container_width=True)
        st.download_button("Download price data as CSV", data=data.reset_index().to_csv(index=False).encode("utf-8"), file_name=f"{symbol.replace('.', '_')}_history.csv", mime="text/csv")
    if show_backtest:
        show_sma_backtest(data, symbol)


def show_sma_backtest(data: pd.DataFrame, symbol: str):
    st.subheader("Simple trend-following strategy — historical simulation")
    st.caption("Long-only demonstration: hold when the short SMA is above the long SMA, otherwise hold cash. It does not short stocks.")
    a, b, c = st.columns(3)
    fast_window = a.selectbox("Short SMA (days)", [5, 10, 15, 20], index=3, key=f"fast_{symbol}")
    slow_window = b.selectbox("Long SMA (days)", [30, 50, 100, 150, 200], index=1, key=f"slow_{symbol}")
    cost_bps = c.number_input("Assumed cost per position change (basis points)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key=f"cost_{symbol}")
    if fast_window >= slow_window:
        st.warning("Choose a short SMA shorter than the long SMA.")
        return
    close = data["Close"].dropna()
    frame = pd.DataFrame(index=close.index)
    frame["close"] = close
    frame["short_sma"] = close.rolling(fast_window, min_periods=fast_window).mean()
    frame["long_sma"] = close.rolling(slow_window, min_periods=slow_window).mean()
    frame["asset_return"] = close.pct_change().fillna(0.0)
    signal = (frame["short_sma"] > frame["long_sma"]).astype(float)
    signal = signal.where(frame["long_sma"].notna(), 0.0)
    frame["position"] = signal.shift(1).fillna(0.0)
    turnover = frame["position"].diff().abs().fillna(frame["position"].abs())
    cost = cost_bps / 10_000.0
    frame["strategy_return"] = frame["position"] * frame["asset_return"] - turnover * cost
    frame["buy_hold_equity"] = (1 + frame["asset_return"]).cumprod()
    frame["strategy_equity"] = (1 + frame["strategy_return"]).cumprod()
    bh = frame["buy_hold_equity"]
    stg = frame["strategy_equity"]
    bh_ret = float(bh.iloc[-1] - 1)
    st_ret = float(stg.iloc[-1] - 1)
    st_vol = annualized_volatility(frame["strategy_return"])
    st_dd = max_drawdown(stg)
    left, middle, right = st.columns(3)
    left.metric("Strategy total return", fmt_pct(st_ret))
    middle.metric("Buy-and-hold total return", fmt_pct(bh_ret))
    right.metric("Strategy max drawdown", fmt_pct(st_dd))
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=frame.index, y=bh, name="Buy and hold"))
    chart.add_trace(go.Scatter(x=frame.index, y=stg, name="SMA strategy"))
    chart.update_layout(title="Growth of a hypothetical 1-unit investment", xaxis_title="Date", yaxis_title="Growth (starting near 1.0)", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(chart, use_container_width=True)
    st.write(f"Strategy annualised volatility: **{fmt_pct(st_vol)}**. This backtest is simplified and may omit taxes, market impact, liquidity constraints, delisted securities, and other real-world effects.")
    st.download_button("Download backtest results as CSV", data=frame.reset_index().to_csv(index=False).encode("utf-8"), file_name=f"{symbol.replace('.', '_')}_sma_backtest.csv", mime="text/csv", key=f"download_bt_{symbol}")


def portfolio_comparison(symbols_text: str, lookback: str):
    symbols = list(dict.fromkeys(clean_symbol(x) for x in symbols_text.split(",") if clean_symbol(x)))
    if len(symbols) < 2:
        st.info("Enter at least two tickers separated by commas, such as RELIANCE.NS, TCS.NS or AAPL, MSFT.")
        return
    if len(symbols) > 10:
        st.error("For this free prototype, compare at most 10 tickers at once.")
        return
    # Avoid presenting a mixed-currency / mixed-calendar portfolio statistic as valid.
    indian_flags = [sym.endswith((".NS", ".BO")) for sym in symbols]
    if any(indian_flags) and not all(indian_flags):
        st.error("This prototype does not calculate a combined portfolio across Indian and foreign markets because it does not convert currencies or model different market holidays. Compare Indian tickers together and foreign tickers together for now.")
        return
    if not any(indian_flags) and any("." in sym for sym in symbols):
        st.warning("Ticker suffixes can indicate different foreign exchanges and currencies. Only compare symbols that share the same currency and market calendar.")
    price_series = {}
    errors = []
    with st.spinner("Loading data for the selected tickers…"):
        for sym in symbols:
            try:
                h = load_history(sym, lookback)
                price_series[sym] = h["Close"].rename(sym)
            except Exception as exc:
                errors.append(f"{sym}: {exc}")
    if errors:
        st.warning("Some tickers could not be loaded: " + " · ".join(errors))
    if len(price_series) < 2:
        st.error("At least two tickers must load successfully.")
        return
    prices = pd.concat(price_series.values(), axis=1).sort_index().ffill().dropna(how="any")
    if len(prices) < 3:
        st.error("Not enough overlapping history among these tickers.")
        return
    rets = prices.pct_change().dropna()
    weights = np.repeat(1 / len(prices.columns), len(prices.columns))
    portfolio_returns = rets @ weights
    equity = (1 + portfolio_returns).cumprod()
    total = float(equity.iloc[-1] - 1)
    vol = annualized_volatility(portfolio_returns)
    dd = max_drawdown(equity)
    c1, c2, c3 = st.columns(3)
    c1.metric("Equal-weight total return", fmt_pct(total))
    c2.metric("Annualised volatility", fmt_pct(vol))
    c3.metric("Portfolio max drawdown", fmt_pct(dd))
    norm_prices = prices / prices.iloc[0]
    fig = go.Figure()
    for col in norm_prices.columns:
        fig.add_trace(go.Scatter(x=norm_prices.index, y=norm_prices[col], name=col))
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equal-weight portfolio", line=dict(width=3, dash="dash")))
    fig.update_layout(title="Normalised growth comparison (each line starts at 1.0)", xaxis_title="Date", yaxis_title="Growth multiple", hovermode="x unified", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Return correlation")
    st.caption("Correlation measures historical co-movement, not safety. Diversification can fail during stressed markets.")
    st.dataframe(rets.corr().style.format("{:.2f}"), use_container_width=True)
    st.download_button("Download aligned closing prices", data=prices.reset_index().to_csv(index=False).encode("utf-8"), file_name="portfolio_comparison_prices.csv", mime="text/csv")


def load_nav_file(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        # AMFI text downloads commonly use pipe separators; auto-detection fallback helps with variants.
        try:
            df = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str, engine="python", on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), sep=None, dtype=str, engine="python", on_bad_lines="skip")
    else:
        df = pd.read_csv(io.BytesIO(raw), sep=None, dtype=str, engine="python", on_bad_lines="skip")
        if len(df.columns) == 1 and "|" in str(df.columns[0]):
            df = pd.read_csv(io.BytesIO(raw), sep="|", dtype=str, engine="python", on_bad_lines="skip")
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def mutual_fund_tab():
    st.subheader("Mutual-fund NAV analyser")
    st.write("Upload an AMFI NAV history/report file or another CSV/TXT containing a date, NAV, and optionally a scheme-name column.")
    st.markdown(f"Official source: [AMFI NAV Download and NAV History]({ 'https://www.amfiindia.com/net-asset-value/nav-download' }). AMFI notes that its NAV history download allows a maximum of 90 days per request.")
    uploaded = st.file_uploader("Upload fund NAV data", type=["csv", "txt"], help="For AMFI, download NAV history from the official page and upload the file here.")
    if uploaded is None:
        st.info("No file uploaded yet. The stock modules can be used without a mutual-fund file.")
        st.markdown("**Expected columns (column names can differ):** scheme name, NAV value, and date. You will map the columns after uploading.")
        return
    try:
        df = load_nav_file(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")
        return
    if df.empty or len(df.columns) < 2:
        st.error("The file appears empty or does not have enough columns. Try downloading the NAV history again from AMFI.")
        return
    st.caption(f"Loaded {len(df):,} rows and {len(df.columns)} columns. Check the preview before continuing.")
    st.dataframe(df.head(8), use_container_width=True)
    cols = list(df.columns)
    date_guess = next((c for c in cols if "date" in c.lower()), cols[-1])
    nav_guess = next((c for c in cols if "nav" in c.lower() or "net asset" in c.lower()), cols[min(1, len(cols)-1)])
    scheme_guess = next((c for c in cols if "scheme name" in c.lower() or "scheme" in c.lower()), cols[0])
    c1, c2, c3 = st.columns(3)
    date_col = c1.selectbox("Date column", cols, index=cols.index(date_guess))
    nav_col = c2.selectbox("NAV column", cols, index=cols.index(nav_guess))
    scheme_choices = ["(No scheme column / all rows)"] + cols
    scheme_idx = scheme_choices.index(scheme_guess) if scheme_guess in scheme_choices else 0
    scheme_col = c3.selectbox("Scheme-name column", scheme_choices, index=scheme_idx)
    work = df.copy()
    if scheme_col != "(No scheme column / all rows)":
        available = sorted(work[scheme_col].dropna().astype(str).unique().tolist())
        chosen = st.selectbox("Choose a scheme", available) if available else None
        if chosen is not None:
            work = work[work[scheme_col].astype(str) == chosen]
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce", dayfirst=True)
    work["_nav"] = pd.to_numeric(work[nav_col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    work = work.dropna(subset=["_date", "_nav"]).query("_nav > 0").sort_values("_date").drop_duplicates("_date", keep="last")
    if len(work) < 2:
        st.error("Could not find at least two valid date/NAV observations. Check your selected columns and date format.")
        return
    first_nav, last_nav = float(work["_nav"].iloc[0]), float(work["_nav"].iloc[-1])
    total_return = last_nav / first_nav - 1
    days = max((work["_date"].iloc[-1] - work["_date"].iloc[0]).days, 1)
    cagr = (last_nav / first_nav) ** (365.25 / days) - 1 if days >= 30 else float("nan")
    nav_dd = max_drawdown(work["_nav"])
    c1, c2, c3 = st.columns(3)
    c1.metric("NAV observations", f"{len(work):,}")
    c2.metric("Return across uploaded period", fmt_pct(total_return))
    c3.metric("NAV max drawdown in file", fmt_pct(nav_dd))
    st.caption(f"Approx. annualised rate for periods of at least 30 days: {fmt_pct(cagr)}. NAV history alone does not include all fees, taxes, distributions, or plan-specific details.")
    fig = go.Figure(go.Scatter(x=work["_date"], y=work["_nav"], mode="lines", name="NAV"))
    fig.update_layout(title="Selected scheme — uploaded NAV history", xaxis_title="Date", yaxis_title="NAV (₹ per unit, if the source is INR)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    out = work[["_date", "_nav"]].rename(columns={"_date": "Date", "_nav": "NAV"})
    st.download_button("Download cleaned NAV series", data=out.to_csv(index=False).encode("utf-8"), file_name="cleaned_mutual_fund_nav.csv", mime="text/csv")


page = st.sidebar.radio(
    "Choose a module",
    ["Stock research", "Strategy backtest", "Portfolio comparison", "Mutual funds", "How to use"],
)

if page == "Stock research":
    st.subheader("Single-stock research")
    with st.form("stock_form"):
        symbol_input = st.text_input("Ticker symbol", value="RELIANCE.NS", help="NSE examples end in .NS; BSE examples often end in .BO; US examples include AAPL, MSFT and NVDA.")
        stock_submit = st.form_submit_button("Analyse stock")
    if stock_submit:
        st.session_state["stock_requested"] = clean_symbol(symbol_input)
    if st.session_state.get("stock_requested"):
        show_stock_analysis(st.session_state["stock_requested"], period)
    else:
        st.info("Enter a ticker and click Analyse stock to load historical data.")

elif page == "Strategy backtest":
    st.subheader("Test a basic strategy against buy-and-hold")
    with st.form("backtest_form"):
        bt_symbol_input = st.text_input("Ticker for historical simulation", value="RELIANCE.NS", help="This is a historical simulation, not a trade instruction.")
        bt_submit = st.form_submit_button("Run historical simulation")
    if bt_submit:
        st.session_state["backtest_requested"] = clean_symbol(bt_symbol_input)
    bt_symbol = st.session_state.get("backtest_requested", "")
    if bt_symbol:
        try:
            with st.spinner("Loading data for the simulation…"):
                bt_data = add_price_indicators(load_history(bt_symbol, period))
            st.caption(f"Data returned for {bt_symbol} ends on {pd.Timestamp(bt_data.index[-1]).strftime('%Y-%m-%d')}.")
            show_sma_backtest(bt_data, bt_symbol)
        except Exception as exc:
            st.error(f"Could not run this simulation: {exc}")
    else:
        st.info("Enter a ticker and click Run historical simulation.")

elif page == "Portfolio comparison":
    st.subheader("Compare a simple equal-weight portfolio")
    with st.form("portfolio_form"):
        tickers_input = st.text_input("Comma-separated tickers", value="RELIANCE.NS, TCS.NS, AAPL", help="All prices are compared in their quoted currencies. This prototype does not convert currencies; mixed-market growth multiples are approximate comparisons only.")
        portfolio_submit = st.form_submit_button("Compare tickers")
    if portfolio_submit:
        st.session_state["portfolio_requested"] = tickers_input
    tickers_text = st.session_state.get("portfolio_requested", "")
    if tickers_text:
        portfolio_comparison(tickers_text, period)
        st.info("Mixed Indian and foreign tickers are shown in normalised growth units, not converted into one currency. INR-based investor returns can differ because of FX changes and taxes.")
    else:
        st.info("Enter at least two comma-separated tickers and click Compare tickers.")

elif page == "Mutual funds":
    mutual_fund_tab()

elif page == "How to use":
    st.subheader("Current prototype boundaries")
    st.markdown("""
- **Stock research:** adjusted daily history, period return, approximate CAGR, annualised volatility, maximum drawdown, 20/50-day simple moving averages, and RSI.
- **Strategy testing:** a simple long-only moving-average strategy compared with buy-and-hold, with a user-set transaction-cost assumption.
- **Portfolio comparison:** historical equal-weight returns and correlations for up to 10 tickers.
- **Mutual funds:** analysis of a NAV file uploaded by you. This prototype does not yet automatically pull every fund's complete NAV history.
- **No orders:** no brokerage login, no trading execution, and no automatic investment recommendation.

**Important data limitations:** price history may be delayed, revised, incomplete, or blocked by the provider. A strategy that performs well on old data may fail in future markets. Backtests need more rigorous testing before being relied on.
    """)
    st.markdown("**Next development stages:** robust unit tests, data-source validation, benchmark-aware fund comparisons, tax/cost modelling, walk-forward evaluation, secure user accounts, and only then a carefully reviewed AI explanation layer.")

st.divider()
st.caption(f"InvestWise AI prototype · generated {datetime.now().strftime('%Y-%m-%d')} · Education and research only")
