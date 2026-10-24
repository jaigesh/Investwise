# InvestWise AI — Research Lab (Prototype)

A beginner-friendly, free-to-deploy educational prototype for historical research across Indian and supported international stocks, a simple SMA strategy simulation, equal-weight portfolio comparisons, and uploaded mutual-fund NAV histories.

## What it does

- **Stock research:** adjusted historical prices, period return, approximate CAGR, annualised volatility, maximum drawdown, 20/50-day moving averages, RSI.
- **Strategy backtest:** long-only short/long moving-average crossover compared with buy-and-hold; editable assumed transaction cost per position change.
- **Portfolio comparison:** equal-weighted historical returns and return correlations for up to ten tickers.
- **Mutual-fund NAV analysis:** upload an AMFI or other compatible CSV/TXT file and map the date, NAV and scheme columns.

## Important limitations

This is an educational research tool, not financial advice. It does not generate personalised buy/sell instructions, connect to a broker, or execute transactions. Data can be delayed, incomplete, or unavailable. Backtests are simplified and do not include every tax, fee, liquidity, FX, or market-impact effect. Do not make financial decisions from this prototype alone.

Stock price data is retrieved through `yfinance`, an open-source interface to Yahoo Finance's publicly available APIs. The yfinance documentation says the data access is intended for personal use; review Yahoo's applicable terms before redistribution, commercial use, or production deployment.

## Run locally (optional)

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free in the cloud (recommended for the user's cloud setup)

1. Download and unzip this project.
2. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md` to its root. Do not upload any passwords, API keys, or financial account data.
3. Open Streamlit Community Cloud: https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository and branch, and set the main file to `app.py`.
5. Click **Deploy**. The first build may take a few minutes. If it fails, open the app's logs and check dependency errors.

Streamlit Community Cloud is a hosted free deployment option, but service/resource limits and availability may change. See https://docs.streamlit.io/deploy/streamlit-community-cloud.

## How to use

### Indian / foreign stocks

- NSE examples: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- BSE tickers often use `.BO`; availability depends on the data provider.
- US examples: `AAPL`, `MSFT`, `NVDA`
- Ticker coverage varies. If a symbol fails, verify the exact Yahoo Finance symbol and retry.

The displayed price is in the currency returned by the data source. Do not compare absolute prices across currencies as though they were the same currency.

### Mutual-fund NAV

1. Open AMFI's official NAV download/history page: https://www.amfiindia.com/net-asset-value/nav-download
2. Download historical NAV for the relevant scheme and period. The AMFI page currently notes a maximum of 90 days per history request.
3. Upload the CSV/TXT file in the **Mutual funds** tab.
4. Select the correct date, NAV, and scheme-name columns, then select the scheme.
5. Check the raw preview and ensure the date/NAV interpretation looks correct before trusting the calculated metrics.

NAV alone is not enough to evaluate a fund completely. A fuller version should compare total returns against the correct benchmark, costs, risk, portfolio overlap, and plan type, and should account for distributions appropriately.

## Suggested development order

1. Validate calculations and add tests.
2. Add source timestamps, caching, logging, provider error handling, and data-quality warnings.
3. Add benchmark comparisons and better mutual-fund data ingestion.
4. Add robust walk-forward backtests, costs, and currency-aware portfolio returns.
5. Add an LLM explanation interface that cites source data, clearly separates facts from estimates, and never treats a prediction as certainty.
6. Keep all real-money trading features out of the prototype until legal, security, suitability, and reliability requirements have been reviewed.
