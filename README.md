# PocketQuant

PocketQuant is a lightweight quantitative finance model, performing random forest regression across tickers to predict future price in small timesteps.

## Features

- Sentiment analysis with webscraping
- Regression on stock prices
- Support for cross ticker regression

## Project Structure

```
PocketQuant/
├── backend/    # Flask API for Google-search-backed news lookup (app.py, filter.py, blocked_urls.txt)
├── model/      # Prediction pipeline: data fetching, XGBoost training, sentiment scraping
├── ui/         # PySide6 desktop GUI (offlineapp.py)
├── scripts/    # Standalone dev/demo scripts (test.py, testpyside.py)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/wetoyo/PocketQuant.git
cd PocketQuant
pip install -r requirements.txt
```

## Usage

### Launching the UI

To start the PocketQuant user interface for offline analysis, run (from the repo root):

```bash
python -m ui.offlineapp
```

### Sentiment Analysis with Webscraping

For sentiment analysis and webscraping features, populate a `.env` file with your Google API key and Custom Search Engine ID:

```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_custom_search_engine_id
```

Then launch the backend search API (from the repo root):

```bash
python -m backend.app
```

Sentiment scraping also requires a local ChromeDriver; update the hardcoded path in `model/scrape.py` (`scrape_article_advanced`) to match your ChromeDriver install location.

## License

MIT License
