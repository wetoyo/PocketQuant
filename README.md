# PocketQuant

PocketQuant is a lightweight quantitative finance toolkit designed for rapid prototyping, analysis, and deployment of trading strategies.

## Features

- Sentiment analysis with webscraping
- Regression on stock prices
- Support for cross ticker regression

## Installation

```bash
git clone https://github.com/wetoyo/PocketQuant.git
cd PocketQuant
pip install -r requirements.txt
```

## Usage

### Launching the UI

To start the PocketQuant user interface for offline analysis, run:

```bash
python offlineapp.py
```

### Sentiment Analysis with Webscraping

For sentiment analysis and webscraping features, populate a `.env` file with your Google API and Google CVX keys:

```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CVX_KEY=your_google_cvx_key
```

Then launch the enhanced app:

```bash
python app.py
```


## License

MIT License