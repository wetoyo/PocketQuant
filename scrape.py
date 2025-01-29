import os
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
# Backend server URL (update if necessary)
BACKEND_URL = "http://127.0.0.1:5000/api/search"

def get_articles_from_backend(stock_name, hours):
    """
    Requests articles from the Flask backend server.
    """
    params = {"stock": stock_name, "time_filter": hours}
    response = requests.get(BACKEND_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching articles: {response.status_code}")
        return []

def scrape_article(url):
    """
    Scrape article content and title from a given URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No title found"

        # Extract paragraphs for content
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])

        return {"title": title, "content": content}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on text content using TextBlob.
    """
    analysis = TextBlob(text)
    return {
        "polarity": analysis.polarity,  # -1 (negative) to 1 (positive)
        "subjectivity": analysis.subjectivity  # 0 (objective) to 1 (subjective)
    }

def analyze_stock_sentiment(stock_name, hours):
    """
    Fetch articles, scrape content, and calculate sentiment analysis.
    """
    print(f"Fetching articles for '{stock_name}' in the last {hours} hours...")
    articles = get_articles_from_backend(stock_name, hours)

    if not articles:
        print("No articles found.")
        return None

    sentiments = []
    for article in articles:
        url = article["link"]
        print(f"Scraping article: {url}")
        article_content = scrape_article(url)

        if article_content:
            sentiment = perform_sentiment_analysis(article_content["content"])
            print(f"Title: {article_content['title']}")
            print(f"Sentiment - Polarity: {sentiment['polarity']}, Subjectivity: {sentiment['subjectivity']}")
            sentiments.append(sentiment)

    if sentiments:
        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)
        avg_subjectivity = sum(s["subjectivity"] for s in sentiments) / len(sentiments)

        print(f"\nAverage Sentiment Analysis for '{stock_name}':")
        print(f"Polarity: {avg_polarity:.2f}, Subjectivity: {avg_subjectivity:.2f}")

        return {"polarity": avg_polarity, "subjectivity": avg_subjectivity}

    return None

# Example usage
if __name__ == "__main__":
    stock_name = "Tesla"
    hours = 24
    analyze_stock_sentiment(stock_name, hours)
