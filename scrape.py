import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from datetime import datetime, timedelta

def search_articles(stock_name, hours):
    """
    Search Google for articles related to a stock within the last specified hours.
    """
    # Google search query for the stock
    time_filter = f"&tbs=qdr:h{hours}"  # Limits results to the last 'hours'
    search_url = f"https://www.google.com/search?q={stock_name}+stock{time_filter}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    }

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        print("Failed to retrieve search results.")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for link in soup.find_all("a"):
        href = link.get("href")
        if href:
            if "https://" in href and "google.com" not in href and "youtube.com" not in href:
                # Extract the actual URL from the Google search result link
                article_url = href.split("https://")[1].split("&")[0]
                results.append("https://" + article_url)

    return results[:10]  # Limit to top 10 results

def scrape_article(url):
    """
    Scrape article content and title from a given URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
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

if __name__ == "__main__":
    # Parameters
    stock_name = "Tesla"  # Replace with the desired stock name
    hours = 24  # Time frame for article search

    print(f"Searching articles for '{stock_name}' in the last {hours} hours...")
    article_urls = search_articles(stock_name, hours)

    if article_urls:
        print(f"Found {len(article_urls)} articles.")
        sentiments = []

        for url in article_urls:
            print(f"Scraping article: {url}")
            article = scrape_article(url)

            if article:
                sentiment = perform_sentiment_analysis(article["content"])
                print(f"Title: {article['title']}")
                print(f"Sentiment - Polarity: {sentiment['polarity']}, Subjectivity: {sentiment['subjectivity']}")
                sentiments.append(sentiment)
            else:
                print("Failed to scrape the article.")

        # Summarize the sentiment analysis
        if sentiments:
            avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)
            avg_subjectivity = sum(s["subjectivity"] for s in sentiments) / len(sentiments)
            print(f"\nAverage Sentiment Analysis for '{stock_name}':")
            print(f"Polarity: {avg_polarity:.2f}, Subjectivity: {avg_subjectivity:.2f}")
    else:
        print("No articles found.")
