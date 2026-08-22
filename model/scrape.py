import os
import requests
from textblob import TextBlob
from datetime import datetime , timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import random

# proxy_list = [
#     "http://45.87.68.7:15321",
#     "http://36.94.8.23:8080",
#     "http://118.193.32.18:887",
# ]


# Backend server URL (update if necessary)
BACKEND_URL = "http://127.0.0.1:5000/api/search"

def get_articles_from_backend(stock_name, hours, date):
    """
    Requests articles from the Flask backend server.
    """
    params = {"stock": stock_name, "time_filter": hours, "date": date}
    response = requests.get(BACKEND_URL, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching articles: {response.status_code}")
        return []
def scrape_article_advanced(url):
    """     
    from selenium.webdriver.chrome.options import Options
    Scrape article content and title from a given URL using Selenium.
    """
    # Set up Chrome options for headless browsing (no GUI)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Optional: run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Optional: Disable GPU (needed for headless mode)
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    service = Service(executable_path = "D:/CODE/chromedriver/chromedriver.exe")
    # Path to ChromeDriver (replace with the correct path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    try:
        # Open the URL with Selenium
        driver.get(url)
        # Wait for the page to load fully (adjust the time as needed)
        time.sleep(1)

        # Get the page source after JavaScript has rendered
        page_source = driver.page_source

        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")
        # Extract the title (assuming the title is in an <h1> tag)
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No title found"
        
        # Extract all paragraphs (<p>) for content
        paragraphs = soup.find_all("p")
        
        content = " ".join([p.get_text(strip=True) for p in paragraphs])

        # Return the scraped content
        return {"title": title, "content": content}

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

    finally:
        # Close the Selenium driver
        driver.quit()
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

    # Create a session to persist cookies
    session = requests.Session()
    session.headers.update(headers)
    # proxies = {"http": random.choice(proxy_list), "https": random.choice(proxy_list)}
    try:    
        # Send GET request using the session
        #proxies=proxies
        response = session.get(url, timeout=10, )
        
        # Check if the request was successful
        response.raise_for_status()

        # Simulate human behavior by adding a delay
        time.sleep(1)  # Delay to avoid being flagged as a bot

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("h1").get_text(strip=True) if soup.find("h1") else "No title found"

        # Extract paragraphs for content
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])

        return {"title": title, "content": content}

    except requests.exceptions.RequestException as e:
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

def analyze_stock_sentiment(stock_name, hours, date = datetime.today().strftime("%Y%m%d")):
    """
    Fetch articles, scrape content, and calculate sentiment analysis.
    """
    #print(f"Fetching articles for '{stock_name}' in the last {hours} hours...")
    articles = get_articles_from_backend(stock_name, hours, date)

    if not articles:
        #print("No articles found.")
        return None

    sentiments = []
    for article in articles:
        url = article["link"]
        #print(f"Scraping article: {url}")
        article_content = scrape_article_advanced(url)

        if article_content:
            sentiment = perform_sentiment_analysis(article_content["content"])
           # print(f"Title: {article_content['title']}")
            #print(f"Sentiment - Polarity: {sentiment['polarity']}, Subjectivity: {sentiment['subjectivity']}")
            sentiments.append(sentiment)

    if sentiments:
        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)
        avg_subjectivity = sum(s["subjectivity"] for s in sentiments) / len(sentiments)

        #print(f"\nAverage Sentiment Analysis for '{stock_name}':")
        #print(f"Polarity: {avg_polarity:.2f}, Subjectivity: {avg_subjectivity:.2f}")

        return {"polarity": avg_polarity, "subjectivity": avg_subjectivity}

    return None
def tester(url):
    article_content = scrape_article_advanced(url)
    if article_content:
        sentiment = perform_sentiment_analysis(article_content["content"])
        return sentiment


# Example usage
if __name__ == "__main__":
    stock_name = "Tesla"
    hours = 24
    #analyze_stock_sentiment(stock_name, hours)
    print(tester("https://www.fool.com/investing/2025/02/02/elon-musk-tesla-bigger-apple-nvidia-10-trillion/"))
