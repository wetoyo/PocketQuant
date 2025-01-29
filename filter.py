import requests
import json
from datetime import datetime
import re

def load_blocked_urls(file_path="blocked_urls.txt"):
    """Load the list of blocked URLs from a file."""
    try:
        with open(file_path, "r") as f:
            blocked_urls = f.read().splitlines()
        return blocked_urls
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []

def filter_articles(articles, time_filter, blocked_urls):
    """Filter articles based on time and blocked URLs."""
    filtered_articles = []
    
    # Loop through each article and apply the filters
    for article in articles:
        # Filter by snippet time if time_filter < 24 hours
        if time_filter < 24:
            snippet = article.get("snippet", "")
            hours_ago_match = re.search(r"(\d+)\s*hour", snippet)
            if hours_ago_match:
                hours_ago = int(hours_ago_match.group(1))
                if hours_ago > time_filter:
                    continue  # Skip if the article is older than the requested time
        
        # Filter by blocked URLs
        article_url = article.get("link", "")
        if any(article_url.startswith(blocked_url) for blocked_url in blocked_urls):
            continue  # Skip if the article URL is in the blocked list
        
        # If the article passes both filters, add it to the result
        filtered_articles.append(article)
    
    return filtered_articles
