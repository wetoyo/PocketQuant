import os
import requests
from datetime import datetime , timedelta     
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from filter import load_blocked_urls, filter_articles   
# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get API credentials from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

@app.route('/api/search', methods=['GET'])
def search_google():
    """Searches Google for stock-related articles using the Google Custom Search API."""
    stock_name = request.args.get('stock')
    time_filter = request.args.get('time_filter', "")
    date = request.args.get('date', "")
    count = request.args.get('count', "")
    time_filter = int(time_filter) 
    if not stock_name:
        return jsonify({"error": "Missing 'stock' parameter"}), 400
    if not time_filter:
        time_filter = -1
    if not date:
        date = datetime.today().strftime("%Y%m%d")
    if not count:
        count = 10
    given_date = datetime.strptime(date, "%Y%m%d")
    week_ago = given_date - timedelta(days=7)
    week_ago_date = week_ago.strftime("%Y%m%d")
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return jsonify({"error": "Missing API key or CX ID"}), 500
    
    # Construct the search URL using Google's API
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": f"{stock_name} stock news",  # Your search query
        "key": GOOGLE_API_KEY,        # Your API key
        "cx": GOOGLE_CX,              # Your Custom Search Engine ID
        "num": count,                    # Number of results to fetch
        "excludeTerms": "quote",            
        "sort": f"date:r:{week_ago_date}:{date}" 
    }
    if (time_filter != -1 and time_filter <= 24):
        params["dateRestrict"] = f"d{time_filter // 24}"
    
    response = requests.get(search_url, params=params)

    if response.status_code == 200:

        blocked_urls = load_blocked_urls()
        
        # Get the search results (articles)
        articles = response.json().get("items", [])
        
        # Filter the articles based on time filter and blocked URLs
        
        filtered_articles = filter_articles(articles, time_filter, blocked_urls)

        article_count = len(filtered_articles)
        tries = 0
        while (article_count <= count and tries < 2):
            params = {
                "q": f"{stock_name} stock news",  # Your search query
                "key": GOOGLE_API_KEY,        # Your API key
                "cx": GOOGLE_CX,              # Your Custom Search Engine ID
                "num": (2 * count - article_count), # Number of results to fetch
                "excludeTerms": "quote",            
                "sort": f"date:r:{week_ago_date}:{date}" 
            }
            response = requests.get(search_url, params=params)
            if response.status_code == 200:
                articles = response.json().get("items", [])
                filtered_articles = filter_articles(articles, time_filter, blocked_urls)
                tries +=1 
                article_count = len(filtered_articles)
            
        # return jsonify(filtered_articles) 
        # return jsonify([{"title": item["title"], "link": item["link"], "snippet": item["snippet"]} for item in articles])
        return jsonify([{"title": item["title"], "link": item["link"], "snippet": item["snippet"]} for item in filtered_articles])
    else:
        return jsonify({"error": "Failed to fetch search results", "details": response.json()}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)

    