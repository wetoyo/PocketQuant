import os
import requests
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
    time_filter = int(time_filter)  
    if not stock_name:
        return jsonify({"error": "Missing 'stock' parameter"}), 400

    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return jsonify({"error": "Missing API key or CX ID"}), 500
    
    # Construct the search URL using Google's API
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": f"{stock_name} stock news",  # Your search query
        "key": GOOGLE_API_KEY,        # Your API key
        "cx": GOOGLE_CX,              # Your Custom Search Engine ID
        "num": 10,                    # Number of results to fetch
        "dateRestrict": f"d{time_filter // 24}"  # Limit results to the last 'hours' in days
    }
    
    response = requests.get(search_url, params=params)

    if response.status_code == 200:

        blocked_urls = load_blocked_urls()
        
        # Get the search results (articles)
        articles = response.json().get("items", [])
        
        # Filter the articles based on time filter and blocked URLs
        filtered_articles = filter_articles(articles, time_filter, blocked_urls)
        
        # return jsonify(filtered_articles) 
        return jsonify([{"title": item["title"], "link": item["link"], "snippet": item["snippet"]} for item in filtered_articles])
    else:
        return jsonify({"error": "Failed to fetch search results", "details": response.json()}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)