#!/usr/bin/env python3
"""
🧪 NEWS & SENTIMENT API TESTING SCRIPT
Tests all your newly configured API keys to verify they're working properly
"""

import os
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_alpha_vantage():
    """Test Alpha Vantage News Sentiment API"""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key or api_key == 'your_alpha_vantage_api_key_here':
        return {"status": "❌ API Key Not Set", "error": "No valid API key"}

    try:
        # Test Alpha Vantage News & Sentiment
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if 'Error Message' in data:
            return {"status": "❌ API Error", "error": data['Error Message']}
        elif 'Note' in data:
            return {"status": "⚠️ Rate Limited", "error": data['Note']}
        elif 'feed' in data:
            return {
                "status": "✅ Working",
                "articles_count": len(data['feed']),
                "sample_headline": data['feed'][0]['title'] if data['feed'] else "No articles"
            }
        else:
            return {"status": "❓ Unknown Response", "data": str(data)[:200]}
    except Exception as e:
        return {"status": "❌ Connection Error", "error": str(e)}

def test_newsapi():
    """Test NewsAPI"""
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key or api_key == 'your_news_api_key_here':
        return {"status": "❌ API Key Not Set", "error": "No valid API key"}

    try:
        # Test NewsAPI Everything endpoint
        url = f"https://newsapi.org/v2/everything?q=Apple&apiKey={api_key}&pageSize=5"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get('status') == 'error':
            return {"status": "❌ API Error", "error": data.get('message', 'Unknown error')}
        elif data.get('status') == 'ok':
            return {
                "status": "✅ Working",
                "total_results": data.get('totalResults', 0),
                "articles_retrieved": len(data.get('articles', [])),
                "sample_headline": data['articles'][0]['title'] if data.get('articles') else "No articles"
            }
        else:
            return {"status": "❓ Unknown Response", "data": str(data)[:200]}
    except Exception as e:
        return {"status": "❌ Connection Error", "error": str(e)}

def test_finnhub():
    """Test Finnhub API"""
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key or api_key == 'your_finnhub_api_key_here':
        return {"status": "❌ API Key Not Set", "error": "No valid API key"}

    try:
        # Test Finnhub Company News
        url = f"https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2025-08-01&to=2025-08-08&token={api_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 401:
            return {"status": "❌ Unauthorized", "error": "Invalid API key"}
        elif response.status_code == 429:
            return {"status": "⚠️ Rate Limited", "error": "API rate limit exceeded"}

        data = response.json()

        if isinstance(data, list):
            return {
                "status": "✅ Working",
                "news_count": len(data),
                "sample_headline": data[0]['headline'] if data else "No news articles"
            }
        else:
            return {"status": "❓ Unexpected Response", "data": str(data)[:200]}
    except Exception as e:
        return {"status": "❌ Connection Error", "error": str(e)}

def test_polygon():
    """Test Polygon API"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key or api_key == 'your_polygon_api_key_here':
        return {"status": "⏳ Not Configured", "note": "Polygon API key not set (optional)"}

    try:
        # Test Polygon News
        url = f"https://api.polygon.io/v2/reference/news?ticker=AAPL&published_utc.gte=2025-08-01&limit=5&apikey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get('status') == 'ERROR':
            return {"status": "❌ API Error", "error": data.get('error', 'Unknown error')}
        elif data.get('status') == 'OK':
            return {
                "status": "✅ Working",
                "news_count": len(data.get('results', [])),
                "sample_headline": data['results'][0]['title'] if data.get('results') else "No articles"
            }
        else:
            return {"status": "❓ Unknown Response", "data": str(data)[:200]}
    except Exception as e:
        return {"status": "❌ Connection Error", "error": str(e)}

def main():
    print("🧪 TESTING NEWS & SENTIMENT APIs")
    print("=" * 50)
    print(f"Test Time: {datetime.now()}")
    print()

    # Test all APIs
    apis = {
        "Alpha Vantage": test_alpha_vantage(),
        "NewsAPI": test_newsapi(),
        "Finnhub": test_finnhub(),
        "Polygon.io": test_polygon()
    }

    # Display results
    for api_name, result in apis.items():
        status = result.get('status', 'Unknown')
        print(f"📊 {api_name:12} | {status}")

        if 'articles_count' in result:
            print(f"   📰 Articles: {result['articles_count']}")
        if 'total_results' in result:
            print(f"   📰 Total Results: {result['total_results']}")
        if 'news_count' in result:
            print(f"   📰 News Count: {result['news_count']}")
        if 'sample_headline' in result:
            print(f"   📰 Sample: {result['sample_headline'][:80]}...")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        if 'note' in result:
            print(f"   ℹ️  Note: {result['note']}")
        print()

    # Summary
    working_apis = [name for name, result in apis.items() if result.get('status', '').startswith('✅')]
    print(f"📈 SUMMARY: {len(working_apis)}/4 APIs working properly")

    if working_apis:
        print(f"✅ Working APIs: {', '.join(working_apis)}")
        print("🎯 Your trading platform now has access to real news sentiment data!")
    else:
        print("⚠️ No APIs are currently working. Check your API keys and internet connection.")

    return len(working_apis) > 0

if __name__ == "__main__":
    main()
