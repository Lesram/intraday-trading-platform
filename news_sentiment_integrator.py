#!/usr/bin/env python3
"""
🚀 NEWS SENTIMENT API INTEGRATOR
Real-time news sentiment integration for enhanced trading signals
Integrates Alpha Vantage and Finnhub news APIs into trading platform
"""

import logging
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NewsSentimentIntegrator:
    """
    Real-time news sentiment analysis using Alpha Vantage and Finnhub APIs
    Provides weighted sentiment scores for trading signal enhancement
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # API Configuration
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')

        # Cache for API responses (5 minute cache)
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes

        # API weights for combined sentiment
        self.api_weights = {
            'alpha_vantage': 0.6,  # Higher weight - more comprehensive sentiment
            'finnhub': 0.4         # Good news coverage
        }

        self.logger.info("🚀 News Sentiment Integrator initialized")
        self.logger.info(f"✅ Alpha Vantage: {'Available' if self.alpha_vantage_key else 'Not configured'}")
        self.logger.info(f"✅ Finnhub: {'Available' if self.finnhub_key else 'Not configured'}")

    def _get_cache_key(self, symbol: str, source: str) -> str:
        """Generate cache key for sentiment data"""
        return f"{source}_{symbol}_{int(time.time() / self.cache_duration)}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.sentiment_cache:
            return False

        cached_time = self.sentiment_cache[cache_key].get('timestamp', 0)
        return time.time() - cached_time < self.cache_duration

    def get_alpha_vantage_sentiment(self, symbol: str) -> dict:
        """
        Get news sentiment from Alpha Vantage API
        Returns sentiment score between -1 (very negative) and 1 (very positive)
        """
        cache_key = self._get_cache_key(symbol, 'alpha_vantage')

        if self._is_cache_valid(cache_key):
            self.logger.debug(f"📊 Using cached Alpha Vantage sentiment for {symbol}")
            return self.sentiment_cache[cache_key]['data']

        if not self.alpha_vantage_key or self.alpha_vantage_key == 'your_alpha_vantage_api_key_here':
            return {"sentiment_score": 0.0, "confidence": 0.0, "error": "API key not configured"}

        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.alpha_vantage_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if 'Error Message' in data:
                self.logger.error(f"❌ Alpha Vantage API error: {data['Error Message']}")
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": data['Error Message']}

            if 'Note' in data:
                self.logger.warning(f"⚠️ Alpha Vantage rate limited: {data['Note']}")
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": "Rate limited"}

            if 'feed' not in data or not data['feed']:
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": "No news data"}

            # Process sentiment data
            articles = data['feed'][:20]  # Use top 20 articles
            sentiment_scores = []
            relevance_scores = []

            for article in articles:
                try:
                    # Get ticker-specific sentiment
                    ticker_sentiments = article.get('ticker_sentiment', [])
                    for ticker_data in ticker_sentiments:
                        if ticker_data.get('ticker') == symbol:
                            sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0))
                            relevance_score = float(ticker_data.get('relevance_score', 0))

                            # Weight by relevance and recency
                            time_published = datetime.fromisoformat(article.get('time_published', '').replace('T', ' ').replace('Z', ''))
                            hours_old = (datetime.utcnow() - time_published).total_seconds() / 3600
                            time_weight = max(0.1, 1 - (hours_old / 24))  # Decay over 24 hours

                            weighted_sentiment = sentiment_score * relevance_score * time_weight
                            sentiment_scores.append(weighted_sentiment)
                            relevance_scores.append(relevance_score * time_weight)

                except (ValueError, KeyError, TypeError):
                    continue

            if not sentiment_scores:
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": "No valid sentiment data"}

            # Calculate weighted average sentiment
            if sum(relevance_scores) > 0:
                avg_sentiment = sum(s * r for s, r in zip(sentiment_scores, relevance_scores, strict=False)) / sum(relevance_scores)
            else:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            # Confidence based on number of articles and average relevance
            confidence = min(1.0, len(sentiment_scores) / 10) * (sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5)

            result = {
                "sentiment_score": max(-1.0, min(1.0, avg_sentiment)),  # Clamp to [-1, 1]
                "confidence": confidence,
                "articles_count": len(sentiment_scores),
                "source": "alpha_vantage"
            }

            # Cache the result
            self.sentiment_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }

            self.logger.info(f"📊 Alpha Vantage sentiment for {symbol}: {result['sentiment_score']:.3f} (confidence: {result['confidence']:.2f})")
            return result

        except Exception as e:
            self.logger.error(f"❌ Alpha Vantage API error for {symbol}: {str(e)}")
            return {"sentiment_score": 0.0, "confidence": 0.0, "error": str(e)}

    def get_finnhub_sentiment(self, symbol: str) -> dict:
        """
        Get news data from Finnhub API and calculate sentiment
        Returns sentiment score between -1 and 1
        """
        cache_key = self._get_cache_key(symbol, 'finnhub')

        if self._is_cache_valid(cache_key):
            self.logger.debug(f"📊 Using cached Finnhub sentiment for {symbol}")
            return self.sentiment_cache[cache_key]['data']

        if not self.finnhub_key or self.finnhub_key == 'your_finnhub_api_key_here':
            return {"sentiment_score": 0.0, "confidence": 0.0, "error": "API key not configured"}

        try:
            # Get recent news (last 7 days)
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": f"HTTP {response.status_code}"}

            news_data = response.json()

            if not news_data or not isinstance(news_data, list):
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": "No news data"}

            # Simple sentiment analysis based on headline keywords
            positive_keywords = [
                'beats', 'exceeds', 'strong', 'growth', 'gains', 'up', 'rise', 'surge',
                'bullish', 'positive', 'profit', 'revenue', 'upgrade', 'buy', 'outperform',
                'breakthrough', 'success', 'expansion', 'partnership', 'acquisition'
            ]

            negative_keywords = [
                'misses', 'falls', 'drops', 'decline', 'down', 'loss', 'weak', 'bearish',
                'negative', 'downgrade', 'sell', 'underperform', 'concern', 'risk',
                'investigation', 'lawsuit', 'scandal', 'bankruptcy', 'layoffs'
            ]

            sentiment_scores = []

            for article in news_data[:30]:  # Use last 30 articles
                headline = article.get('headline', '').lower()
                summary = article.get('summary', '').lower()
                text = f"{headline} {summary}"

                # Calculate sentiment based on keyword presence
                positive_count = sum(1 for keyword in positive_keywords if keyword in text)
                negative_count = sum(1 for keyword in negative_keywords if keyword in text)

                if positive_count + negative_count > 0:
                    article_sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    article_sentiment = 0.0

                # Weight by recency
                article_time = datetime.fromtimestamp(article.get('datetime', time.time()))
                hours_old = (datetime.now() - article_time).total_seconds() / 3600
                time_weight = max(0.1, 1 - (hours_old / 168))  # Decay over 7 days

                weighted_sentiment = article_sentiment * time_weight
                sentiment_scores.append(weighted_sentiment)

            if not sentiment_scores:
                return {"sentiment_score": 0.0, "confidence": 0.0, "error": "No sentiment data"}

            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(1.0, len(sentiment_scores) / 20) * 0.7  # Lower confidence than Alpha Vantage

            result = {
                "sentiment_score": max(-1.0, min(1.0, avg_sentiment)),
                "confidence": confidence,
                "articles_count": len(sentiment_scores),
                "source": "finnhub"
            }

            # Cache the result
            self.sentiment_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }

            self.logger.info(f"📊 Finnhub sentiment for {symbol}: {result['sentiment_score']:.3f} (confidence: {result['confidence']:.2f})")
            return result

        except Exception as e:
            self.logger.error(f"❌ Finnhub API error for {symbol}: {str(e)}")
            return {"sentiment_score": 0.0, "confidence": 0.0, "error": str(e)}

    def get_combined_sentiment(self, symbol: str) -> dict:
        """
        Get combined sentiment from all available news APIs
        Returns weighted sentiment score and confidence
        """
        self.logger.info(f"🎯 Getting combined sentiment for {symbol}")

        # Get sentiment from all sources
        av_sentiment = self.get_alpha_vantage_sentiment(symbol)
        fh_sentiment = self.get_finnhub_sentiment(symbol)

        # Collect valid sentiments
        valid_sentiments = []
        total_weight = 0

        if av_sentiment.get('confidence', 0) > 0:
            valid_sentiments.append({
                'score': av_sentiment['sentiment_score'],
                'confidence': av_sentiment['confidence'],
                'weight': self.api_weights['alpha_vantage'],
                'source': 'alpha_vantage'
            })
            total_weight += self.api_weights['alpha_vantage']

        if fh_sentiment.get('confidence', 0) > 0:
            valid_sentiments.append({
                'score': fh_sentiment['sentiment_score'],
                'confidence': fh_sentiment['confidence'],
                'weight': self.api_weights['finnhub'],
                'source': 'finnhub'
            })
            total_weight += self.api_weights['finnhub']

        if not valid_sentiments:
            return {
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "sources_used": [],
                "error": "No valid sentiment data available"
            }

        # Calculate weighted sentiment
        weighted_sentiment = 0
        weighted_confidence = 0

        for sentiment in valid_sentiments:
            weight = sentiment['weight'] / total_weight
            weighted_sentiment += sentiment['score'] * weight * sentiment['confidence']
            weighted_confidence += sentiment['confidence'] * weight

        result = {
            "sentiment_score": max(-1.0, min(1.0, weighted_sentiment)),
            "confidence": weighted_confidence,
            "sources_used": [s['source'] for s in valid_sentiments],
            "individual_scores": {
                "alpha_vantage": av_sentiment,
                "finnhub": fh_sentiment
            }
        }

        self.logger.info(f"🎯 Combined sentiment for {symbol}: {result['sentiment_score']:.3f} "
                        f"(confidence: {result['confidence']:.2f}, sources: {result['sources_used']})")

        return result

    def get_sentiment_for_trading(self, symbol: str) -> float:
        """
        Get sentiment score optimized for trading signals
        Returns value between 0.0 and 1.0 where:
        - 0.5 = neutral sentiment
        - 0.0 = very negative sentiment  
        - 1.0 = very positive sentiment
        """
        combined_sentiment = self.get_combined_sentiment(symbol)

        if combined_sentiment.get('confidence', 0) < 0.3:
            # Low confidence, return neutral
            return 0.5

        # Convert from [-1, 1] to [0, 1] range
        sentiment_score = combined_sentiment['sentiment_score']
        trading_sentiment = (sentiment_score + 1) / 2

        # Apply confidence weighting
        confidence = combined_sentiment['confidence']

        # Blend with neutral (0.5) based on confidence
        final_sentiment = (trading_sentiment * confidence) + (0.5 * (1 - confidence))

        return max(0.0, min(1.0, final_sentiment))

# Global instance
_news_sentiment_integrator = None

def get_news_sentiment_integrator() -> NewsSentimentIntegrator:
    """Get global news sentiment integrator instance"""
    global _news_sentiment_integrator
    if _news_sentiment_integrator is None:
        _news_sentiment_integrator = NewsSentimentIntegrator()
    return _news_sentiment_integrator

def get_sentiment_for_symbol(symbol: str) -> float:
    """
    Quick function to get sentiment score for trading
    Returns value between 0.0 and 1.0 (0.5 = neutral)
    """
    integrator = get_news_sentiment_integrator()
    return integrator.get_sentiment_for_trading(symbol)

# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    integrator = NewsSentimentIntegrator()

    # Test with Apple
    print("Testing AAPL sentiment:")
    sentiment = integrator.get_combined_sentiment("AAPL")
    print(f"Combined sentiment: {sentiment}")

    trading_score = integrator.get_sentiment_for_trading("AAPL")
    print(f"Trading sentiment score: {trading_score}")
