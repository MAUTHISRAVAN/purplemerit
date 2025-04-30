!pip install newsapi-python
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os
import json

class NewsDataCollector:
    def __init__(self, api_key=None):
        self.api_key = api_key
        if api_key:
            self.api = NewsApiClient(api_key=api_key)
        
    def fetch_articles_from_api(self, days_back=7, categories=None):
        """Fetch articles from NewsAPI"""
        if not self.api_key:
            raise ValueError("API key is required to fetch articles")
            
        today = datetime.now()
        past = today - timedelta(days=days_back)
        
        all_articles = []
        
        if not categories:
            categories = ['business', 'entertainment', 'general', 'health', 
                         'science', 'sports', 'technology']
        
        for category in categories:
            try:
                response = self.api.get_top_headlines(
                    category=category,
                    language='en',
                    page_size=100
                )
                
                if response['status'] == 'ok':
                    for article in response['articles']:
                        article['category'] = category
                    all_articles.extend(response['articles'])
            except Exception as e:
                print(f"Error fetching {category} articles: {e}")
        
        df = pd.DataFrame(all_articles)
        df['article_id'] = range(len(df))
        return df
    
    def load_sample_dataset(self, filepath=None):
        """Load a sample news dataset if API is not available"""
        if filepath and os.path.exists(filepath):
            return pd.read_csv(filepath)
        
        # Create a sample dataset
        categories = ['business', 'entertainment', 'technology', 'sports', 'politics']
        sample_articles = []
        
        for i in range(500):
            category = np.random.choice(categories)
            article = {
                'article_id': i,
                'title': f"Sample Article {i} about {category}",
                'description': f"This is a sample description for article {i} in the {category} category.",
                'content': f"Sample content for article {i}. This article discusses various aspects of {category}.",
                'category': category,
                'publishedAt': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                'source': {'name': f"Source {np.random.randint(1, 10)}"}
            }
            sample_articles.append(article)
            
        df = pd.DataFrame(sample_articles)
        
        if filepath:
            df.to_csv(filepath, index=False)
            
        return df
    
    def simulate_user_interactions(self, num_users, articles_df):
        """Simulate user interaction data"""
        # Create user profiles with category preferences
        categories = articles_df['category'].unique()
        user_profiles = []
        
        for user_id in range(num_users):
            # Each user has different category preferences
            preferences = {cat: np.random.uniform(0.1, 1.0) for cat in categories}
            # Normalize preferences
            total = sum(preferences.values())
            preferences = {k: v/total for k, v in preferences.items()}
            
            user_profiles.append({
                'user_id': user_id,
                'preferences': preferences
            })
        
        # Generate interactions based on preferences
        interactions = []
        
        for user in user_profiles:
            user_id = user['user_id']
            preferences = user['preferences']
            
            # Determine number of interactions for this user
            num_interactions = np.random.randint(10, 50)
            
            # Select articles based on category preferences
            for _ in range(num_interactions):
                # Select category based on preferences
                category = np.random.choice(
                    list(preferences.keys()),
                    p=list(preferences.values())
                )
                
                # Filter articles by category
                category_articles = articles_df[articles_df['category'] == category]
                
                if len(category_articles) > 0:
                    # Select a random article from this category
                    article = category_articles.sample(1).iloc[0]
                    
                    # Create interaction
                    interaction = {
                        'user_id': user_id,
                        'article_id': article['article_id'],
                        'read_time': np.random.randint(10, 300),  # seconds
                        'liked': np.random.choice([0, 1], p=[0.3, 0.7]),  # More likely to like preferred categories
                        'clicked': 1,
                        'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 14))).isoformat()
                    }
                    interactions.append(interaction)
        
        interactions_df = pd.DataFrame(interactions)
        return interactions_df, pd.DataFrame(user_profiles)
