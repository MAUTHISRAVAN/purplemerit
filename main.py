import os
import pandas as pd
import numpy as np
from data_collector import NewsDataCollector
from user_profiler import UserProfiler
from embedding_model import NewsEmbeddingModel
from collaborative_filter import CollaborativeFilter
from recommender import NewsRecommender
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Initialize components
    api_key = os.getenv('NEWS_API_KEY')
    collector = NewsDataCollector(api_key)
    
    # Fetch or load articles
    try:
        if api_key:
            articles_df = collector.fetch_articles_from_api()
            print(f"Fetched {len(articles_df)} articles from NewsAPI")
        else:
            articles_df = collector.load_sample_dataset('data/sample_data.csv')
            print(f"Loaded {len(articles_df)} sample articles")
    except Exception as e:
        print(f"Error fetching articles: {e}")
        articles_df = collector.load_sample_dataset('data/sample_data.csv')
        print(f"Loaded {len(articles_df)} sample articles")
    
    # Simulate user interactions
    user_interactions, user_profiles = collector.simulate_user_interactions(
        num_users=100, articles_df=articles_df
    )
    print(f"Generated {len(user_interactions)} interactions for {len(user_profiles)} users")
    
    # Initialize embedding model
    embedding_model = NewsEmbeddingModel()
    print("Initialized embedding model")
    
    # Fine-tune embedding model (optional - can be time-consuming)
    # embedding_model.fine_tune(articles_df, epochs=1)
    # print("Fine-tuned embedding model")
    
    # Initialize user profiler
    user_profiler = UserProfiler(embedding_model)
    print("Initialized user profiler")
    
    # Train collaborative filter
    cf = CollaborativeFilter()
    cf.train(user_interactions)
    print("Trained collaborative filter")
    
    # Initialize recommender
    recommender = NewsRecommender(embedding_model, cf, user_profiler)
    print("Initialized recommender")
    
    # Get recommendations for a sample user
    user_id = 0
    recommendations = recommender.get_recommendations(
        user_id, user_interactions, articles_df, top_n=5
    )
    
    # Display recommendations
    print(f"\nTop recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   Category: {rec['category']}")
        print(f"   Score: {rec['score']:.3f}")
        print(f"   Description: {rec['description']}")
    
    # Simulate a new interaction
    if recommendations:
        new_article_id = recommendations[0]['article_id']
        new_article = articles_df[articles_df['article_id'] == new_article_id].iloc[0]
        
        new_interaction = {
            'user_id': user_id,
            'article_id': new_article_id,
            'read_time': 120,  # 2 minutes
            'liked': 1,
            'clicked': 1,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Get user profile
        user_profile = user_profiler.create_user_profile(
            user_id, user_interactions, articles_df
        )
        
        # Update recommendations
        updated_recommendations, updated_profile = recommender.update_recommendations(
            user_id, new_interaction, new_article, user_profile, 
            user_interactions, articles_df, top_n=5
        )
        
        print("\nAfter interaction with article:", new_article['title'])
        print("\nUpdated recommendations:")
        for i, rec in enumerate(updated_recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Category: {rec['category']}")
            print(f"   Score: {rec['score']:.3f}")
            print(f"   Description: {rec['description']}")

if __name__ == "__main__":
    main()
