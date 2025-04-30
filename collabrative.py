import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

class CollaborativeFilter:
    def __init__(self):
        self.model = SVD()
        self.trained = False
        self.user_items = defaultdict(list)
        
    def train(self, user_interactions):
        """Train the collaborative filtering model"""
        # Prepare data for Surprise
        reader = Reader(rating_scale=(0, 1))
        
        # Convert interactions to ratings
        # Use 'liked' as the rating, or could use normalized read_time
        ratings_data = user_interactions[['user_id', 'article_id', 'liked']]
        
        # Create Surprise dataset
        data = Dataset.load_from_df(ratings_data, reader)
        
        # Split into train and test
        trainset, _ = train_test_split(data, test_size=0.2)
        
        # Train model
        self.model.fit(trainset)
        
        # Store user-item interactions for filtering
        for _, row in user_interactions.iterrows():
            self.user_items[row['user_id']].append(row['article_id'])
            
        self.trained = True
        return self
    
    def predict_ratings(self, user_id, article_ids, filter_seen=True):
        """Predict ratings for a list of articles"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Filter out articles the user has already seen
        if filter_seen:
            unseen_article_ids = [aid for aid in article_ids 
                                if aid not in self.user_items[user_id]]
        else:
            unseen_article_ids = article_ids
            
        # Make predictions
        predictions = []
        for article_id in unseen_article_ids:
            pred = self.model.predict(user_id, article_id)
            predictions.append({
                'article_id': article_id,
                'predicted_rating': pred.est
            })
            
        return pd.DataFrame(predictions)
