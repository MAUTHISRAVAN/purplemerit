import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel

class UserProfiler:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        
    def create_user_profile(self, user_id, user_interactions, articles_df):
        """Create a user profile based on interaction history"""
        # Filter interactions for this user
        user_data = user_interactions[user_interactions['user_id'] == user_id]
        
        if len(user_data) == 0:
            return None
        
        # Get articles this user interacted with
        interacted_articles = articles_df[articles_df['article_id'].isin(user_data['article_id'])]
        
        # Calculate category preferences
        category_counts = interacted_articles['category'].value_counts()
        total_interactions = len(interacted_articles)
        category_preferences = {cat: count/total_interactions for cat, count in category_counts.items()}
        
        # Calculate article embeddings if embedding model is available
        article_embeddings = []
        if self.embedding_model:
            for _, article in interacted_articles.iterrows():
                article_id = article['article_id']
                # Get interaction details
                interaction = user_data[user_data['article_id'] == article_id].iloc[0]
                
                # Get article embedding
                embedding = self.embedding_model.get_article_embedding(article)
                
                # Weight by interaction strength (read time, liked)
                weight = 1.0
                if interaction['liked']:
                    weight *= 1.5
                weight *= min(interaction['read_time'] / 60, 5) / 5  # Normalize read time
                
                article_embeddings.append((embedding, weight))
        
        # Create user profile
        profile = {
            'user_id': user_id,
            'category_preferences': category_preferences,
            'interaction_count': len(user_data),
            'avg_read_time': user_data['read_time'].mean(),
            'liked_ratio': user_data['liked'].mean()
        }
        
        # Add embedding if available
        if article_embeddings:
            # Weighted average of article embeddings
            total_weight = sum(weight for _, weight in article_embeddings)
            weighted_embedding = sum(emb * weight for emb, weight in article_embeddings) / total_weight
            profile['embedding'] = weighted_embedding
            
        return profile
    
    def update_user_profile(self, existing_profile, new_interaction, article, decay_factor=0.9):
        """Update user profile with new interaction data"""
        if not existing_profile:
            return None
            
        # Update category preferences
        category = article['category']
        preferences = existing_profile['category_preferences']
        
        # Apply decay to existing preferences
        preferences = {k: v * decay_factor for k, v in preferences.items()}
        
        # Update preference for this category
        if category in preferences:
            preferences[category] += (1 - decay_factor)
        else:
            preferences[category] = (1 - decay_factor)
            
        # Normalize preferences
        total = sum(preferences.values())
        preferences = {k: v/total for k, v in preferences.items()}
        
        # Update interaction stats
        new_count = existing_profile['interaction_count'] + 1
        new_avg_read_time = (existing_profile['avg_read_time'] * (new_count - 1) + 
                            new_interaction['read_time']) / new_count
        new_liked_ratio = (existing_profile['liked_ratio'] * (new_count - 1) + 
                          new_interaction['liked']) / new_count
        
        # Update embedding if available
        if 'embedding' in existing_profile and self.embedding_model:
            article_embedding = self.embedding_model.get_article_embedding(article)
            
            # Weight for new interaction
            weight = 1.0
            if new_interaction['liked']:
                weight *= 1.5
            weight *= min(new_interaction['read_time'] / 60, 5) / 5
            
            # Update embedding with weighted average
            existing_embedding = existing_profile['embedding']
            updated_embedding = (existing_embedding * decay_factor + 
                               article_embedding * (1 - decay_factor) * weight)
            
            # Normalize embedding
            updated_embedding = updated_embedding / np.linalg.norm(updated_embedding)
            existing_profile['embedding'] = updated_embedding
        
        # Update profile
        existing_profile['category_preferences'] = preferences
        existing_profile['interaction_count'] = new_count
        existing_profile['avg_read_time'] = new_avg_read_time
        existing_profile['liked_ratio'] = new_liked_ratio
        
        return existing_profile
