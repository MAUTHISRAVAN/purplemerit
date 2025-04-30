import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NewsRecommender:
    def __init__(self, embedding_model, collaborative_filter, user_profiler):
        self.embedding_model = embedding_model
        self.collaborative_filter = collaborative_filter
        self.user_profiler = user_profiler
        
    def get_recommendations(self, user_id, user_interactions, articles_df, top_n=10, 
                           cf_weight=0.4, content_weight=0.6):
        """Get personalized recommendations for a user"""
        # Get user profile
        user_profile = self.user_profiler.create_user_profile(
            user_id, user_interactions, articles_df
        )
        
        if not user_profile:
            # New user, use popularity-based recommendations
            return self._get_popular_recommendations(articles_df, top_n)
        
        # Get collaborative filtering recommendations
        if self.collaborative_filter.trained:
            cf_predictions = self.collaborative_filter.predict_ratings(
                user_id, articles_df['article_id'].tolist()
            )
            
            # Normalize CF scores to [0, 1]
            if not cf_predictions.empty:
                min_rating = cf_predictions['predicted_rating'].min()
                max_rating = cf_predictions['predicted_rating'].max()
                if max_rating > min_rating:
                    cf_predictions['cf_score'] = (cf_predictions['predicted_rating'] - min_rating) / (max_rating - min_rating)
                else:
                    cf_predictions['cf_score'] = 0.5  # Default if all predictions are the same
            
        # Get content-based recommendations
        content_scores = []
        
        if 'embedding' in user_profile:
            user_embedding = user_profile['embedding']
            
            for _, article in articles_df.iterrows():
                article_id = article['article_id']
                
                # Skip articles the user has already interacted with
                if article_id in user_interactions[user_interactions['user_id'] == user_id]['article_id'].values:
                    continue
                
                # Get article embedding
                article_embedding = self.embedding_model.get_article_embedding(article)
                
                # Calculate similarity
                similarity = cosine_similarity([user_embedding], [article_embedding])[0][0]
                
                # Add category preference boost
                category = article['category']
                category_boost = user_profile['category_preferences'].get(category, 0)
                
                # Combine similarity and category preference
                content_score = 0.7 * similarity + 0.3 * category_boost
                
                content_scores.append({
                    'article_id': article_id,
                    'content_score': content_score
                })
                
        content_df = pd.DataFrame(content_scores)
        
        # Combine recommendations
        if self.collaborative_filter.trained and not cf_predictions.empty and not content_df.empty:
            # Merge CF and content scores
            merged = pd.merge(cf_predictions, content_df, on='article_id', how='outer').fillna(0)
            
            # Calculate final score
            merged['final_score'] = (cf_weight * merged['cf_score'] + 
                                   content_weight * merged['content_score'])
        elif not content_df.empty:
            # Only content-based
            merged = content_df.copy()
            merged['final_score'] = merged['content_score']
        elif self.collaborative_filter.trained and not cf_predictions.empty:
            # Only CF-based
            merged = cf_predictions.copy()
            merged['final_score'] = merged['cf_score']
        else:
            # Fallback to popularity
            return self._get_popular_recommendations(articles_df, top_n)
        
        # Sort by final score
        merged = merged.sort_values('final_score', ascending=False)
        
        # Get top N recommendations
        top_articles = merged.head(top_n)['article_id'].tolist()
        
        # Get article details
        recommendations = []
        for article_id in top_articles:
            article = articles_df[articles_df['article_id'] == article_id].iloc[0]
            recommendations.append({
                'article_id': article_id,
                'title': article['title'],
                'description': article['description'] if pd.notna(article['description']) else '',
                'category': article['category'],
                'score': float(merged[merged['article_id'] == article_id]['final_score'].iloc[0])
            })
            
        return recommendations
    
    def _get_popular_recommendations(self, articles_df, top_n=10):
        """Get popularity-based recommendations for new users"""
        # Sort by recency
        recent_articles = articles_df.sort_values('publishedAt', ascending=False)
        
        # Get top N recent articles
        recommendations = []
        for _, article in recent_articles.head(top_n).iterrows():
            recommendations.append({
                'article_id': article['article_id'],
                'title': article['title'],
                'description': article['description'] if pd.notna(article['description']) else '',
                'category': article['category'],
                'score': 0.5  # Default score
            })
            
        return recommendations
    
    def update_recommendations(self, user_id, new_interaction, article, user_profile, 
                             user_interactions, articles_df, top_n=10):
        """Update recommendations based on new interaction"""
        # Update user profile
        updated_profile = self.user_profiler.update_user_profile(
            user_profile, new_interaction, article
        )
        
        # Add new interaction to dataset
        updated_interactions = user_interactions.append(new_interaction, ignore_index=True)
        
        # Get new recommendations
        return self.get_recommendations(
            user_id, updated_interactions, articles_df, top_n
        ), updated_profile
