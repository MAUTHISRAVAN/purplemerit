# purplemerit
# AI News Recommendation System

This project implements an AI-powered news recommendation system that combines collaborative filtering with fine-tuned embeddings to deliver personalized news feeds to users.

## Features

- **Data Collection**: Fetches news articles from NewsAPI or uses a simulated dataset
- **User Profiling**: Creates user profiles based on interaction history and content preferences
- **Embedding Model**: Uses transformer-based embeddings to represent articles and user preferences
- **Collaborative Filtering**: Implements SVD-based collaborative filtering for user-item recommendations
- **Hybrid Recommendation**: Combines content-based and collaborative filtering approaches
- **Adaptive Learning**: Updates user profiles and recommendations based on new interactions
- **Visualization**: Includes notebooks for visualizing user preferences and recommendation results

## Project Structure

## How It Works

1. **Data Collection**: The system collects news articles and user interaction data.
2. **User Profiling**: User profiles are created based on interaction history and content preferences.
3. **Embedding Generation**: Articles are represented as embeddings using transformer models.
4. **Collaborative Filtering**: The system learns patterns from user-article interactions.
5. **Recommendation Generation**: Recommendations are generated using a hybrid approach that combines collaborative filtering scores with content similarity.
6. **Adaptation**: The system updates user profiles and recommendations based on new interactions.

## Customization

- Adjust the weights in the recommender to balance collaborative filtering and content-based approaches
- Fine-tune the embedding model on your specific news dataset
- Modify the user profiling logic to capture different aspects of user behavior
