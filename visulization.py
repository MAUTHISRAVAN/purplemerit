{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# News Recommendation System Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.manifold import TSNE\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "from src.data_collector import NewsDataCollector\n",
    "from src.user_profiler import UserProfiler\n",
    "from src.embedding_model import NewsEmbeddingModel\n",
    "from src.collaborative_filter import CollaborativeFilter\n",
    "from src.recommender import NewsRecommender"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize collector\n",
    "collector = NewsDataCollector()\n",
    "\n",
    "# Load sample dataset\n",
    "articles_df = collector.load_sample_dataset('../data/sample_data.csv')\n",
    "print(f\"Loaded {len(articles_df)} articles\")\n",
    "\n",
    "# Simulate user interactions\n",
    "user_interactions, user_profiles = collector.simulate_user_interactions(\n",
    "    num_users=100, articles_df=articles_df\n",
    ")\n",
    "print(f\"Generated {len(user_interactions)} interactions for {len(user_profiles)} users\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initialize Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize embedding model\n",
    "embedding_model = NewsEmbeddingModel()\n",
    "\n",
    "# Initialize user profiler\n",
    "user_profiler = UserProfiler(embedding_model)\n",
    "\n",
    "# Train collaborative filter\n",
    "cf = CollaborativeFilter()\n",
    "cf.train(user_interactions)\n",
    "\n",
    "# Initialize recommender\n",
    "recommender = NewsRecommender(embedding_model, cf, user_profiler)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Visualize User Preferences"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get user profiles for visualization\n",
    "user_id = 0\n",
    "user_profile = user_profiler.create_user_profile(\n",
    "    user_id, user_interactions, articles_df\n",
    ")\n",
    "\n",
    "# Visualize category preferences\n",
    "plt.figure(figsize=(10, 6))\n",
    "categories = list(user_
        # Visualize category preferences
    plt.figure(figsize=(10, 6))
    categories = list(user_profile['category_preferences'].keys())
    preferences = list(user_profile['category_preferences'].values())
    
    sns.barplot(x=categories, y=preferences)
    plt.title(f"Category Preferences for User {user_id}")
    plt.xlabel("Category")
    plt.ylabel("Preference Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
                   # Visualize article embeddings by category
    # Get embeddings for a sample of articles
    sample_size = min(100, len(articles_df))
    sample_articles = articles_df.sample(sample_size)
    
    embeddings = []
    categories = []
    
    for _, article in sample_articles.iterrows():
        embedding = embedding_model.get_article_embedding(article)
        embeddings.append(embedding)
        categories.append(article['category'])
    
    # Reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    viz_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'category': categories
    })
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=viz_df, x='x', y='y', hue='category', palette='viridis')
    plt.title("Article Embeddings by Category (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Category")
    plt.tight_layout()
    plt.show()
              # Visualize user-article interactions
    # Get top users by interaction count
    top_users = user_interactions['user_id'].value_counts().head(10).index
    
    # Filter interactions for these users
    top_user_interactions = user_interactions[user_interactions['user_id'].isin(top_users)]
    
    # Get category for each interaction
    interaction_categories = []
    for _, row in top_user_interactions.iterrows():
        article = articles_df[articles_df['article_id'] == row['article_id']]
        if not article.empty:
            interaction_categories.append(article['category'].iloc[0])
        else:
            interaction_categories.append('unknown')
    
    top_user_interactions['category'] = interaction_categories
    
    # Create heatmap of user-category interactions
    user_category_counts = pd.crosstab(
        top_user_interactions['user_id'], 
        top_user_interactions['category']
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(user_category_counts, annot=True, cmap='YlGnBu', fmt='d')
    plt.title("User-Category Interaction Counts")
    plt.xlabel("Category")
    plt.ylabel("User ID")
    plt.tight_layout()
    plt.show()
           # Visualize recommendation diversity
    # Get recommendations for multiple users
    all_recommendations = []
    
    for user_id in range(5):  # Get recommendations for first 5 users
        recs = recommender.get_recommendations(
            user_id, user_interactions, articles_df, top_n=10
        )
        
        for rec in recs:
            rec['user_id'] = user_id
            all_recommendations.append(rec)
    
    # Convert to DataFrame
    recs_df = pd.DataFrame(all_recommendations)
    
    # Plot category distribution in recommendations
    plt.figure(figsize=(12, 6))
    sns.countplot(data=recs_df, x='category', hue='user_id')
    plt.title("Category Distribution in Recommendations")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="User ID")
    plt.tight_layout()
    plt.show()
    
    # Plot recommendation scores by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=recs_df, x='category', y='score')
    plt.title("Recommendation Scores by Category")
    plt.xlabel("Category")
    plt.ylabel("Recommendation Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()                 
