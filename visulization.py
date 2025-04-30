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
