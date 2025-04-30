import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

class NewsDataset(Dataset):
    def __init__(self, articles, tokenizer, max_length=512):
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles.iloc[idx]
        
        # Combine title and content
        text = article['title']
        if pd.notna(article['content']) and article['content']:
            text += " " + article['content']
            
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert to tensors
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(article['category_id'], dtype=torch.long)
        }

class NewsEmbeddingModel:
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.fine_tuned = False
        
    def get_article_embedding(self, article):
        """Get embedding for a single article"""
        # Combine title and content
        text = article['title']
        if pd.notna(article['content']) and article['content']:
            text += " " + article['content']
            
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            padding='max_length',
            truncation=True
        ).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding or mean of all token embeddings
            if self.fine_tuned:
                # For fine-tuned model, use the CLS token
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            else:
                # For base model, use mean of all tokens
                # Use attention mask to ignore padding tokens
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                masked_embeddings = outputs.last_hidden_state * mask
                summed = torch.sum(masked_embeddings, 1)
                counts = torch.sum(mask, 1)
                embedding = (summed / counts).cpu().numpy()[0]
                
        return embedding
    
    def fine_tune(self, articles_df, epochs=3):
        """Fine-tune the embedding model on news articles"""
        # Prepare category mapping
        categories = articles_df['category'].unique()
        category_to_id = {cat: i for i, cat in enumerate(categories)}
        articles_df['category_id'] = articles_df['category'].map(category_to_id)
        
        # Create dataset
        dataset = NewsDataset(articles_df, self.tokenizer)
        
        # Load sequence classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(categories)
        ).to(self.device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train model
        trainer.train()
        
        # Switch back to base model for embeddings
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.load_state_dict(trainer.model.base_model.state_dict())
        
        self.fine_tuned = True
        return self
