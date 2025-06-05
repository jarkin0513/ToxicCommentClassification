## Toxic Comment Classification 
A multi-label NLP classification project focused on identifying toxic online content using models ranging from traditional machine learning to transformer-based architectures.

## Problem Overview 
This project addresses the challenge of detecting harmful or toxic content in online comments, which may include insults, threats, profanity, and identity-based hate. The task is treated as multi-label classification, meaning a single comment can exhibit more than one type of toxicity.

#### Made in collaboration with [Evin Bayer](https://github.com/EvinB)

## Dataset 
Source: Jigsaw Toxic Comment Classification Challenge
- 160,000+ labeled comments (from Wikipedia)
- Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
- 80% of data used for training
- 20% of data used for validation

## Libraries & Tools Used
- NumPy – Efficient numerical computations and array operations
- Pandas – Data loading and manipulation
- Scikit-learn – Traditional ML models (Logistic Regression, SVM), evaluation metrics
- Keras / TensorFlow – Implementation of LSTM + Feedforward networks
- PyTorch – Training and fine-tuning of DistilBERT and other transformer models
- Hugging Face Transformers – Pretrained DistilBERT and GPT2 models, tokenizers, and training utilities

## Models Implemented 
- Logistic Regression: Baseline using TF-IDF features; fast but limited semantic understanding
- Support Vector Machine: Better handling of class separation, but still shallow
- LSTM + Feedforward: Captures temporal structure in comments; struggles with rare classes
- DistilBERT + Feedforward Network: 	Best performing; contextual embeddings handle nuance and rare labels

## Primary Architecture (DistilBERT + FFN) 
- Tokenizer: DistilBERT tokenizer
- Encoder: Pre-trained DistilBERT
- Classifier: Fully connected feedforward layer (multi-label sigmoid output)
- Loss: Binary Cross Entropy with Logits
- Optimized with learning rate scheduling and early stopping

## Evalutation Metrics
- Macro/Micro F1 Score
- ROC-AUC (Macro)
- PR-AUC (Macro)
- Per-label confusion matrices

Best Macro F1 score: 0.9226
Threat category F1: 0.822 despite < 1k samples

![](https://github.com/jarkin0513/ToxicCommentClassification/blob/main/images/data.png)
![](https://github.com/jarkin0513/ToxicCommentClassification/blob/main/images/matrix.png)
