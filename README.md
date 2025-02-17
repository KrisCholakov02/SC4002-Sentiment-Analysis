# SC4002 – Group Project: Sentiment Analysis Ablation Study

This repository contains the code and documentation for Group 3’s sentiment analysis project for SC4002. Our work focuses on building and evaluating various NLP models on the Rotten Tomatoes movie review dataset, with an emphasis on ablation studies, effective out-of-vocabulary (OoV) mitigation, and the exploration of advanced transformer architectures.

## Group Members
- Cholakov Kristiyan Kamenov (U2123543B)
- Jerome Wang (U2120160F)
- Goh Soong Wen Ryan (U2120980L)
- Denzyl David Peh (U2122190F)
- Clement Tan Kang Hao (U2122052K)
- Tiwana Teg Singh (U2122816B)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Contributions](#key-contributions)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Analysis](#results-and-analysis)
- [References](#references)

## Project Overview
In this project, we developed a sentiment classification system using the Rotten Tomatoes movie review dataset. Our objective was to investigate a range of approaches—from classical RNN-based architectures to state-of-the-art transformer models—and to determine the impact of embedding choice and OoV mitigation strategies on model performance. Key experiments include:
- Comparing Word2Vec and GloVe embeddings (with extensive pre-processing to reduce OoV words).
- Evaluating several RNN-based models (using last state, max pooling, average pooling, and attention pooling) and their performance improvements.
- Enhancing model performance through transformer-based approaches, including a custom transformer architecture with attention-based pooling and robust regularization.
- Combining multiple models via an ensemble approach.
- Fine-tuning pre-trained models (BERT, DistilBERT, RoBERTa, XLNet, ALBERT, and ELECTRA) to benchmark against our custom models.

## Key Contributions
- **OoV Mitigation:** Extensive pre-processing (apostrophe removal, hyphen splitting, lemmatization, and stemming) dramatically reduced OoV words for GloVe (from 1865 to 602), improving overall model performance.
- **Model Ablation Study:** We systematically compared multiple RNN architectures (including attention pooling variants) to understand which sequence representations best capture sentiment.
- **Transformer Enhancements:** Our custom transformer models, incorporating learnable positional encoding and attention-based pooling, achieved competitive performance relative to state-of-the-art pretrained models.
- **Ensemble Modeling:** Combining predictions from individual models (RNNs, BiLSTM, BiGRU, and CNN) further improved robustness and accuracy.
- **Pre-trained Model Benchmarking:** Fine-tuning advanced models such as ELECTRA yielded a test accuracy of over 90%, establishing a strong performance baseline.

## File Structure
```
.
├── data/                            # (Optional) Directory for dataset files (e.g., Rotten Tomatoes reviews)
├── glove/                           # Pre-trained embedding files
│   ├── glove.6B.50d.txt
│   ├── glove.6B.300d.txt
│   ├── glove.6B.50d.word2vec.txt
│   └── glove.6B.300d.word2vec.txt
├── histories/                       # Pickle files storing model training histories
│   ├── bigru_history.pkl
│   ├── bilstm_history.pkl
│   ├── cnn_history.pkl
│   ├── rnn_attention_pool_history.pkl
│   ├── rnn_attention_pool_trainable_embed_history.pkl
│   ├── rnn_attention_pool_trainable_embed_mit_oov_history.pkl
│   ├── rnn_avg_pool_history.pkl
│   ├── rnn_last_state_history.pkl
│   └── rnn_max_pool_history.pkl
├── notebooks/                       # Jupyter notebooks for experiments and model training
│   ├── main_word2vec.ipynb          # Models using Word2Vec embeddings
│   ├── main_glove.ipynb             # Models using GloVe embeddings
│   ├── ensemble.ipynb               # Ensemble model implementation
│   └── pretrained_models.ipynb      # Fine-tuning pre-trained Transformer models
├── requirements.txt                 # Python package dependencies
└── README.md                        # Project documentation (this file)
```

## Installation
1. **Clone the repository:**
   ```
   git clone https://github.com/KrisCholakov02/SC4002-Sentiment-Analysis.git
   ```
2. **Navigate to the project directory:**
   ```
   cd SC4002-Sentiment-Analysis
   ```
3. **Install the required Python packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage
- **Jupyter Notebooks:**  
  The experiments are organized in the notebooks found in the `notebooks/` folder. Open them using Jupyter Notebook or JupyterLab to run the code and reproduce the experiments:
  - `main_word2vec.ipynb` – Implements models with Word2Vec embeddings.
  - `main_glove.ipynb` – Implements models with GloVe embeddings, which show superior performance.
  - `ensemble.ipynb` – Demonstrates the ensemble model combining individual model predictions.
  - `pretrained_models.ipynb` – Fine-tunes and evaluates state-of-the-art Transformer models.

## Results and Analysis
Our experimental results demonstrate:
- **RNN Architectures:**  
  Attention pooling improved baseline RNN performance significantly over simpler pooling methods. Further enhancements, such as making embeddings trainable and mitigating OoV issues, boosted accuracy markedly.
- **Bidirectional Models:**  
  Both BiLSTM and BiGRU outperformed unidirectional RNNs by capturing richer contextual information.
- **Transformer Models:**  
  Custom transformer architectures, after hyperparameter tuning, achieved test accuracies in the vicinity of 79%, while fine-tuned pre-trained models (notably ELECTRA) reached over 90% accuracy.
- **Ensemble Approach:**  
  Averaging predictions from multiple architectures resulted in a robust system with improved generalization.
  
For an in-depth description of the experiments, ablation studies, and model configurations, please refer to the attached project report.

## References
1. Bahdanau, D., Cho, K., & Bengio, Y. (2016). Neural machine translation by jointly learning to align and translate.
2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space.
3. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation.
4. Senel, L. K., Utlu, I., Yücesoy, V., Koç, A., & Çukur, T. (2018). Semantic structure and interpretability of word embeddings.
