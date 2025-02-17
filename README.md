---

# SC4002 - Group Project

This repository contains the code submission for Group 3's SC4002 group project on sentiment analysis using various NLP models.

## Group Members
- Jerome Wang (U2120160F)
- Cholakov Kristiyan Kamenov (U2123543B)
- Goh Soong Wen Ryan (U2120980L)
- Denzyl David Peh (U2122190F)
- Clement Tan Kang Hao (U2122052K)
- Tiwana Teg Singh (U2122816B)

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Notebooks](#running-the-notebooks)
- [File Descriptions](#file-descriptions)

---

## Project Structure

The project files are organized as follows:

.
├── glove
│   ├── glove.6B.50d.txt
│   ├── glove.6B.300d.txt
│   ├── glove.6B.50d.word2vec.txt
│   └── glove.6B.300d.word2vec.txt
├── histories
│   ├── bigru_history.pkl
│   ├── bilstm_history.pkl
│   ├── cnn_history.pkl
│   ├── rnn_attention_pool_history.pkl
│   ├── rnn_attention_pool_trainable_embed_history.pkl
│   ├── rnn_attention_pool_trainable_embed_mit_oov_history.pkl
│   ├── rnn_avg_pool_history.pkl
│   ├── rnn_last_state_history.pkl
│   └── rnn_max_pool_history.pkl
├── main_word2vec.ipynb
├── main_glove.ipynb
├── pretrained_models.ipynb
├── requirements.txt

---

## Installation

First, ensure that all required Python libraries are installed. Run the following command in your terminal to install packages specified in requirements.txt:

pip install -r requirements.txt

---

## Running the Notebooks

This project consists of four main Jupyter notebooks. Once dependencies are installed, you can open and run each notebook in order to execute the respective code and reproduce the results.

1. main_word2vec.ipynb — Implements models with Word2Vec embeddings.
2. main_glove.ipynb — Implements models with GloVe embeddings, which generally yield better performance than Word2Vec in this project.
3. ensemble.ipynb — Contains code for training and evaluating an ensemble model based on the individual models.
4. pretrained_models.ipynb — Fine-tunes pretrained Transformer models (BERT, RoBERTa, etc.) and evaluates their performance on the dataset.

---

## File Descriptions

- `main_word2vec.ipynb`: This notebook contains all code for models that use Word2Vec embeddings.
- `main_glove.ipynb`: This notebook covers models utilizing GloVe embeddings, which generally perform better than Word2Vec-based models in this project.
- `ensemble.ipynb`: This notebook focuses on creating an ensemble model by combining predictions from various individual models.
- `pretrained_models.ipynb`: This notebook is dedicated to fine-tuning pretrained Transformer models to evaluate and compare their performance on the sentiment analysis task.

---