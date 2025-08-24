# PricePoke  
*A PyTorch-based Classifier for Predicting Pokémon Card Price Surges*

---

## Overview
**PricePoke** is a machine learning project that leverages historical Pokémon TCG card sales data to train a neural network classifier.  
The goal: given a card’s metadata and past price history, predict whether its price will **increase by at least 40% over the next 6 months**.

This project combines:
- **Data preprocessing & cleaning** (merging card metadata with sales/price history).  
- **Feature engineering** (variants, rarity, dates, sales aggregates).  
- **Modeling** with a PyTorch classification neural network (logistic regression baseline → deeper networks later).  
- **Evaluation** using metrics relevant for imbalanced classification (precision, recall, F1-score, ROC-AUC).  

---

## Project Structure
```text
pricepoke/
├── data/
│   ├── raw/                 # Raw input data (pokemon_data.csv, pokemon_prices.csv)
│   ├── processed/           # Cleaned and merged datasets
├── src/
│   ├── preprocess/
│   │   └── cleaning.py      # Data cleaning & merge pipeline
│   ├── models/
│   │   └── classifier.py    # PyTorch model definitions (logistic regression, MLP, etc.)
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Evaluation & metrics
├── notebooks/               # Jupyter notebooks for exploration / experiments
├── README.md                # This file
└── requirements.txt         # Python dependencies