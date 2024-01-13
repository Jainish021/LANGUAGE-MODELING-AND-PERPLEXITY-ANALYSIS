# Language Modeling and Perplexity Analysis

## Overview

This project focuses on creating unigram and bigram language models to analyze text data. Key aspects include:

- Preprocessing text data
- Implementing unigram and bigram models using HashMaps
- Calculating token probabilities
- Applying smoothing techniques (Laplace and Add-k)
- Handling unknown/rare words
- Computing perplexity to evaluate models
- Comparing performance of unigram vs bigram models

## Models

- **Unigram**: Calculated probability of individual tokens
- **Bigram**: Calculated probability of token pairs

## Results

- Bigram model exhibited lower perplexity compared to the unigram model
- Increasing the smoothing factor alpha initially reduced perplexity
- A higher threshold k unexpectedly led to lower perplexity

## Installation

The code was implemented in Python, and no additional libraries are required.