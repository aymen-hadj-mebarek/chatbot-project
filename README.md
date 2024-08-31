# Chatbot Project

## Overview

This chatbot project demonstrates a simple yet effective conversational agent built using a neural network. The chatbot is designed to understand and respond to user queries based on predefined patterns and intents.

## Features

- **Natural Language Processing (NLP)**: Utilizes tokenization and stemming to process and understand user inputs.
- **Neural Network**: Built with Keras, featuring dense layers to classify intents and generate responses.
- **Dynamic Response**: Generates responses based on user inputs and predefined intent-response pairs.

## How It Works

1. **Data Preparation**:
    - Uses intents from a JSON file, which includes various patterns and corresponding tags.
    - Preprocesses the data by tokenizing, stemming, and converting it into a bag-of-words representation.

1. **Model Training**:
    - Constructs a neural network with Keras, consisting of input, hidden, and output layers.
    - Trains the model using categorical cross-entropy loss and the Adam optimizer.
    - Saves the model weights for future use.

1. **Prediction**:    
    - The chatbot processes user inputs, converts them to the bag-of-words format, and predicts the intent.
    - Responses are selected based on the predicted intent and are displayed to the user.

## Installation

To set up this project on your local machine:

1. **Clone the Repository**:

```bash
git clone https://github.com/yourusername/chatbot-project.git
```
    
2. **Navigate to the Project Directory**:
    
```bash
cd chatbot-project    
```
    
3. **Install Dependencies**:
    
```bash
pip install -r requirements.txt
```
    
5. **Run the Chatbot**:
    
    **Run the full code from the jupyter note book directly, or export it as python to run**
    
## Experience and Future Improvements

Working on this chatbot project has been a rewarding experience. I’ve learned valuable insights into natural language processing, neural network design, and model training with Keras.

**Future Enhancements**:

- **Augmenting the Dataset**: Expanding the dataset of `responses.json` with more diverse patterns and intents to improve the chatbot's understanding and response accuracy.
- **Advanced NLP Techniques**: Incorporating more advanced NLP techniques and pre-trained language models to enhance conversational capabilities.

## Acknowledgements

- **Keras**: For simplifying the process of building and training neural networks.
- **NLTK**: For providing tools for natural language processing.