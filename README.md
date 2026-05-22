# Emotion Classification using BERT

Emotion Classification project built using **BERT (bert-base-uncased)** and **Hugging Face Transformers** for multi-class emotion prediction.

## Overview
The model was fine-tuned on the Hugging Face Emotion dataset to classify text into six emotion categories.

## Dataset
Source: Hugging Face Emotion Dataset
Classes:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Google Colab

## Workflow
Dataset Loading  
→ Tokenization  
→ BERT Fine-tuning  
→ Model Training  
→ Evaluation  
→ Model Saving

## Example Prediction
Input:
I am happy today
Predicted Emotion:
Joy

## Features
✔ BERT-based emotion classification  
✔ Hugging Face Trainer API  
✔ Tokenization and preprocessing  
✔ Model evaluation  
✔ Saved model for inference

## Project Structure
Emotion-Classification-BERT/
│── emotion_classification_bert.ipynb
│── README.md
│── requirements.txt

## Future Improvements
- Streamlit / Gradio interface
- Web deployment
- Real-time emotion prediction API
