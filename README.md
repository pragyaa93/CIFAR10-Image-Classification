# CIFAR-10 Image Classification Using CNN

## Project Overview
This project implements an image classification system
using a Convolutional Neural Network (CNN) and the CIFAR-10
dataset.

The application allows users to upload an image and
predict its category using a trained CNN model.

## Dataset
CIFAR-10 contains 60,000 colour images across 10 classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Technologies Used
- Python
- PyTorch
- CNN
- Streamlit
- Matplotlib
- Scikit-learn

## Features
- Image classification
- Top 3 predictions
- Prediction confidence
- Confusion matrix
- Classification report
- Accuracy visualization

## Run the Application

Install dependencies:

pip install -r requirements.txt

Start Streamlit:

streamlit run app.py

## Project Structure

- app.py: Streamlit application
- cnn.pth: Trained CNN model weights
- mlproject.ipynb: Model development and evaluation
- requirements.txt: Project dependencies
