# Resume Classifier

A simple machine learning project that classifies resumes into job categories such as Software Engineer, Data Scientist, Web Developer, Product Manager, Python Developer, Sales, etc.

## Features

- Upload resumes as plain text (`.txt`) or PDF (`.pdf`) files
- Extracts text from PDFs on the client side using PDF.js
- Classifies resume text into predefined job categories using a trained ML model
- Simple and responsive web interface built with HTML, CSS, and JavaScript
- Flask backend serving predictions via a REST API

## Tech Stack

- Frontend: HTML, CSS, JavaScript, PDF.js
- Backend: Python, Flask
- Machine Learning: scikit-learn (TF-IDF + Logistic Regression)
- Text processing with Python (cleaning and vectorization)

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Annie0159/ResumeClassifier.git
cd ResumeClassifier
```

### 2. Install Dependencies

pip istall flask scikit-learn pandas joblib

### 3. Prepare your dataset and train the model

python train_model.py --data_path resumes.csv

### 4. Run the Flask backend

python app.py

-----------------

