import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)  
    return text

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Category', 'Resume'])  
    df['Resume'] = df['Resume'].apply(clean_text)
    return df['Resume'].tolist(), df['Category'].tolist()
