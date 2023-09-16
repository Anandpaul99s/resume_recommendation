import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Load DistilBERT tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)



st.title("Resume Recommendation App")

job_description = st.text_area("Enter job Description","")


# file path for the pickle file
pickle_file_path = 'resume_embeddings.pkl'

# load data from the pickle file
with open(pickle_file_path, 'rb') as file:
    resume_embeddings = pickle.load(file)

resumes_1 = pd.read_csv('resumes_ex.csv')

def clean_text(text):

    # remove the html tags
    text = re.sub('<.*?>', '', text)
    # Remove https
    text = re.sub('http\S+\s', ' ', text)

    # removing @gmails
    text = re.sub('@\S+', ' ', text)

    # Handle contractions (e.g., can't -> cannot)
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'s": " is",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
        "'ve": " have"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Remove the special characters and punctuation
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # convert text to lower case
    text = text.lower()

    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text

# Text preprocessing


def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Normalization (convert to lowercase)
    normalized_words = [word.lower() for word in stemmed_words]

    # Join the cleaned words back into a text
    cleaned_text = ' '.join(normalized_words)

    return cleaned_text


def text_to_embeddings(text):
    tokens = tokenizer(text, return_tensors='pt',
                       padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return embeddings


def recommend_resume(job_description, resume_embeddings, resumes_1):
    # clean and preprocess the text
    cleaned_text = clean_text(job_description)
    processed_text = preprocess_text(cleaned_text)

    # Tokenize the jd and extract word embeddings
    jd_embedding = text_to_embeddings(processed_text)
    jd_embedding_2d = jd_embedding.reshape(1, -1)
    resume_embeddings_flat = np.array(
        [emb.flatten() for emb in resume_embeddings['resume_embeddings'].values])
    # Calucate cosine similarity

    cosine_similarities = cosine_similarity(
        jd_embedding_2d, resume_embeddings_flat)

    sorted_indices = np.argsort(cosine_similarities[0])[::1]

    top_5_indices = sorted_indices[:5]
    top_5_similarities = cosine_similarities[0][top_5_indices]

    recommended_resumes = []

    for j, index in enumerate(top_5_indices):
        resume_id = resumes_1['ID'].iloc[index]
        similarity_score = top_5_similarities[j]
        original_resume = resumes_1['resume_str'].iloc[index]

        recommended_resumes.append({
            "Resume_ID": resume_id,
            "Cosine_Similarity": similarity_score,
            "Original_Resume": original_resume
        })

    return recommended_resumes








if st.button('Recommend Resumes'):
    if job_description:

        recommended_resumes = recommend_resume(
            job_description, resume_embeddings, resumes_1)
        
        # Displaying recommended resumes
        st.subheader("Recommended Resumes:")
        cnt = 0
        for resume in recommended_resumes:
            cnt = cnt+1
            st.write('Resume:',cnt)
            st.write(f"Resume ID: {resume['Resume_ID']}")
            st.write(f"Cosine Similarity: {resume['Cosine_Similarity']}")
            with st.expander(f"Original Resume for ID: {resume['Resume_ID']}"):
                st.write(resume['Original_Resume'])


    