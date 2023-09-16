import streamlit as st
import os
import csv
import PyPDF2
from PyPDF2 import PdfReader
import re
import io

skills_pattern = r"Skills(.*?)(Education|$)"
education_pattern = r"Education(.*?)(Skills|$)"

# 1.Function to extract text from the pdf using pypdf2


def extract_text_from_pdf(uploaded_file):
    try:
        # Convert the UploadedFile object to bytes
        pdf_bytes = uploaded_file.read()

        # Use PyPDF2 to extract text from the bytes
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(pdf_reader.pages)
        results = []
        for i in range(num_pages):
            page = pdf_reader.pages[i]
            text = page.extract_text()
            results.append(text)
        s = ' '.join(results)
        return s
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return ""

# 2.Function to extract skills and education from text extracted from pdf using re


def extract_skills_and_education(text):
    skills_match = re.search(skills_pattern, text, re.DOTALL)
    education_match = re.search(education_pattern, text, re.DOTALL)

    skills = skills_match.group(1).strip() if skills_match else ""
    education = education_match.group(1).strip() if education_match else ""

    return skills, education




# Streamlit UI
st.title("PDF Text, Skills, and Education Extractor")
st.sidebar.header("Upload PDF File")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.sidebar.success("File uploaded successfully!")

    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(uploaded_file)

    # Extract skills and education
    extracted_skills, extracted_education = extract_skills_and_education(
        pdf_text)

    # Display the extracted text, skills, and education
    st.header("Extracted Text:")
    st.write(pdf_text)

    st.header("Extracted Skills:")
    st.write(extracted_skills)

    st.header("Extracted Education:")
    st.write(extracted_education)
