


import streamlit as st
import pytesseract
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, BertTokenizer, BertForSequenceClassification
from PIL import Image
import platform
import torch

# Set up Tesseract based on the operating system
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path
elif platform.system() == "Linux":
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux path
else:
    st.error("Unsupported OS for Tesseract. Please configure manually.")

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load ClinicalBERT model and tokenizer
clinical_bert_model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
clinical_bert_tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Function to extract text using Tesseract OCR
def extract_text_from_image(image):
    if len(image.shape) == 2:  # If grayscale
        gray_img = image
    elif len(image.shape) == 3:  # Convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format. Please provide a valid image.")
    text = pytesseract.image_to_string(gray_img)
    return text

# Function to analyze text for medical relevance
def analyze_text_and_describe(text):
    num_chars = len(text)
    num_words = len(text.split())
    description = "The text contains: "
    keywords = {
        "tumor": "tumor or cancer-related content",
    "cancer": "cancer-related details",
    "carcinoma": "cancer-related content",
    "neoplasm": "cancer-related content",
    "oncology": "cancer-related content",
    "sarcoma": "cancer-related content",
    "lymphoma": "cancer-related content",
    "melanoma": "cancer-related content",
    "leukemia": "cancer-related content",
    "breast cancer": "cancer-related content",
    "prostate cancer": "cancer-related content",
    "lung cancer": "cancer-related content",
    "colon cancer": "cancer-related content",
    "ovarian cancer": "cancer-related content",
    "pancreatic cancer": "cancer-related content",
    "bladder cancer": "cancer-related content",
    "skin cancer": "cancer-related content",
    "cervical cancer": "cancer-related content",
    "endometrial cancer": "cancer-related content",
    "esophageal cancer": "cancer-related content",
    "renal cancer": "cancer-related content",
    "head and neck cancer": "cancer-related content",
    "testicular cancer": "cancer-related content",
    "cancer metastasis": "cancer-related content",
    
    "heart": "cardiovascular-related content",
    "cardiovascular": "cardiovascular-related content",
    "coronary artery disease": "cardiovascular-related content",
    "myocardial infarction": "cardiovascular-related content",
    "heart attack": "cardiovascular-related content",
    "arrhythmia": "cardiovascular-related content",
    "hypertension": "hypertension-related content",
    "high blood pressure": "hypertension-related content",
    "stroke": "stroke-related data",
    "brain hemorrhage": "stroke-related data",
    "cerebrovascular accident": "stroke-related data",
    "aneurysm": "stroke-related data",
    "congestive heart failure": "cardiovascular-related content",
    "atrial fibrillation": "cardiovascular-related content",
    "valvular heart disease": "cardiovascular-related content",
    "cardiomyopathy": "cardiovascular-related content",
    "deep vein thrombosis": "cardiovascular-related content",
    "pulmonary embolism": "cardiovascular-related content",
    "peripheral artery disease": "cardiovascular-related content",
    "chronic venous insufficiency": "cardiovascular-related content",
    "asthma": "respiratory-related content",
    "bronchial asthma": "respiratory-related content",
    "chronic obstructive pulmonary disease": "respiratory-related content",
    "COPD": "respiratory-related content",
    "emphysema": "respiratory-related content",
    "chronic bronchitis": "respiratory-related content",
    "pneumonia": "respiratory-related content",
    "lung infection": "respiratory-related content",
    "pulmonary fibrosis": "respiratory-related content",
    "bronchitis": "respiratory-related content",
    "tuberculosis": "respiratory-related content",
    "pulmonary embolism": "respiratory-related content",
    "lung cancer": "respiratory-related content",
    "pulmonary hypertension": "respiratory-related content",
    "sleep apnea": "respiratory-related content",
    "sarcoidosis": "respiratory-related content",
    "atelectasis": "respiratory-related content",
    
    "arthritis": "musculoskeletal-related content",
    "rheumatoid arthritis": "musculoskeletal-related content",
    "osteoarthritis": "musculoskeletal-related content",
    "psoriatic arthritis": "musculoskeletal-related content",
    "gout": "musculoskeletal-related content",
    "ankylosing spondylitis": "musculoskeletal-related content",
    "lupus": "autoimmune-related content",
    "systemic lupus erythematosus": "autoimmune-related content",
    "scleroderma": "autoimmune-related content",
    "rheumatic fever": "autoimmune-related content",
    "rheumatic heart disease": "autoimmune-related content",
    "fibromyalgia": "musculoskeletal-related content",
    "myositis": "musculoskeletal-related content",
    "bursitis": "musculoskeletal-related content",
     "HbA1c": "diabetes-related content",
    "blood glucose": "diabetes-related content",
    "fasting blood sugar": "diabetes-related content",
    "postprandial blood sugar": "diabetes-related content",
    "insulin resistance": "diabetes-related content",
    "glycemic index": "diabetes-related content",
    "diabetic ketoacidosis": "diabetes-related content",
    "insulin therapy": "diabetes-related content",
    "oral hypoglycemics": "diabetes-related content",
    "blood sugar monitoring": "diabetes-related content",
    "type 1 diabetes": "diabetes-related content",
    "type 2 diabetes": "diabetes-related content",
    "gestational diabetes": "diabetes-related content",
    "diabetic neuropathy": "diabetes-related content",
    "diabetic retinopathy": "diabetes-related content",
    "diabetic nephropathy": "diabetes-related content",
    "pre-diabetes": "diabetes-related content",
    "insulin pumps": "diabetes-related content",
    "glucose tolerance test": "diabetes-related content",
    "diabetes management": "diabetes-related content",
    "diabetes diet": "diabetes-related content",
    "carbohydrate counting": "diabetes-related content",
    "metformin": "diabetes-related content",
    "sulfonylureas": "diabetes-related content",
    "GLP-1 agonists": "diabetes-related content",
    "diabetes complications": "diabetes-related content",
    "pancreatic beta cells": "diabetes-related content",
    "hyperglycemia": "diabetes-related content",
    "hypoglycemia": "diabetes-related content",
    "continuous glucose monitor": "diabetes-related content",
    "diabetes research": "diabetes-related content",
    
    "migraine": "neurological-related content",
    "chronic migraine": "neurological-related content",
    "cluster headache": "neurological-related content",
    "tension headache": "neurological-related content",
    "severe headache": "neurological-related content",
    "neuralgia": "neurological-related content",
    "epilepsy": "neurological-related content",
    "seizure": "neurological-related content",
    "stroke": "neurological-related content",
    "Parkinson's disease": "neurological-related content",
    "Alzheimer's disease": "neurological-related content",
    "dementia": "neurological-related content",
    "amyotrophic lateral sclerosis": "neurological-related content",
    "Huntington's disease": "neurological-related content",
    "multiple sclerosis": "neurological-related content",
    "brain tumor": "neurological-related content",
    "neurodegenerative disease": "neurological-related content",
    
    "depression": "mental health-related content",
    "major depressive disorder": "mental health-related content",
    "bipolar disorder": "mental health-related content",
    "mood disorder": "mental health-related content",
    "schizophrenia": "mental health-related content",
    "anxiety": "mental health-related content",
    "panic disorder": "mental health-related content",
    "obsessive-compulsive disorder": "mental health-related content",
    "post-traumatic stress disorder": "mental health-related content",
    "generalized anxiety disorder": "mental health-related content",
    "social anxiety disorder": "mental health-related content",
    "eating disorder": "mental health-related content",
    "binge eating disorder": "mental health-related content",
    "bulimia nervosa": "mental health-related content",
    "anorexia nervosa": "mental health-related content",
    
    "anemia": "hematological-related content",
    "iron deficiency anemia": "hematological-related content",
    "vitamin B12 deficiency": "hematological-related content",
    "sickle cell anemia": "hematological-related content",
    "thalassemia": "hematological-related content",
    "hemophilia": "hematological-related content",
    "leukemia": "hematological-related content",
    "lymphoma": "hematological-related content",
    "blood cancer": "hematological-related content",
    "myelodysplastic syndrome": "hematological-related content",
    "polycythemia vera": "hematological-related content",
    
    "allergy": "immunological-related content",
    "allergic reaction": "immunological-related content",
    "hay fever": "immunological-related content",
    "seasonal allergy": "immunological-related content",
    "food allergy": "immunological-related content",
    "drug allergy": "immunological-related content",
    "skin allergy": "immunological-related content",
    "latex allergy": "immunological-related content",
    
    "obesity": "metabolic-related content",
    "overweight": "metabolic-related content",
    "morbid obesity": "metabolic-related content",
    "metabolic syndrome": "metabolic-related content",
    "type 2 diabetes": "metabolic-related content",
    "insulin resistance": "metabolic-related content",
    
    "diabetes": "metabolic-related content",
    "type 1 diabetes": "metabolic-related content",
    "diabetic retinopathy": "metabolic-related content",
    "diabetic neuropathy": "metabolic-related content",
    "gestational diabetes": "metabolic-related content",
    
    "hepatitis": "liver-related content",
    "hepatitis A": "liver-related content",
    "hepatitis B": "liver-related content",
    "hepatitis C": "liver-related content",
    "cirrhosis": "liver-related content",
    "fatty liver disease": "liver-related content",
    "liver cancer": "liver-related content",
    
    "kidney disease": "renal-related content",
    "chronic kidney disease": "renal-related content",
    "nephropathy": "renal-related content",
    "dialysis": "renal-related content",
    "kidney failure": "renal-related content",
    "polycystic kidney disease": "renal-related content",
    
    "thyroid": "endocrine-related content",
    "hypothyroidism": "endocrine-related content",
    "hyperthyroidism": "endocrine-related content",
    "Hashimoto's thyroiditis": "endocrine-related content",
    "Graves' disease": "endocrine-related content",
    "goiter": "endocrine-related content",
    "thyroid cancer": "endocrine-related content",
    
    "HIV": "infectious disease-related content",
    "AIDS": "infectious disease-related content",
    "human immunodeficiency virus": "infectious disease-related content",
    "tuberculosis": "infectious disease-related content",
    "malaria": "infectious disease-related content",
    "hepatitis B": "infectious disease-related content",
    "hepatitis C": "infectious disease-related content",
    "pneumonia": "infectious disease-related content",
    "influenza": "infectious disease-related content",
    "chickenpox": "infectious disease-related content",
    "measles": "infectious disease-related content",
    "smallpox": "infectious disease-related content",
    "Zika virus": "infectious disease-related content",
    "Ebola": "infectious disease-related content",
    "dengue fever": "infectious disease-related content",
    "malaria": "infectious disease-related content",
    
    "autism": "neurodevelopmental-related content",
    "autism spectrum disorder": "neurodevelopmental-related content",
    "attention deficit hyperactivity disorder": "neurodevelopmental-related content",
    "ADHD": "neurodevelopmental-related content",
    "learning disability": "neurodevelopmental-related content",
    "Down syndrome": "genetic disorder-related content",
    "fragile X syndrome": "genetic disorder-related content",
    "Turner syndrome": "genetic disorder-related content",
    "Klinefelter syndrome": "genetic disorder-related content",
    "sickle cell anemia": "genetic disorder-related content",
    
    "sepsis": "infectious disease-related content",
    "septicemia": "infectious disease-related content",
    
    "gastroenteritis": "digestive system-related content",
    "irritable bowel syndrome": "digestive system-related content",
    "Crohn's disease": "digestive system-related content",
    "ulcerative colitis": "digestive system-related content",
    "Celiac disease": "digestive system-related content",
    "gastritis": "digestive system-related content",
    "gallstones": "digestive system-related content",
    "peptic ulcer": "digestive system-related content",
    "hepatitis": "digestive system-related content",
    
    "multiple sclerosis": "neurological-related content",
    "Parkinson's disease": "neurological-related content",
    "Alzheimer's disease": "neurological-related content",
    "epilepsy": "neurological-related content",
    "stroke": "neurological-related content",
    "dementia": "neurological-related content",
    
    "dengue": "infectious disease-related content",
    "dengue fever": "infectious disease-related content",
    "tuberculosis": "infectious disease-related content",
    "typhoid": "infectious disease-related content",
    "cholera": "infectious disease-related content",
    "malaria": "infectious disease-related content",
    "measles": "infectious disease-related content",
    
    "herpes": "infectious disease-related content",
    "herpes simplex": "infectious disease-related content",
    "herpes zoster": "infectious disease-related content",
    
    "chronic fatigue syndrome": "fatigue-related disorder",
    "fibromyalgia": "fatigue-related disorder",
    "sleep apnea": "sleep-related disorder",
    "narcolepsy": "sleep-related disorder",
    "insomnia": "sleep-related disorder",
    
    "meningitis": "infectious disease-related content",
    "encephalitis": "infectious disease-related content",
    "brain abscess": "infectious disease-related content",
    "spinal cord infection": "infectious disease-related content",
    
    "polio": "neurological disease",
    "poliomyelitis": "neurological disease",
    "Guillain-Barré syndrome": "neurological disease",
    "Prostate": ["prostate gland", "prostatic capsule", "seminal vesicles"],
    "Kidneys": ["kidney", "hydronephrosis", "corticomedullary differentiation"],
    "Liver": ["liver", "hepatic veins", "portal veins"],
    "Gall Bladder": ["gall bladder", "cholecystitis", "calculi"],
    "Spleen": ["spleen", "focal lesions", "splenic abnormalities"],
    "Pancreas": ["pancreas", "focal lesions", "pancreatic abnormalities"],
    "Pelvis": ["pelvic floor muscles", "pelvic girdle", "bladder"],
    "Aorta": ["aorta", "IVC", "inferior vena cava"]
    
    }
    medical_content_found = False
    detected_diseases = []

    for keyword, meaning in keywords.items():
        if keyword.lower() in text.lower():
            description += f"{meaning}, "
            medical_content_found = True
            detected_diseases.append(keyword)

    description = description.rstrip(", ")
    if description == "The text contains: ":
        description += "uncertain content."
    return num_chars, num_words, description, medical_content_found, detected_diseases

# Function to classify disease and severity using ClinicalBERT
def classify_disease_and_severity(text):
    inputs = clinical_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = clinical_bert_model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Modify for more advanced classes if necessary (Assuming binary classification: 0: disease, 1: severity level)
    severity_label = "Mild" if predicted_class == 0 else "Severe"
    
    # For simplicity, use keywords in the text to classify disease
    if "heart" in text.lower():
        disease_label = "Heart Disease"
    elif "cancer" in text.lower():
        disease_label = "Cancer"
    elif "HBA1C" in text.lower():
        disease_label = "diabetes"
    else:
        disease_label = "Unknown"

    return severity_label, disease_label

# Function to generate image description using BLIP
def generate_blip_description(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Links for diseases
disease_links = {
    "tumor": "https://www.cancer.gov/about-cancer/diagnosis-staging/tumors",
    "heart": "https://www.heart.org/en/health-topics/heart-attack",
    "diabetes": "https://www.diabetes.org/",
    "cancer": "https://www.cancer.org/",
}

# Streamlit app
st.title("Medical Image Analysis and Text Extraction")

# Upload an image
uploaded_image = st.file_uploader("Upload a medical report image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    img_np = np.array(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract text
    try:
        extracted_text = extract_text_from_image(img_np)
        st.subheader("Extracted Text")
        st.text(extracted_text)

        # Analyze extracted text
        num_chars, num_words, description, medical_content_found, detected_diseases = analyze_text_and_describe(extracted_text)
        st.subheader("Text Analysis")
        st.write(f"Characters: {num_chars}")
        st.write(f"Words: {num_words}")
        st.write(description)

        # Display links to disease-related websites based on detected keywords
        if medical_content_found:
            st.success("Medical data found in text.")
            for disease in detected_diseases:
                if disease in disease_links:
                    st.markdown(f"[Learn about {disease}]({disease_links[disease]})")

            # Use ClinicalBERT for further analysis
            severity, disease = classify_disease_and_severity(extracted_text)
            st.subheader("Severity and Disease Analysis")
            st.write(f"Severity: {severity}")
            st.write(f"Disease: {disease}")
            
            # Display links to disease-related websites based on classification
            if disease in disease_links:
                st.markdown(f"[Learn more about {disease}]({disease_links[disease]})")
        else:
            st.warning("No medical-related content found in the extracted text.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
