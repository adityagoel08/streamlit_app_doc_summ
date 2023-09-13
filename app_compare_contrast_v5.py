from __future__ import print_function
import os
import json
import re
import streamlit as st
import PyPDF2
import openai
import pandas as pd
import numpy as np
from num2words import num2words
from PIL import Image
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

pd.set_option('display.max_colwidth', None)

# Loading Image using PIL
im = Image.open('hsbc.png')
# Adding Image to web app
st.set_page_config(page_title="PDF Comparison App", page_icon=im, layout="wide")

# Display a static banner image
banner_image = "banner.jpg"  # Replace with the path to your banner image
st.image(banner_image, use_column_width=True)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

openai.api_type = 'azure'
openai.api_base = 'https://openai-south-central-us-innovation.openai.azure.com/'
openai.api_version = '2023-05-15'

model_name = "gpt-4"

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def clean_text(text, token_limit=None):
    # Remove special characters and keep only alphanumeric, whitespace, full stops, and commas
    cleaned_text = re.sub(r'[^a-zA-Z\s.,]', '', text)
    # Remove newline and tab characters
    cleaned_text = re.sub(r'[\n\t]', ' ', cleaned_text)
    tokens = cleaned_text.split()
    if token_limit:
        tokens = tokens[:token_limit]
    return ' '.join(tokens)

def count_tokens(text):
    tokens = word_tokenize(text)
    count = len(tokens)
    return count

# Edit chunk size and overlap as required
def break_up_file_to_chunks(text, chunk_size=2100, overlap=100):

    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    
    chunks = []
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

def extract_information(option, bullet_points = None, cleaned_text1 = None, cleaned_text2 = None, specific_terms=None, user_prompt = None, model_name = 'gpt-4'):
    # Set up the prompt for GPT-4 based on the selected option
    if option == 1:
        prompt = f"Extract information from the document and summarize in {bullet_points} bullet points:\n{cleaned_text1}\n"
    elif option == 2:
        prompt = f"Extract information from the document and summarize in {bullet_points} bullet points:\n{cleaned_text2}\n"
    elif option == 3:
        prompt = f"Extract information from both the documents and summarize in {bullet_points} bullet points each, separately:\nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
    elif option == 4:
        #prompt = f"Summarize both the following documents in {bullet_points} bullet points each, separately, without losing out key information. In addition, also retain information related to the following keywords: {specific_terms} . Mention the keywords in brackets. \nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
        prompt = f"Extract and generate a general summary for both the following documents provided, in {bullet_points} bullet points each, separately, without losing out key information. The summaries should capture the main points, ideas, and key takeaways from each document. Additionally, please extract points from the documents that are related to specific keywords provided: {specific_terms}. For each keyword provided, extract and summmarize sentences or phrases that discuss, define, or elaborate on the keyword within the context of the document. Please format these extracted points as bullet points as well. \nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
    elif option == 5:
        prompt = f"I need to compare and contrast 2 financial documents in {bullet_points} bullet points. Extract SPECIFIC key point of difference and key similarities between both documents, and quote the words from the documents as is. Mention page number of document if possible. Bullets should be numbered. Extract the following information as:\nKey similarities between both documents:\nKey differences between both documents:\nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
    elif option == 6:
        prompt = f"I need to compare and contrast 2 financial documents specific to {specific_terms}. Extract key point of differences and similarities between both documents only for {specific_terms}, and quote the words from the documents as is. Mention page number of document if possible. Bullets should be numbered. Extract the following information as first of {specific_terms}: Similarities - \nDifferences - \n, then second of {specific_terms}:\n and so on, for:.\nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
    elif option == 7:
        prompt = f"Extract sentences and numbers from following documents without any loss of information related to: \n - Escalation of matters \n - Items being notified to committee or senior leadership \n - Key issues and concerns \n - Reported breaches. Please arrange the extracted information in bullet points while maintaining the original context. \nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
    elif option == 8:
        prompt = f"Identify and highlight any contradictory statements or information within the given document. Focus on inconsistencies, opposing viewpoints, or conflicting data that might indicate contradictions. Also locate and underline instances of self-contradiction within the document. Highlight situations where statement A is presented initially, but diverges to statement B or conflicting information later in the text. Arrange the output in bullet points. Document:\n{cleaned_text1}\n"
    elif option == 9:
        prompt = f"Identify and highlight any contradictory statements or information within the given document. Focus on inconsistencies, opposing viewpoints, or conflicting data that might indicate contradictions. Also locate and underline instances of self-contradiction within the document. Highlight situations where statement A is presented initially, but diverges to statement B or conflicting information later in the text. Arrange the output in bullet points. Document:\n{cleaned_text2}\n"
    elif option == 10:
        prompt = f"Identify and highlight any contradictory statements or information within the given documents. Focus on inconsistencies, opposing viewpoints, or conflicting data that might indicate contradictions. Also locate and underline instances of self-contradiction within the document. Highlight situations where statement A is presented initially, but diverges to statement B or conflicting information later in the text. Arrange the output in bullet points. \nDocument 1:\n{cleaned_text1}\nDocument 2:\n{cleaned_text2}\n"
    elif option == 11:
        prompt = f"{user_prompt}. Document:\n{cleaned_text1}\n"
    elif option == 12:
        prompt = f"{user_prompt}. Document:\n{cleaned_text2}\n"
    
    

    # Generate response using GPT-4
    response = openai.ChatCompletion.create(
        engine=model_name,
        temperature=0.2,
        messages=[
            {'role': 'system', 'content': "You are a financial analyst with a very crisp and concise way of presentation, and are very accurate with numbers."},
            {'role': 'user', 'content': prompt}
        ]
    )

    # Extract information from GPT-4 response
    information = response.choices[0]["message"]["content"].strip()

    # Return the output
    return information

# Streamlit code
st.title("PDF Summarizer and Comparison Accelerator")
st.write("This accelerator helps in summarization as well as compare-and-contrast between 2 PDF documents.")

left, right = st.columns(2)

with left:
    # User input
    st.header("User Input")
    
    azure_openai_api_key = st.text_input("Enter your Azure OpenAI API Key:")
    openai.api_key = azure_openai_api_key
    os.environ['OPENAI_API_KEY'] = azure_openai_api_key
    
    #'911e57abc4ac48b9b3cd1837ee51d881'
    
    pdf1_file = st.file_uploader("Upload PDF 1", type="pdf")
    if pdf1_file is not None:
        pdf1_text = extract_text_from_pdf(pdf1_file)
        cleaned_text1 = clean_text(pdf1_text)
        tokens1 = count_tokens(cleaned_text1)
        st.write("PDF1 Tokens:", tokens1)
        
    pdf2_file = st.file_uploader("Upload PDF 2", type="pdf")
    if pdf2_file is not None:
        pdf2_text = extract_text_from_pdf(pdf2_file)
        cleaned_text2 = clean_text(pdf2_text)
        tokens2 = count_tokens(cleaned_text2)
        st.write("PDF2 Tokens:", tokens2)
    
    if pdf1_file is not None and pdf2_file is not None:
        st.write("")
    else:
        st.write("<p style='font-weight: bold; color: #DB0011;'>Upload both files to see all options.</p>", unsafe_allow_html=True)

    #chunk_size = st.number_input("Select chunk size:", min_value=100, max_value=5000, value=2100, step=100)
    #TOKEN LIMIT
    #if pdf1_file is not None and pdf2_file is not None:
    #    token_limit = st.slider("Select token limit (0-16000)", min_value=0, max_value=16000, value=1000, step=100)
    #else:
    #    token_limit = st.slider("Select token limit (0-32000)", min_value=0, max_value=32000, value=1000, step=100)
    
    if pdf1_file is None and pdf2_file is None:
        token_limit = None
    elif pdf2_file is None:
        token_limit = tokens1
    elif pdf1_file is None:
        token_limit = tokens2
    else:
        token_limit = tokens1 + tokens2
        
    
    model_options = ["gpt-35-turbo", "gpt-4", "gpt-4-32k"]
    model_options_1 = ["gpt-4", "gpt-4-32k"]
    model_options_2 = ["gpt-4-32k"]
    #selected_model = st.radio("Select a model:", model_options,horizontal=True)
    
    if token_limit == None:
        st.write("<p style='font-weight: bold; color: #000000;'>Upload one or more files </p>", unsafe_allow_html=True)
    elif token_limit <= 3500:
        selected_model = st.radio("Select a model:", model_options, horizontal=True)
    elif token_limit <=7000:
        selected_model = st.radio("Select a model:", model_options_1,horizontal=True)
    elif token_limit <=25000:
        selected_model = st.radio("Select a model:", model_options_2,horizontal=True)
    elif token_limit>25000:
        st.write("<p style='color: #DB0011;'>Token Limit exceeds LLM limit. Please select token limit to filter for first N tokens of the document(s).</p>", unsafe_allow_html=True)
        if pdf1_file is not None and pdf2_file is not None:
            token_limit = st.slider("Select token limit (0-12500)", min_value=0, max_value=12500, value=1000, step=100)
        else:
            token_limit = st.slider("Select token limit (0-25000)", min_value=0, max_value=25000, value=1000, step=100)
            
        if token_limit <= 3500:
            selected_model = st.radio("Select a model:", model_options, horizontal=True)
        elif token_limit <=7000:
            selected_model = st.radio("Select a model:", model_options_1,horizontal=True)
        elif token_limit <=25000:
            selected_model = st.radio("Select a model:", model_options_2,horizontal=True)
    
        
    if pdf1_file is not None and pdf2_file is not None:
        prompt_option = st.selectbox("Select an option:", ("Summarization", "Compare n Contrast", "High Priority Keywords", "Contradiction", "User Prompts"))
        if prompt_option == "Summarization":
            summarization_option = st.radio("Select a Summarization option:", ("Summarize PDF1", "Summarize PDF2", "Summarize both PDFs", "Summarize both PDFs - with priority keywords"))
            st.write("You selected:", summarization_option)
            f_option = summarization_option

        elif prompt_option == "Compare n Contrast":
            compare_option = st.radio("Select a Compare n Contrast option:", ("Compare - Overall", "Compare - On specific terms"))
            st.write("You selected:", compare_option)
            f_option = compare_option

        elif prompt_option == "High Priority Keywords":
            keyword_option = st.radio("Select a High Priority Keywords option:", ("Escalations and Concerns"))
            st.write("You selected:", keyword_option)
            f_option = keyword_option

        elif prompt_option == "Contradiction":
            contradiction_option = st.radio("Select a Contradiction option:", ("Find contradictions in PDF1", "Find contradictions in PDF2", "Find contradictions in both PDFs"))
            st.write("You selected:", contradiction_option)
            f_option = contradiction_option

        elif prompt_option == "User Prompts":
            user_input_option = st.radio("Select a User Input option:", ("Prompt PDF1", "Prompt PDF2"))
            st.write("You selected:", user_input_option)
            f_option = user_input_option
        
    elif pdf2_file is None:
        prompt_option = st.selectbox("Select an option:", ("Summarization", "High Priority Keywords", "Contradiction", "User Prompts"))
        if prompt_option == "Summarization":
            summarization_option = st.radio("Select a Summarization option:", (["Summarize PDF1"]))
            st.write("You selected:", summarization_option)
            f_option = summarization_option

        elif prompt_option == "High Priority Keywords":
            keyword_option = st.radio("Select a High Priority Keywords option:", (["Escalations and Concerns"]))
            st.write("You selected:", keyword_option)
            f_option = keyword_option

        elif prompt_option == "Contradiction":
            contradiction_option = st.radio("Select a Contradiction option:", (["Find contradictions in PDF1"]))
            st.write("You selected:", contradiction_option)
            f_option = contradiction_option

        elif prompt_option == "User Prompts":
            user_input_option = st.radio("Select a User Input option:", (["Prompt PDF1"]))
            st.write("You selected:", user_input_option)
            f_option = user_input_option
        
        
    elif pdf1_file is None:
        prompt_option = st.selectbox("Select an option:", ("Summarization", "High Priority Keywords", "Contradiction", "User Prompts"))
        if prompt_option == "Summarization":
            summarization_option = st.radio("Select a Summarization option:", (["Summarize PDF2"]))
            st.write("You selected:", summarization_option)
            f_option = summarization_option

        elif prompt_option == "High Priority Keywords":
            keyword_option = st.radio("Select a High Priority Keywords option:", (["Escalations and Concerns"]))
            st.write("You selected:", keyword_option)
            f_option = keyword_option

        elif prompt_option == "Contradiction":
            contradiction_option = st.radio("Select a Contradiction option:", (["Find contradictions in PDF2"]))
            st.write("You selected:", contradiction_option)
            f_option = contradiction_option

        elif prompt_option == "User Prompts":
            user_input_option = st.radio("Select a User Input option:", (["Prompt PDF2"]))
            st.write("You selected:", user_input_option)
            f_option = user_input_option
        
    else:
        st.warning("Please upload one or more PDF files.")
    
    # Assign numbers to the options selected
    if f_option == "Summarize PDF1":
        option = 1
    elif f_option == "Summarize PDF2":
        option = 2
    elif f_option == "Summarize both PDFs":
        option = 3
    elif f_option == "Summarize both PDFs - with priority keywords":
        option = 4    
    elif f_option == "Compare - Overall":
        option = 5
    elif f_option == "Compare - On specific terms":
        option = 6
    elif f_option == "Escalations and Concerns":
        option = 7
    elif f_option == "Find contradictions in PDF1":
        option = 8
    elif f_option == "Find contradictions in PDF2":
        option = 9
    elif f_option == "Find contradictions in both PDFs":
        option = 10
    elif f_option == "Prompt PDF1":
        option = 11
    elif f_option == "Prompt PDF2":
        option = 12
    else:
        option = None
    
    if option <=5:
        bullet_points = st.number_input("Enter the number of bullet points:", min_value=1, max_value=20, value=10, step=1)
    
    if option == 4 or option ==6:
        specific_terms = st.text_input("Enter specific terms (comma-separated):")
    if option == 11 or option == 12:
        user_prompt_f = st.text_input("Enter your prompt:")

with right:
    if st.button("Process PDFs"):
        if pdf1_file is not None and pdf2_file is not None:
            #pdf1_text = extract_text_from_pdf(pdf1_file)
            #pdf2_text = extract_text_from_pdf(pdf2_file)

            cleaned_text1 = clean_text(pdf1_text, token_limit=token_limit)
            cleaned_text2 = clean_text(pdf2_text, token_limit=token_limit)

            if option == 4 or option ==6:
                specific_terms_list = [term.strip() for term in specific_terms.split(',')]
                specific_terms_str = ', '.join(specific_terms_list)
                if option==4:
                    information = extract_information(option = option, cleaned_text1 = cleaned_text1, cleaned_text2 = cleaned_text2, bullet_points = bullet_points, specific_terms=specific_terms_str, model_name = selected_model)
                else:
                    information = extract_information(option = option, cleaned_text1 = cleaned_text1, cleaned_text2 = cleaned_text2, specific_terms=specific_terms_str, model_name = selected_model)
            elif option == 11 or option == 12:
                information = extract_information(option = option, cleaned_text1 = cleaned_text1, cleaned_text2 = cleaned_text2, user_prompt=user_prompt_f, model_name = selected_model)
                
            elif option >5 and option <11 and option:
                information = extract_information(option = option, cleaned_text1 = cleaned_text1, cleaned_text2 = cleaned_text2, model_name = selected_model)
                
            else:
                information = extract_information(option = option, cleaned_text1 = cleaned_text1, cleaned_text2 = cleaned_text2, bullet_points = bullet_points, model_name = selected_model)
                
            st.header("Output")
            #st.write(" ".join(information))
            st.text_area("Results:", information)
            
        elif pdf2_file is None:
            cleaned_text1 = clean_text(pdf1_text, token_limit=token_limit)
            if option == 11 or option == 12:
                information = extract_information(option = option, cleaned_text1 = cleaned_text1, user_prompt=user_prompt_f, model_name = selected_model)
            elif option >5 and option <11:
                information = extract_information(option = option, cleaned_text1 = cleaned_text1, model_name = selected_model)    
            else:
                information = extract_information(option = option, cleaned_text1 = cleaned_text1, bullet_points = bullet_points, model_name = selected_model)

            st.header("Output")
            st.text_area("Results:", information)
            
        elif pdf1_file is None:
            cleaned_text2 = clean_text(pdf2_text, token_limit=token_limit)
            if option == 11 or option == 12:
                information = extract_information(option = option, cleaned_text2 = cleaned_text2, user_prompt=user_prompt_f, model_name = selected_model)
            elif option >5 and option <11:
                information = extract_information(option = option, cleaned_text2 = cleaned_text2, model_name = selected_model)     
            else:
                information = extract_information(option = option, cleaned_text2 = cleaned_text2, bullet_points = bullet_points, model_name = selected_model)

            st.header("Output")
            st.text_area("Results:", information)   
            
        else:
            st.warning("Please upload one or more PDF files.")
        
        output_tokens = count_tokens(information)
        st.write("Output Tokens:", output_tokens)
