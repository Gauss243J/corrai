import os
import streamlit as st
import pandas as pd
import time
import tempfile
import openai
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import base64

# Install the required Azure library
# os.system("pip install --upgrade azure-cognitiveservices-vision-computervision")
# os.system("pip install openai")

# Set up environment variables for Azure Cognitive Services
#  st.write('Enter your secret computer vision key:')
# Hide the default Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Enter your secret computer vision key
cv_key = st.secrets["CV_KEY"]

# Change the cv_endpoint below to your endpoint.
cv_endpoint =st.secrets["ENDPOINT_KEY"]

openai_api_key = st.secrets["API_KEY"]

# Do some basic validation
if len(cv_key) == 32:
    st.success("Success, COMPUTER_VISION_SUBSCRIPTION_KEY is loaded.")
else:
    st.error("Error, The COMPUTER_VISION_SUBSCRIPTION_KEY is not the expected length, please check it.")

# Authenticate with Azure Cognitive Services
computervision_client = None
if cv_key and cv_endpoint:
    computervision_client = ComputerVisionClient(cv_endpoint, CognitiveServicesCredentials(cv_key))

# Function to extract text from an image using Azure Cognitive Services
def extract_text_from_image(image_path):
    with open(image_path, "rb") as image_stream:
        read_response = computervision_client.read_in_stream(image_stream, raw=True)
    
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(1)
    
    extracted_text = ""
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                extracted_text += line.text + "\n"
    
    return extracted_text

# Set up the Streamlit app layout
st.title("Student Copy Correction System")
st.write("Upload a reference copy and student copies for automatic grading.")

# Input section for the reference copy
st.header("Reference Copy")
reference_image = st.file_uploader("Upload the reference copy as an image file:", type=["jpg", "jpeg", "png"])

reference_text = ""
if reference_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(reference_image.getbuffer())
        reference_image_path = tmp_file.name

    reference_text = extract_text_from_image(reference_image_path)
    st.text_area("Extracted Reference Copy Text", reference_text, height=200)

# Input section for student copies
st.header("Student Copies")
uploaded_files = st.file_uploader("Upload student copies as image files:", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Function to calculate the grade using ChatGPT 3.5-turbo
def grade_student_copy(reference, student, api_key):
    openai.api_key = api_key
    
    prompt = f"""
    Reference answer:
    {reference}

    Student answer:
    {student}

    Please grade the student answer based on its precision compared to the reference answer. Provide a score between 0 and 100 and a short feedback.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that grades student answers based on a reference answer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.1,
        top_p=1
    )

    try:
        feedback = response['choices'][0]['message']['content'].strip()
        score_line = feedback.split('\n')[0]
        score = int(''.join(filter(str.isdigit, score_line)))
    except (KeyError, IndexError, ValueError):
        score = "Error"
        feedback = "Error in generating the score."

    return score, feedback

# Processing the uploaded student copies
if st.button("Grade Student Copies"):
    
    if reference_text and uploaded_files and openai_api_key:
        results = []
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                student_image_path = tmp_file.name
            
            student_text = extract_text_from_image(student_image_path)
            score, feedback = grade_student_copy(reference_text, student_text, openai_api_key)
            student_name = os.path.splitext(uploaded_file.name)[0]
            results.append({"Number": i+1, "Student Name": student_name, "Score": score, "Feedback": feedback})

        # Display the results
        df_results = pd.DataFrame(results)
        st.header("Results")
        st.dataframe(df_results)

        # Function to convert DataFrame to CSV
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Function to convert DataFrame to Excel
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
                writer.close()
            processed_data = output.getvalue()
            return processed_data

        # Download results as CSV
        csv = convert_df_to_csv(df_results)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='results.csv',
            mime='text/csv'
        )

        # Download results as Excel
        excel = convert_df_to_excel(df_results)
        b64 = base64.b64encode(excel).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="results.xlsx">Download results as Excel</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Please enter the reference copy, upload at least one student copy, and provide the OpenAI API key.")
