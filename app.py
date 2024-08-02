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
cv_endpoint = st.secrets["ENDPOINT_KEY"]

openai_api_key = st.secrets["API_KEY"]

# Do some basic validation
if len(cv_key) == 32:
    st.success("Succès, la clé COMPUTER_VISION_SUBSCRIPTION_KEY est chargée.")
else:
    st.error("Erreur, la clé COMPUTER_VISION_SUBSCRIPTION_KEY n'a pas la longueur attendue, veuillez vérifier.")

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
st.title("CorrAI : Système de Correction de Copies")
st.write("Téléchargez une copie de référence et des copies d'étudiants pour une correction automatique.")

# Input section for the reference copy
st.header("Copie de Référence")
reference_image = st.file_uploader("Téléchargez la copie de référence sous forme de fichier image :", type=["jpg", "jpeg", "png"])

reference_text = ""
if reference_image:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(reference_image.getbuffer())
        reference_image_path = tmp_file.name

    reference_text = extract_text_from_image(reference_image_path)
    st.text_area("Texte Extrait de la Copie de Référence", reference_text, height=200)

# Input section for student copies
st.header("Copies des Étudiants")
uploaded_files = st.file_uploader("Téléchargez les copies des étudiants sous forme de fichiers image :", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Function to calculate the grade using ChatGPT 3.5-turbo
def grade_student_copy(reference, student, api_key):
    openai.api_key = api_key
    
    prompt = f"""
    Réponse de référence :
    {reference}

    Réponse de l'étudiant :
    {student}

    Veuillez évaluer la réponse de l'étudiant en fonction de sa précision par rapport à la réponse de référence. Fournissez une note entre 0 et 100 et un court retour.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Vous êtes un assistant utile qui évalue les réponses des étudiants en fonction d'une réponse de référence."},
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
        score = "Erreur"
        feedback = "Erreur lors de la génération de la note."

    return score, feedback

# Processing the uploaded student copies
if st.button("Évaluer les Copies des Étudiants"):
    
    if reference_text and uploaded_files and openai_api_key:
        results = []
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                student_image_path = tmp_file.name
            
            student_text = extract_text_from_image(student_image_path)
            score, feedback = grade_student_copy(reference_text, student_text, openai_api_key)
            student_name = os.path.splitext(uploaded_file.name)[0]
            results.append({"Numéro": i+1, "Nom de l'Étudiant": student_name, "Note": score, "Retour": feedback})

        # Display the results
        df_results = pd.DataFrame(results)
        st.header("Résultats")
        st.dataframe(df_results)

        # Function to convert DataFrame to CSV
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Function to convert DataFrame to Excel
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Résultats')
                writer.close()
            processed_data = output.getvalue()
            return processed_data

        # Download results as CSV
        csv = convert_df_to_csv(df_results)
        st.download_button(
            label="Télécharger les résultats en format CSV",
            data=csv,
            file_name='resultats.csv',
            mime='text/csv'
        )

        # Download results as Excel
        excel = convert_df_to_excel(df_results)
        b64 = base64.b64encode(excel).decode('utf-8')
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="resultats.xlsx">Télécharger les résultats en format Excel</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("Veuillez entrer la copie de référence, télécharger au moins une copie d'étudiant et fournir la clé API OpenAI.")

