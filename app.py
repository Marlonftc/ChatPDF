import openai
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import SelfQueryRetriever
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import langchain
from bs4 import BeautifulSoup
from datetime import datetime
import getpass

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un archivo HTML
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = Chroma.from_texts(chunks, embeddings) if chunks else None

    return knowledge_base

# Función para extraer texto de un elemento RSS
def extract_text_from_rss_item(item):
    return item.get_text()

# Función principal de la aplicación
def main():
    st.title("IA WEB HTML/XML")
    uploaded_files = st.file_uploader("Sube tus archivos HTML/XML", type=["html", "xml"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "html":
                soup = BeautifulSoup(uploaded_file, 'html.parser')
                text = soup.get_text()
            elif file_extension == "xml":
                text = uploaded_file.getvalue().decode("latin-1")  
                # Aquí puedes necesitar lógica adicional de procesamiento XML dependiendo de la estructura de tu XML

            # Crea un objeto de base de conocimientos a partir del texto
            knowledgeBase = process_text(text)

    # Caja de entrada de texto para que el usuario escriba su pregunta
    query = st.text_input('Escribe tu pregunta para el para el archivo htlm...')

    # Botón para cancelar la pregunta
    cancel_button = st.button('Cancelar')

    if cancel_button:
      st.stop()  # Detiene la ejecución de la aplicación

    if query:
      # Realiza una búsqueda de similitud en la base de conocimientos
      docs = knowledgeBase.similarity_search(query)

      # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros

      model = "gpt-3.5-turbo-instruct" # Acepta 4096 tokens
      temperature = 0  # Valores entre 0 - 1

      llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

      # Carga la cadena de preguntas y respuestas
      chain = load_qa_chain(llm, chain_type="stuff")

      # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
      with get_openai_callback() as cost:
        response = chain.invoke(input={"question": query, "input_documents": docs})
        print(cost)  # Imprime el costo de la operación

        st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()  # Llama a la función principal
