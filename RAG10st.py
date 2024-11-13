__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
import os
import subprocess
import shlex
import streamlit as st


SYSTEM_PROMPT = """You are Talos, a highly trained artificial intelligence assistant in the field of medium voltage electrical engineering.
Your goal is to provide clear, precise and useful answers to users' and clients' questions and concerns. 
You are able to explain technical concepts in a simple and understandable way, and you always stay up to date with the latest trends and advances in the field of medium voltage controllers. 
Your tone is professional, friendly and respectful."""


@st.fragment()
def ollama_install():
    command="systemctl is-enabled ollama"
    command=shlex.split(command)
    stat = subprocess.run(command)
    print(stat.returncode)
    if(stat.returncode !=0):  # if 0 (active), print "Active"
        # curl -fsSL https://ollama.com/install.sh | sh
        command="curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz"
        command=shlex.split(command)
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print("ollama downloaded")

        command="sudo tar -C /usr/local -xzf ollama-linux-amd64.tgz"
        command=shlex.split(command)
        process=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(process.stdout)
        print("ollama Installed")


        # command="export PATH=$PATH:/usr/local/bin"
        # command=shlex.split(command)
        # process=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # print(process.stdout)
        # print("ollama Installed")

        

        #ollama pull llama3.1:8b
        command="/usr/local/ollama pull llama3.1:8b"
        command=shlex.split(command)
        process=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print(process.stdout)
        print("llama3.1:8b downloaded")

        command="/usr/local/ollama serve"
        command=shlex.split(command)
        process=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        #print(process.stdout)
        print("ollama serve")

        
    else:
        print("ollama already active")
ollama_install()


llm = Ollama(model="llama3.1:8b", system=SYSTEM_PROMPT, temperature=0, top_k=1, top_p=1)

# Model definition
model="sentence-transformers/all-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=model)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

template_1 = """
Answer the question based only on the following context:
{context} 
The answer has to specify the source document used for the answer.
If you can not answer the question based on the context, answer the question based on your own knowledge but beginning the sentence with "The relevant information is not available in the context, but based on my own knowledge".
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template_1)
# qa_chain = LLMChain(llm=llm, prompt=prompt)
qa_chain = prompt | llm | StrOutputParser()

messages = [("system", "You are Jose.")]

st.title("Virtual Expert Assistant for ADVC")
st.write("Welcome to the virtual assistant dedicated to answering your questions about Shneider Electric medium voltage controllers.")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes de chat del historial al recargar la app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Reaccionar a la entrada del usuario
if my_prompt := st.chat_input("Escribe tu mensaje..."):
    # Mostrar mensaje del usuario en el contenedor de mensajes del chat
    st.chat_message("user").markdown(my_prompt)
    # Agregar mensaje del usuario al historial del chat
    st.session_state.messages.append({"role": "user", "content": my_prompt})
    messages.append(["human", my_prompt])
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    resultados_similares = vectorstore.similarity_search(my_prompt, k=10) # Probar con k=10
    print(len(resultados_similares))
    contexto=""
    for doc in resultados_similares:
        contexto += doc.page_content
    respuesta = qa_chain.invoke({"question": my_prompt, "context": contexto})
    resultado = respuesta
    # Mostrar respuesta del asistente en el contenedor de mensajes del chat
    with st.chat_message("assistant"):
        st.markdown(resultado)
    # Agregar respuesta del asistente al historial de chat
    st.session_state.messages.append({"role": "assistant", "content": resultado})

print("Finish")