from langchain_community.llms import Ollama
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
import os
import argparse

DOC_DIRECTORY = "data" # directorio donde se encuentran los documentos
DB_DIRECTORY = "./chroma_db" # directorio donde se guarda la base de datos de vectores
MODEL="sentence-transformers/all-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
vectorstore = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)

# Actualizar todos los embeddings
def update_all_embeddings():
    loader = DirectoryLoader(DOC_DIRECTORY)
    docs = loader.load_and_split(text_splitter=text_splitter)
    print(f"\nDocumentos añadidos: {len(docs)}")
    list_all_sources()
    return

# Add a document to the embeddings. Devuelve True si se añade correctamente, False en caso contrario
def add_document_to_embeddings(file_path)->bool:
    try:
        loader = UnstructuredFileLoader(file_path)
        new_docs = loader.load_and_split(text_splitter=text_splitter)
        vectorstore.add_documents(new_docs)
    except Exception as e:
        print(f"Ha ocurrido un error al añadir el documento: {e}")
        return False
    return True

#Añade todos los ficheros de un directorio a los embeddings haciendo uso de la función add_document_to_embeddings
def add_all_documents_to_embeddings():
    count = 0
    for file_name in os.listdir(DOC_DIRECTORY):
        file_path = os.path.join(DOC_DIRECTORY, file_name)
        print(f"\nAñadiendo documento: {file_name}")
        if add_document_to_embeddings(file_path):
            count += 1
    print(f"\nDocumentos añadidos: {count}")
    return

# Añade todos los documentos de un directorio a los embeddings, haciendo uso de la función add_document_to_embeddings
# def add_all_documents_to_embeddings():
#     for file_name in os.listdir(DOC_DIRECTORY):
#         file_path = os.path.join(DOC_DIRECTORY, file_name)
#         add_document_to_embeddings(file_path)
#     return

# Delete a document from the embeddings
def delete_document_from_embeddings(file_name):
    # vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    ids = vectorstore.get(where={'source': os.path.join(DOC_DIRECTORY, file_name)})['ids']
    vectorstore.delete(ids)
    print(f"\nDocumento eliminado: {file_name}")
    return

# Definir la función para comprobar si un fichero está en los sources
def check_file_in_sources(file_name):
    try:
        docs = vectorstore.get()
        sources = {doc["source"] for doc in docs["metadatas"] if "source" in doc}
        if file_name in sources:
            print(f"El fichero '{file_name}' está en la base de datos de vectores.")
        else:
            print(f"El fichero '{file_name}' NO está en la base de datos de vectores.")
    except Exception as e:
        print(f"Ha ocurrido un error al comprobar el fichero: {e}")
    return

# Definiendo la función para listar todos los documentos en la base de datos de vectores
def list_all_sources():
    try:
        docs = vectorstore.get()
        sources = {doc["source"] for doc in docs["metadatas"] if "source" in doc}
        print("\nDocumentos en la base de datos de vectores:")
        for source in sources:
            print(f" - {source}")
    except Exception as e:
        print(f"Ha ocurrido un error al listar los documentos: {e}")

# Definir la función para eliminar la base de datos
def delete_database():
    try:
        # vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        vectorstore.delete_collection()
        print("\nBase de datos de vectores eliminada.")
    except Exception as e:
        print(f"Ha ocurrido un error al eliminar la base de datos: {e}")

# Configurar argparse para aceptar el nombre del fichero como argumento
def main():
    # vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # vectorstore.delete_collection()

    parser = argparse.ArgumentParser(description="Gestor del vector store de Chroma.")
    parser.add_argument("-r", "--remove", type=str, help="Nombre del fichero a eliminar")
    parser.add_argument("-u", "--update", action="store_true", help="Actualizar todos los embeddings")
    parser.add_argument("-a", "--add", type=str, help="Nombre del fichero a añadir a los embeddings")
    parser.add_argument("-f", "--file", type=str, help="Comprueba si el fichero está en la base de datos de vectores")
    parser.add_argument("-l", "--list", action="store_true", help="Listar todos los documentos en la base de datos de vectores")
    parser.add_argument("-x", "--delete", action="store_true", help="Eliminar la base de datos de vectores")
    # Verificar si se proporcionaron argumentos
    if len(os.sys.argv) == 1:
        parser.print_usage()
        #list_all_sources()
        os.sys.exit(1)
    args = parser.parse_args()
    
    # Llamar a la función para eliminar el documento si se proporciona el argumento -r
    if args.remove:
        delete_document_from_embeddings(args.remove)

    # Llamar a la función para actualizar todos los embeddings si se proporciona el argumento -u
    if args.update:
        add_all_documents_to_embeddings()
        
    # Llamar a la función para añadir el documento si se proporciona el argumento -a
    if args.add:
        file_path = os.path.join(DOC_DIRECTORY, args.add)
        add_document_to_embeddings(file_path)

    # Llamar a la función para comprobar si el fichero está en los sources si se proporciona el argumento -f
    if args.file:
        file_path = os.path.join(DOC_DIRECTORY, args.file)
        check_file_in_sources(file_path)

    # Llamar a la función para listar todos los documentos en la base de datos de vectores si se proporciona el argumento -l
    if args.list:
        list_all_sources()

    # Llamar a la función para eliminar la base de datos si se proporciona el argumento -x
    if args.delete:
        delete_database()

if __name__ == "__main__":
    main()