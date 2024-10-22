# script2.py
import os
import configparser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Leer el archivo de configuración
config = configparser.ConfigParser()
config.read('config.ini')

# Obtener las configuraciones desde el archivo
openai_api_key = config['DEFAULT'].get('OPENAI_API_KEY', fallback='')

fragment_store_directory = config['CARGA_RESOLUCIONES'].get('FRAGMENT_STORE_DIR', fallback='/content/chroma_fragment_store')
summary_store_directory = config['CARGA_RESOLUCIONES'].get('SUMMARY_STORE_DIR', fallback='/content/chroma_summary_store')
# Modificar la función para aceptar un parámetro adicional "tipo"
max_results_config = config['CARGA_RESOLUCIONES'].getint('max_results', fallback=5)

if not openai_api_key:
    raise ValueError("API key de OpenAI no está configurada.")

os.environ["OPENAI_API_KEY"] = openai_api_key
embedding_function = OpenAIEmbeddings()

def load_vectorstore(collection_name, persist_directory):
    try:
        return Chroma(collection_name=collection_name, embedding_function=embedding_function, persist_directory=persist_directory)
    except Exception as e:
        print(f"Error al cargar vector store {collection_name}: {e}")
        return None

fragment_vectorstore = load_vectorstore("fragment_store", fragment_store_directory)
summary_vectorstore = load_vectorstore("summary_store", summary_store_directory)

def search_vectorstore(vectorstore, query, max_results=5):
    if vectorstore is None:
        print("Vector store no disponible.")
        return []
    try:
        return vectorstore.similarity_search(query, k=max_results)
    except Exception as e:
        print(f"Error al realizar búsqueda de similitud: {e}")
        return []

def process_results(results, label):
    if results:
        processed_results = []
        for i, result in enumerate(results, start=1):
            processed_results.append({
                "content": result.page_content,
                "metadata": result.metadata
            })
        return processed_results
    else:
        return []


def buscar_similitud(query, tipo, max_results=max_results_config):
    if tipo == "fragmento":
        print("-----------------##### Busqueda simimiliutd  Por fragmento ######-----------------------")
        # Búsqueda en fragmentos (funciona como antes)
        print(f'Consulta realizada (tipo: {tipo}): {query}')
        fragment_results = search_vectorstore(fragment_vectorstore, query, max_results=max_results)
        
        # Procesar los resultados de fragmentos
        fragment_result = process_results(fragment_results, "fragmentos")
        
        # Si se encontraron fragmentos, retornar los resultados
        return fragment_result

    elif tipo == "resumen":
        print("-----------------##### Busqueda simimiliutd  Por resumen ######-----------------------")
        # Búsqueda en resúmenes (nueva lógica implementada)
        print(f'Consulta realizada (tipo: {tipo}): {query}')
        summary_results = search_vectorstore(summary_vectorstore, query, max_results=max_results)

        # Procesar los resultados de resúmenes
        summary_result = process_results(summary_results, "resúmenes")

        # Si se encontraron resúmenes, retornar los resultados
        return summary_result

    else:
        raise ValueError("Tipo no válido, debe ser 'fragmento' o 'resumen'")
