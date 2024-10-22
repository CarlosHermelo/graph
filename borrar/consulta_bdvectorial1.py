#######################BUSQUEDA POR SIMILITUD########################
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

# Validar que la clave API está presente
if not openai_api_key:
    raise ValueError("API key de OpenAI no está configurada.")

# Establecer la clave de OpenAI API directamente
os.environ["OPENAI_API_KEY"] = openai_api_key

# Definir las funciones de embebido de OpenAI para los vectores
embedding_function = OpenAIEmbeddings()

# Cargar las bases de datos vectoriales de fragmentos y resúmenes
def load_vectorstore(collection_name, persist_directory):
    try:
        return Chroma(collection_name=collection_name, embedding_function=embedding_function, persist_directory=persist_directory)
    except Exception as e:
        print(f"Error al cargar vector store {collection_name}: {e}")
        return None

fragment_vectorstore = load_vectorstore("fragment_store", fragment_store_directory)
summary_vectorstore = load_vectorstore("summary_store", summary_store_directory)

# Realizar una búsqueda de similitud en el vector store de fragmentos
def search_vectorstore(vectorstore, query, max_results=5):
    try:
        return vectorstore.similarity_search(query, k=max_results)
    except Exception as e:
        print(f"Error al realizar búsqueda de similitud: {e}")
        return []

query = "¿Plan pañales?"
print(f'Consulta realizada: {query}')
fragment_results = search_vectorstore(fragment_vectorstore, query)

# Procesar y mostrar los resultados
def process_results(results, label):
    if results:
        print('-' * 50)
        print(f"Resultado del vector store ({label}):")
        for i, result in enumerate(results, start=1):
            print(f"Resultado {i}:")
            print(result.page_content)
            print("Metadatos asociados:")
            for key, value in result.metadata.items():
                print(f"{key}: {value}")
            print('-' * 50)
        return results
    else:
        print('-' * 50)
        print(f"\nNo se encontraron {label} relevantes para la consulta.")
        return None

# Mostrar los resultados de fragmentos
fragment_result = process_results(fragment_results, "fragmentos")

# Buscar en el vector store de resúmenes si se encontró un fragmento relevante
if fragment_result:
    print(f'Consulta realizada: {query}')
    summary_results = search_vectorstore(summary_vectorstore, query)
    process_results(summary_results, "resúmenes")

# Mostrar los metadatos si el fragmento fue encontrado
if fragment_result:
    print('-' * 50)
    print("\nMetadatos del fragmento encontrado:")
    for key, value in fragment_result[0].metadata.items():
        print(f"{key}: {value}")
