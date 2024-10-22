######## Modulo de carga Resumidos y fragmentos ##############################
######## por cada fragmento pone la metada ########################
# --- Módulo 1: Cargar y almacenar documentos ---
#---- ponte marca de Fragmento de tal resolucion y fecha
import uuid
import os
import fitz  # PyMuPDF para manejar archivos PDF
import configparser

# Importar las librerías de LangChain necesarias
from langchain_chroma import Chroma  # Actualizado para evitar advertencias de deprecación
from langchain_openai import OpenAIEmbeddings  # Actualizado para evitar advertencias de deprecación
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Leer el archivo de configuración
config = configparser.ConfigParser()
config.read('config.ini')

# Obtener las configuraciones desde el archivo
chunk_size_padre = config['CARGA_RESOLUCIONES'].getint('chunk_size_padre', fallback=2000)
chunk_size_hijo = config['CARGA_RESOLUCIONES'].getint('chunk_size_hijo', fallback=400)
openai_api_key = config['DEFAULT'].get('OPENAI_API_KEY', fallback='')

# El directorio donde están la BDV
fragment_store_directory = config['CARGA_RESOLUCIONES'].get('FRAGMENT_STORE_DIR', fallback='')
summary_store_directory = config['CARGA_RESOLUCIONES'].get('SUMMARY_STORE_DIR', fallback='')

# Directorios y archivos para fragmentos y resúmenes
lista_fragmentos_path = config['CARGA_RESOLUCIONES'].get('LISTA_PDF_FRAGMENTOS_DIR', '')
lista_resumenes_path = config['CARGA_RESOLUCIONES'].get('LISTA_PDF_RESUMEN_DIR', '')
lista_fragmentos_file = config['CARGA_RESOLUCIONES'].get('LISTA_PDF_FRAGMENTOS_FILE', '')
lista_resumenes_file = config['CARGA_RESOLUCIONES'].get('LISTA_PDF_RESUMEN_FILE', '')
fragmento_dir = config['CARGA_RESOLUCIONES'].get('FRAGMENTO_DIR', '')
resumen_dir = config['CARGA_RESOLUCIONES'].get('RESUMEN_DIR', '')

# Función para imprimir valores de las variables
def print_variable(name, value):
    if value:
        print(f"{name}: {value}")
    else:
        print(f"{name}: No tiene valor asignado")

# Mostrar las variables de configuración leídas
print_variable("chunk_size_padre", chunk_size_padre)
print_variable("chunk_size_hijo", chunk_size_hijo)
print_variable("openai_api_key", openai_api_key)
print_variable("fragment_store_directory", fragment_store_directory)
print_variable("summary_store_directory", summary_store_directory)
print_variable("lista_fragmentos_path", lista_fragmentos_path)
print_variable("lista_resumenes_path", lista_resumenes_path)
print_variable("lista_fragmentos_file", lista_fragmentos_file)
print_variable("lista_resumenes_file", lista_resumenes_file)
print_variable("fragmento_dir", fragmento_dir)
print_variable("resumen_dir", resumen_dir)

# Configurar la API Key de OpenAI
if not openai_api_key:
    raise ValueError("Error: La API Key de OpenAI no está configurada correctamente.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Leer el archivo de lista de PDFs de fragmentos
full_fragment_file_path = os.path.join(lista_fragmentos_path, lista_fragmentos_file)
if not os.path.exists(full_fragment_file_path):
    print(f"Advertencia: No se encontró el archivo en la ruta especificada: {full_fragment_file_path}")
    print("Intentando buscar el archivo en el directorio actual...")
    full_fragment_file_path = os.path.join(os.getcwd(), lista_fragmentos_file)
    if not os.path.exists(full_fragment_file_path):
        raise FileNotFoundError(f"Error: No se pudo encontrar el archivo de lista de fragmentos: {lista_fragmentos_file}")

print(f"Leyendo archivo de lista de fragmentos: {full_fragment_file_path}")
with open(full_fragment_file_path, "r") as f:
    fragment_lines = f.readlines()

# Procesar cada línea del archivo de lista de PDFs de fragmentos
fragment_documents = []
for line in fragment_lines:
    line = line.strip()
    if not line:
        continue

    # Cada línea contiene: ruta_pdf, doc_id, fecha XA el METADATA
    if len(line.split(",")) != 3:
        print(f"Error: Formato incorrecto en la línea del archivo de fragmentos: {line}")
        continue
    fragment_file, fragment_doc_id, fragment_fecha = line.split(",")

    # Verificar si el archivo existe
    full_fragment_path = os.path.join(fragmento_dir, fragment_file)
    if not os.path.exists(full_fragment_path):
        print(f"Error: No se encontró el archivo de fragmento {fragment_file}. Se omite este archivo.")
        continue

    print(f"Cargando archivo de fragmento: {full_fragment_path}")
    # Cargar el PDF de fragmentos y convertirlo en texto
    try:
        pdf_document = fitz.open(full_fragment_path)
    except Exception as e:
        print(f"Error al abrir el archivo {fragment_file}: {e}. Se omite este archivo.")
        continue

    pdf_text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pdf_text += page.get_text()

    # Crear un documento de LangChain a partir del texto del PDF
    pdf_text = f"Este fragmento corresponde a la resolución: {fragment_doc_id} {pdf_text}"
    fragment_document = Document(page_content=pdf_text, metadata={"Fragmento de la resolución": fragment_doc_id, "Fecha de la resolución": fragment_fecha})

    # Dividir el documento en fragmentos más pequeños para el vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_hijo)
    sub_docs = text_splitter.split_documents([fragment_document])
    fragment_documents.extend(sub_docs)

# Almacenar los fragmentos en un vector store específico para fragmentos
fragment_vectorstore = Chroma(collection_name="fragment_store", embedding_function=OpenAIEmbeddings(), persist_directory=fragment_store_directory)

# Añadir fragmentos al vector store de fragmentos (indexación para búsquedas más precisas)
fragment_vectorstore.add_documents(fragment_documents)

# Leer el archivo de lista de PDFs de resúmenes
full_resumen_file_path = os.path.join(lista_resumenes_path, lista_resumenes_file)
if not os.path.exists(full_resumen_file_path):
    raise FileNotFoundError(f"Error: No se encontró el archivo de lista de resúmenes: {full_resumen_file_path}")

with open(full_resumen_file_path, "r") as f:
    resumen_lines = f.readlines()

# Procesar cada línea del archivo de lista de PDFs de resúmenes
summary_documents = []
print("Procesando archivo de resúmenes...")
for line in resumen_lines:
    line = line.strip()
    if not line:
        continue

    # Cada línea contiene: ruta_pdf, doc_id, fecha  XA METADA
    if len(line.split(",")) != 3:
        print(f"Error: Formato incorrecto en la línea del archivo de resúmenes: {line}")
        continue
    resumen_file, resumen_doc_id, resumen_fecha = line.split(",")

    # Verificar si el archivo existe
    full_resumen_path = os.path.join(resumen_dir, resumen_file)
    if not os.path.exists(full_resumen_path):
        print(f"Error: No se encontró el archivo de resumen {resumen_file}. Se omite este archivo.")
        continue

    print(f"Cargando archivo de resumen: {full_resumen_path}")
    # Cargar el archivo de resumen correspondiente y agregarlo al vector store de resúmenes
    try:
        resumen_document = fitz.open(full_resumen_path)
    except Exception as e:
        print(f"Error al abrir el archivo {resumen_file}: {e}. Se omite este archivo.")
        continue

    resumen_text = ""
    for page_num in range(len(resumen_document)):
        page = resumen_document.load_page(page_num)
        resumen_text += page.get_text()

    # Crear un documento de LangChain a partir del texto del resumen
    summary_doc = Document(page_content=resumen_text, metadata={"archivo": resumen_file, "Resolucion": resumen_doc_id, "fecha": resumen_fecha})
    summary_documents.append(summary_doc)

# Almacenar los resúmenes en un vector store específico para resúmenes
summary_vectorstore = Chroma(collection_name="summary_store", embedding_function=OpenAIEmbeddings(), persist_directory=summary_store_directory)

# Añadir el resumen al vector store de resúmenes
summary_vectorstore.add_documents(summary_documents)
