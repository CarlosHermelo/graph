import configparser
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
import json
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage

# Cargar la configuración desde config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Leer las variables API Key y modelo desde el archivo de configuración
api_key = config['DEFAULT'].get('openai_api_key')
model_name = config['DEFAULT'].get('modelo')

# Definimos la clase de estado
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Mensajes intercambiados
    question_type: Optional[str]  # Tipo de pregunta ('general' o 'fragmento')
    improved_question: Optional[str]  # Pregunta mejorada
    retrieved_data: Optional[str]  # Información recuperada
    date_from: Optional[str]  # Fecha desde (hardcodeado inicialmente)
    date_to: Optional[str]  # Fecha hasta (hardcodeado inicialmente)

# Creamos el gráfico de estado con la clase de estado definida
graph_builder = StateGraph(State)

# Nodo 1: Evaluar el tipo de pregunta y mejorarla
llm = ChatOpenAI(
    api_key=api_key,
    model_name=model_name,
    temperature=0.7
)
memory = ConversationBufferMemory(k=5, return_messages=True)

# Definimos la función evaluate_question
evaluate_question_prompt = PromptTemplate(
    input_variables=["user_message"],
    template="""
    Tarea:
    Asume el rol de un abogado experto en redacción de resoluciones dentro de una obra social. Tu objetivo es ayudar a otro abogado o empleado de la organización a evaluar y mejorar una pregunta que desea formular, así como a generar subpreguntas que contextualicen mejor la búsqueda de información en una base de datos vectorial.

    El proceso debe seguir los siguientes pasos:

    Evaluar la pregunta: Determina si la pregunta es de carácter general o temática (específica). Considera si la pregunta puede responderse con una visión global (general) o si requiere detalles específicos que pueden encontrarse en fragmentos de varios documentos (temática).

    Mejorar la pregunta: Reescribe la pregunta original para hacerla más clara, completa y bien redactada, como lo haría un abogado con experiencia en la redacción de resoluciones para una organización de obra social. Mantén el sentido original de la pregunta, pero asegúrate de que sea más precisa.

    System: Eres un asistente experto que debe categorizar y mejorar la pregunta del usuario. Devuelve el resultado en el siguiente formato JSON: {{"question_type": "tipo_de_pregunta", "improved_question": "pregunta_mejorada"}}.

    User: {user_message}
    """
)

def evaluate_question(state: State):
    try:
        # Obtenemos el último mensaje del usuario
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage):
            user_message = last_message.content
        else:
            user_message = last_message["content"]

        # Renderizamos el prompt con las variables de entrada
        prompt = evaluate_question_prompt.format(user_message=user_message)

        # Ejecutamos el modelo LLM con el prompt ya renderizado
        output = llm.invoke(prompt)
        
        # Imprimimos la salida para depuración
        ##print(f"Salida del LLM: {output}")

        # Procesamos la respuesta
        if isinstance(output, AIMessage):
            content = output.content
        else:
            content = output

        try:
            response = json.loads(content)
            state["question_type"] = response.get("question_type", "general")
            state["improved_question"] = response.get("improved_question", user_message)
        except json.JSONDecodeError:
            print("La salida del LLM no es un JSON válido. Usando la salida directamente.")
            state["question_type"] = "general"
            state["improved_question"] = content
    except Exception as e:
        print(f"Error inesperado en evaluate_question: {e}")
        state["question_type"] = "general"
        state["improved_question"] = user_message
    
    print(f"Tipo de pregunta: {state['question_type']}")
    print(f"Pregunta mejorada: {state['improved_question']}")
    return state

graph_builder.add_node("evaluate_question", evaluate_question)

# Nodo 2: Recuperar información general (con resultado hardcodeado)
def retrieve_general_info(state: State):
    # Usar la pregunta mejorada para recuperar información (por ahora hardcodeado)
    state["retrieved_data"] = f"Información recuperada basada en: {state['improved_question']}"
    return state

graph_builder.add_node("retrieve_general_info", retrieve_general_info)

# Nodo 3: Recuperar información específica (con resultado hardcodeado)
def retrieve_specific_info(state: State):
    # Usar la pregunta mejorada para recuperar información específica (por ahora hardcodeado)

    #aca quiero ingresar LA FUNCION que tomar state["improved_question"] 
    ######--------------------------------------------------------
    query = state["improved_question"]
    # Usar la pregunta mejorada para recuperar información específica
    from funcion_RETRIEVE import buscar_similitud  # Importamos la función del script2
# Pasar el valor de state["improved_question"] a script2

# Definir el tipo que será "fragmento" en este caso
    tipo = "fragmento"

# Invocar la función buscar_similitud y pasar el nuevo parámetro "tipo"
   # query = "¿Hay algun plan integral oncologico??"
    resultados = buscar_similitud(query, tipo)

# Procesar el resultado para agregarlo a state["retrieved_data"]
    state["retrieved_data"] = resultados

    return state

    ###----------------------------------------------------------

    
graph_builder.add_node("retrieve_specific_info", retrieve_specific_info)

# Nodo 4: Generar la respuesta final (con resultado hardcodeado)
def generate_response(state: State):
    # Generamos una respuesta final con la información recuperada
    response_message = f"Respuesta final basada en: {state['retrieved_data']} (Fechas: desde {state['date_from']} hasta {state['date_to']})"
    state["messages"].append(("assistant", response_message))
    return state

graph_builder.add_node("generate_response", generate_response)

# Definir las transiciones del gráfico
graph_builder.add_edge(START, "evaluate_question")

# Definir la transición condicional después de evaluar la pregunta
graph_builder.add_conditional_edges(
    "evaluate_question",
    lambda state: "retrieve_general_info" if state["question_type"] == "general" else "retrieve_specific_info",
    {"retrieve_general_info": "retrieve_general_info", "retrieve_specific_info": "retrieve_specific_info"}
)

# Definir las transiciones hacia el nodo de respuesta final
graph_builder.add_edge("retrieve_general_info", "generate_response")
graph_builder.add_edge("retrieve_specific_info", "generate_response")

graph_builder.add_edge("generate_response", END)

# Compilamos el gráfico
graph = graph_builder.compile()

# Función para ejecutar el gráfico e interactuar con el chatbot
def run_chatbot(user_input: str, date_from: str, date_to: str):
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "date_from": date_from,
        "date_to": date_to,
    }
    # Mostramos la pregunta y las fechas hardcodeadas
    print(f"Pregunta: {user_input}")
    print(f"Fecha desde: {date_from}")
    print(f"Fecha hasta: {date_to}")
    
    # Ejecutamos el gráfico con el mensaje del usuario y mostramos la respuesta del chatbot
    events = graph.stream(initial_state)
    for event in events:
        for value in event.values():
            last_message = value["messages"][-1]
            if isinstance(last_message, tuple) and last_message[0] == "assistant":
                print("Chatbot:", last_message[1])

# Ejemplo de conversación simple
if __name__ == "__main__":
    # Hardcodeamos la pregunta y las fechas para probar el flujo completo
    pregunta = " ¿Hay algun plan integral oncologico??"
    fecha_desde = config['DEFAULT'].get('fecha_desde', "2023-01-01")
    fecha_hasta = config['DEFAULT'].get('fecha_hasta', "2023-12-31")
    run_chatbot(pregunta, fecha_desde, fecha_hasta)
