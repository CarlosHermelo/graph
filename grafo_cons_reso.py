from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os

# Definimos la clase de estado
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Mensajes intercambiados
    question_type: Optional[str]  # Tipo de pregunta ('general' o 'fragmento')
    retrieved_data: Optional[str]  # Información recuperada
    date_from: Optional[str]  # Fecha desde (hardcodeado inicialmente)
    date_to: Optional[str]  # Fecha hasta (hardcodeado inicialmente)

# Creamos el gráfico de estado con la clase de estado definida
graph_builder = StateGraph(State)

# Nodo 1: Evaluar el tipo de pregunta (con resultado hardcodeado)
def evaluate_question(state: State):
    # Para pruebas iniciales, siempre consideramos que la pregunta es 'general'
    state["question_type"] = "particular"
    return state

graph_builder.add_node("evaluate_question", evaluate_question)

# Nodo 2: Recuperar información general (con resultado hardcodeado)
def retrieve_general_info(state: State):
    # Hardcodeamos la respuesta de recuperación de datos
    state["retrieved_data"] = "Información general recuperada."
    return state

graph_builder.add_node("retrieve_general_info", retrieve_general_info)

# Nodo 3: Recuperar información específica (con resultado hardcodeado)
def retrieve_specific_info(state: State):
    # Hardcodeamos la respuesta de recuperación específica
    state["retrieved_data"] = "Información específica recuperada."
    return state

graph_builder.add_node("retrieve_specific_info", retrieve_specific_info)

# Nodo 4: Generar la respuesta final (con resultado hardcodeado)
def generate_response(state: State):
    # Generamos una respuesta final con la información recuperada
    response_message = f"Rrrrrrespuesta final basada en: {state['retrieved_data']} (Fechas: desde {state['date_from']} hasta {state['date_to']})"
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
    # Mostramos la pregunta y las fechas hardcodeadas
    print(f"Pregunta: {user_input}")
    print(f"Fecha desde: {date_from}")
    print(f"Fecha hasta: {date_to}")
    
    # Ejecutamos el gráfico con el mensaje del usuario y mostramos la respuesta del chatbot
    initial_state = {
        "messages": [("user", user_input)],
        "date_from": date_from,
        "date_to": date_to,
    }
    events = graph.stream(initial_state)
    for event in events:
        for value in event.values():
            last_message = value["messages"][-1]
            if isinstance(last_message, tuple) and last_message[0] == "assistant":
                print("Chatbot:", last_message[1])

# Ejemplo de conversación simple
if __name__ == "__main__":
    # Hardcodeamos la pregunta y las fechas para probar el flujo completo
    pregunta = "¿ cuales son las normativas reslacionadas a los subsidios energeticos de luz  y gas?"
    fecha_desde = "2023-01-01"
    fecha_hasta = "2023-12-31"
    run_chatbot(pregunta, fecha_desde, fecha_hasta)
