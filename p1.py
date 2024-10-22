from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import ChatOpenAI
import os

# Configuramos la clave de la API de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Definimos el estado del gráfico, que en este caso será solo una lista de mensajes
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Creamos el gráfico de estado con el estado definido anteriormente
graph_builder = StateGraph(State)

# Definimos el modelo de lenguaje que utilizaremos (OpenAI en este caso)
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# Definimos la función del chatbot que procesará el estado y generará una respuesta
def chatbot(state: State):
    # Invocamos al modelo para generar una respuesta y actualizamos el estado con los nuevos mensajes
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Añadimos el nodo "chatbot" al gráfico
graph_builder.add_node("chatbot", chatbot)

# Definimos las transiciones del gráfico: 
# START -> chatbot -> END (el flujo comienza en START, pasa por chatbot y termina en END)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compilamos el gráfico
graph = graph_builder.compile()

# Función para ejecutar el gráfico e interactuar con el chatbot
def run_chatbot(user_input: str):
    # Ejecutamos el gráfico con el mensaje del usuario y mostramos la respuesta del chatbot
    events = graph.stream({"messages": [("user", user_input)]})
    for event in events:
        for value in event.values():
            print("Chatbot:", value["messages"][-1].content)

# Ejemplo de conversación simple
if __name__ == "__main__":
    user_input = input("Escribe algo: ")
    run_chatbot(user_input)
