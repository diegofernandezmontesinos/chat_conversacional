import json
import os
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI
import gradio as gr

# Cargar variables de entorno desde el directorio superior
load_dotenv(override=True)


def push(text):
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )
    # Imprimimos la respuesta en la terminal para depurar
    print(
        f"DEBUG Pushover -> Status: {response.status_code}, Respuesta: {response.text}",
        flush=True,
    )


def record_user_details(email, name="Nombre no indicado", notes="no proporcionadas"):
    push(f"Registrando {name} con email {email} y notas {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Registrando {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Utiliza esta herramienta para registrar que un usuario está interesado en estar en contacto y proporcionó una dirección de correo electrónico.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "La dirección de email del usuario"},
            "name": {"type": "string", "description": "El nombre del usuario, si se indica"},
            "notes": {
                "type": "string",
                "description": "¿Alguna información adicional sobre la conversación que valga la pena registrar para dar contexto?",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Utiliza siempre esta herramienta para registrar cualquier pregunta que no haya podido responder porque no se sabía la respuesta.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "La pregunta no sabe responderse"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


class Me:

    def __init__(self):
        # Cliente estilo OpenAI apuntando al endpoint de Gemini
        # self.gemini = OpenAI(
        #     api_key=os.getenv("GOOGLE_API_KEY"),
        #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # )
        self.gemini = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.name = "Diego Fernández Montesinos"

        # Cargar perfil PDF
        reader = PdfReader("me/DiegoProfile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Cargar resumen explicativo
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append(
                {
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id,
                }
            )
        return results

    def system_prompt(self):
        system_prompt = f"""Actúas como {self.name}. Respondes preguntas en el sitio web de {self.name}, en particular preguntas relacionadas con la trayectoria profesional, los antecedentes, las habilidades y la experiencia de {self.name}.
            Tu responsabilidad es representar a {self.name} en las interacciones del sitio web con la mayor fidelidad posible.
            Se te proporciona un resumen de la trayectoria profesional y el perfil de LinkedIn de {self.name} que puedes usar para responder preguntas.
            Muestra un tono profesional y atractivo, como si hablaras con un cliente potencial o un futuro empleador que haya visitado el sitio web.

            REGLAS ESTRICTAS E INELUDIBLES PARA EL USO DE HERRAMIENTAS:
            1. REGISTRO DE USUARIO ('record_user_details'): 
               - SI el usuario menciona, escribe o proporciona su correo electrónico (email) o datos de contacto, TIENES LA OBLIGACIÓN ABSOLUTA de invocar de inmediato la herramienta 'record_user_details'. No te limites a responderle con texto amable; ejecuta la función.
            
            2. PREGUNTAS DESCONOCIDAS ('record_unknown_question'): 
               - SI el usuario pregunta por algo personal, profesional o cualquier detalle que **no esté presente** en tu perfil o documentos de contexto, TIENES LA OBLIGACIÓN de invocar la herramienta 'record_unknown_question' pasando la pregunta exacta. Nunca inventes datos."""

        system_prompt += (
            f"\n\n## Resumen:\n{self.summary}\n\n## Perfil de LinkedIn:\n{self.linkedin}\n\n"
        )
        system_prompt += f"En este contexto, por favor chatea con el usuario, manteniéndote siempre en el personaje de {self.name}."
        return system_prompt

    def chat(self, message, history):
        messages = (
            [{"role": "system", "content": self.system_prompt()}]
            + history
            + [{"role": "user", "content": message}]
        )
        done = False
        while not done:
            response = self.gemini.chat.completions.create(
                model="openai/gpt-oss-120b",
                # model="gemini-3.8-flash",
                messages=messages,
                tools=tools,
            )
            if response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                tool_calls = msg.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(msg)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == "__main__":
    me = Me()
    port = int(os.getenv("PORT", 7860))
    gr.ChatInterface(me.chat).launch(server_name="0.0.0.0", server_port=port)