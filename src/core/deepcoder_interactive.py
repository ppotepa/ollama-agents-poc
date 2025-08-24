from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent
try:
    # LC 0.3.x
    from langchain.agents import AgentType
    AGENT_TYPE = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
except Exception:
    # fallback na nazwę stringową
    AGENT_TYPE = "structured-chat-zero-shot-react-description"

from langchain.tools import StructuredTool
import os
import traceback

# ---------- NARZĘDZIA (multi-input) ----------

def write_file(filepath: str, contents: str) -> str:
    """Zapisz (utwórz/nadpisz) plik na dysku."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(contents)
    return f"✅ Zapisano do {filepath} ({len(contents)} bajtów)"

def read_file(filepath: str) -> str:
    """Odczytaj zawartość pliku."""
    if not os.path.exists(filepath):
        return f"❌ Plik {filepath} nie istnieje"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

tools = [
    StructuredTool.from_function(
        func=write_file,
        name="write_file",
        description=(
            "Zapisz/utwórz plik. Argumenty: "
            "filepath (ścieżka, np. './src/main.py'), "
            "contents (pełna treść pliku)."
        ),
    ),
    StructuredTool.from_function(
        func=read_file,
        name="read_file",
        description=(
            "Odczytaj plik. Argumenty: "
            "filepath (ścieżka, np. './src/main.py')."
        ),
    ),
]

# ---------- MODEL ----------
llm = ChatOllama(
    model="deepcoder:14b",
    temperature=0.2,
    num_ctx=8192,
)

# ---------- AGENT (Structured Chat) ----------
SYSTEM_MSG = (
    "Jesteś asystentem-koderem z narzędziami plikowymi. "
    "Gdy użytkownik prosi o stworzenie/edycję/odczyt pliku, "
    "UŻYJ odpowiednio write_file lub read_file, przekazując wszystkie argumenty. "
    "Nie pytaj o format – sam zdecyduj i działaj."
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AGENT_TYPE,              # <— obsługa multi-input tools
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"system_message": SYSTEM_MSG},
)

# ---------- REPL ----------
if __name__ == "__main__":
    print("🤖 DeepCoder REPL (Structured Chat). Wpisz 'exit' aby wyjść.\n")
    while True:
        try:
            cmd = input("📝> ").strip()
            if cmd.lower() in ("exit", "quit"):
                print("👋 Do zobaczenia!")
                break

            # W LC 0.3.x unikaj .run(); używaj .invoke({"input": ...})
            result = agent.invoke({"input": cmd})
            output = result.get("output", result)
            print(f"\n💡 Odpowiedź:\n{output}\n{'-'*60}\n")

        except KeyboardInterrupt:
            print("\n👋 Przerwano.")
            break
        except Exception as e:
            print("❌ Błąd:", e)
            traceback.print_exc()
