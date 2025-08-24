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

# ---------- PUBLIC FUNCTIONS FOR MAIN.PY ----------
def run_single_query(query: str, agent_name: str = "deepcoder") -> str:
    """Run a single query against the DeepCoder agent."""
    try:
        result = agent.invoke({"input": query})
        return result.get("output", str(result))
    except Exception as e:
        return f"❌ Error: {e}"

def run_interactive_session(agent_name: str = "deepcoder"):
    """Run an interactive session with the DeepCoder agent."""
    print("🤖 DeepCoder Interactive Session. Type 'exit' to quit.\n")
    
    # Check if this agent supports coding and prompt for repository if needed
    try:
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from core.helpers import check_agent_supports_coding, validate_repository_requirement
        
        agent_supports_coding = check_agent_supports_coding(agent_name)
        
        if agent_supports_coding:
            print(f"🔍 Agent '{agent_name}' is a coding agent and works best with a repository.")
            
            while True:
                repo_url = input("📂 Please enter a git repository URL (or 'skip' to continue without): ").strip()
                
                if repo_url.lower() == 'skip':
                    print("⚠️  Continuing without repository - some features may be limited.\n")
                    break
                elif repo_url:
                    try:
                        # Use current working directory as base
                        original_cwd = os.getcwd()
                        data_path = os.path.join(original_cwd, "data")
                        
                        print(f"🔄 Setting up repository: {repo_url}")
                        validation_passed, working_dir = validate_repository_requirement(agent_name, ".", repo_url, data_path)
                        
                        if working_dir != ".":
                            print(f"✓ Repository cloned and validated: {working_dir}")
                            os.chdir(working_dir)
                            print(f"✓ Changed working directory to: {working_dir}")
                        else:
                            print(f"✓ Repository validation passed")
                        break
                        
                    except Exception as e:
                        print(f"❌ Error setting up repository: {e}")
                        continue
                else:
                    print("Please enter a valid git URL or 'skip'")
    
    except Exception as e:
        print(f"⚠️  Warning: Could not check coding capability: {e}")
    
    # Show available models
   
    print(f"\n💡 Using model: deepcoder:14b")
    print("💡 Use Ctrl+C to exit\n")
    
    while True:
        try:
            cmd = input("📝> ").strip()
            if cmd.lower() in ("exit", "quit"):
                print("👋 Session ended!")
                break

            # Process the command
            result = agent.invoke({"input": cmd})
            output = result.get("output", result)
            print(f"\n💡 Response:\n{output}\n{'-'*60}\n")

        except KeyboardInterrupt:
            print("\n👋 Session interrupted.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            if hasattr(e, '__traceback__'):
                traceback.print_exc()

# ---------- REPL (for direct execution) ----------
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
