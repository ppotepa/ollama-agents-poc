"""InterceptorAgent - specialized agent for prompt analysis and command recommendation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.base.base_agent import AbstractAgent
from src.agents.descriptors import AgentDescriptor, Capabilities
from src.core.enums import AgentCapability, Domain
from src.core.prompt_interceptor import InterceptionMode
from src.utils.animations import stream_text

try:  # LangChain optional
    from langchain_ollama import ChatOllama
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain ChatOllama imported successfully for InterceptorAgent")
except Exception as e:
    LANGCHAIN_AVAILABLE = False
    print(f"⚠️  LangChain ChatOllama import failed for InterceptorAgent: {e}")
    # Define dummy class to avoid import errors
    class ChatOllama:
        pass


@dataclass
class CommandRecommendation:
    """Represents a recommended command for a given prompt."""
    command: str
    confidence: float
    description: str
    category: str
    required_context: list[str]


@AgentDescriptor(
    name="Prompt Interceptor",
    backend_image="phi3:mini",
    description="Specialized agent for analyzing prompts and recommending appropriate commands",
    domain=Domain.GENERAL
)
@Capabilities(AgentCapability.STREAMING, AgentCapability.FUNCTION_CALLS)
class InterceptorAgent(AbstractAgent):
    """Agent specialized for analyzing prompts and recommending appropriate commands."""

    def __init__(self, agent_id: str, config: dict[str, Any]):
        super().__init__(agent_id, config)
        self._command_patterns = self._initialize_command_patterns()
        self._context_requirements = self._initialize_context_requirements()
        # Load the LLM and tools
        self.load()

    @property
    def llm(self):
        """Access to the built LLM."""
        return self._llm

    def analyze_prompt_with_llm(self, prompt: str) -> str:
        """Analyze prompt using the phi3:mini LLM directly."""
        if not self._llm:
            return "LLM not available"

        try:
            llm_prompt = f"""Analyze this user prompt and suggest appropriate repository/coding commands:

Prompt: "{prompt}"

Available commands:
- analyze_repo_structure: Analyze repository file structure
- analyze_repo_languages: Identify programming languages used
- analyze_repo_directories: Analyze directory structure
- read_file: Read file contents
- write_file: Create or modify files
- list_files: List files in directories
- duck_search: Search the web
- fetch_url: Fetch content from URLs
- run_python: Execute Python code

Respond with a JSON object containing command recommendations:
{{
    "analysis": "Brief analysis of the prompt",
    "recommendations": [
        {{"command": "command_name", "confidence": 0.8, "reasoning": "why this command helps"}}
    ]
}}"""

            response = self._llm.invoke(llm_prompt)
            return response.content

        except Exception as e:
            return f"LLM analysis failed: {e}"

    def _build_llm(self) -> Any:
        """Build LLM for the interceptor agent using phi3:mini."""
        if not LANGCHAIN_AVAILABLE:
            print("⚠️  LangChain not available for phi3:mini LLM")
            return None

        try:
            # Create ChatOllama instance with phi3:mini model
            llm = ChatOllama(
                model="phi3:mini",
                temperature=0.1,
                num_ctx=2048,
                num_predict=512,
                base_url="http://127.0.0.1:11434"
            )

            return llm

        except Exception as e:
            print(f"⚠️  Could not build phi3:mini LLM: {e}")
            return None

    def _build_tools(self) -> list[Any]:
        """Build tools for the interceptor agent (not used for analysis)."""
        # InterceptorAgent doesn't use external tools, but needs to implement abstract method
        return []

    def _initialize_command_patterns(self) -> dict[str, dict]:
        """Initialize patterns for command recognition."""
        return {
            "repository_analysis": {
                "patterns": [
                    "analyze", "structure", "overview", "organization", "architecture",
                    "files", "directories", "project", "codebase", "repository"
                ],
                "commands": ["analyze_repo_structure", "analyze_repo_languages", "analyze_repo_directories"],
                "confidence_base": 0.8
            },
            "file_operations": {
                "patterns": [
                    "read", "write", "create", "delete", "copy", "move", "list",
                    "file", "directory", "folder", "content", "edit"
                ],
                "commands": ["read_file", "write_file", "list_files", "delete_file", "copy_file"],
                "confidence_base": 0.9
            },
            "code_analysis": {
                "patterns": [
                    "function", "class", "method", "variable", "import", "syntax",
                    "error", "debug", "review", "quality", "refactor"
                ],
                "commands": ["read_file", "analyze_repo_structure", "run_python"],
                "confidence_base": 0.7
            },
            "documentation": {
                "patterns": [
                    "document", "readme", "docs", "comment", "explain", "description",
                    "guide", "tutorial", "help", "usage"
                ],
                "commands": ["generate_documentation", "read_file", "write_file"],
                "confidence_base": 0.6
            },
            "search_and_fetch": {
                "patterns": [
                    "search", "find", "look", "fetch", "download", "get", "retrieve",
                    "url", "web", "internet", "duck", "google"
                ],
                "commands": ["duck_search", "fetch_url"],
                "confidence_base": 0.8
            },
            "execution": {
                "patterns": [
                    "run", "execute", "test", "compile", "build", "python", "script"
                ],
                "commands": ["run_python"],
                "confidence_base": 0.9
            }
        }

    def _initialize_context_requirements(self) -> dict[str, list[str]]:
        """Initialize context requirements for different command types."""
        return {
            "analyze_repo_structure": ["repository_path", "working_directory"],
            "analyze_repo_languages": ["repository_path"],
            "analyze_repo_directories": ["repository_path"],
            "read_file": ["file_path"],
            "write_file": ["file_path", "content"],
            "list_files": ["directory_path"],
            "delete_file": ["file_path"],
            "copy_file": ["source_path", "destination_path"],
            "generate_documentation": ["target_files"],
            "duck_search": ["search_query"],
            "fetch_url": ["url"],
            "run_python": ["python_code"]
        }

    def analyze_prompt(self, prompt: str, mode: InterceptionMode = InterceptionMode.SMART) -> list[CommandRecommendation]:
        """Analyze a prompt and return recommended commands."""
        # First try using the phi3:mini LLM for intelligent analysis
        llm_recommendations = self._analyze_prompt_with_llm(prompt)

        if llm_recommendations:
            return llm_recommendations

        # Fallback to pattern-based analysis
        return self._analyze_prompt_with_patterns(prompt)

    def _analyze_prompt_with_llm(self, prompt: str) -> list[CommandRecommendation]:
        """Use phi3:mini LLM to analyze the prompt and recommend commands."""
        try:
            llm = self._build_llm()
            if not llm:
                return []

            # Create a structured prompt for phi3:mini
            analysis_prompt = f"""You are a command recommendation assistant. Analyze this user prompt and recommend appropriate commands.

User prompt: "{prompt}"

Available commands:
- analyze_repo_structure: Analyze repository file structure and organization
- analyze_repo_languages: Identify programming languages used in repository
- analyze_repo_directories: Analyze directory structure and contents
- read_file: Read and display file contents
- write_file: Create or modify file with new content
- list_files: List files and directories in a location
- delete_file: Remove a file from the filesystem
- copy_file: Copy a file to a new location
- generate_documentation: Generate documentation for code files
- duck_search: Search the web using DuckDuckGo
- fetch_url: Fetch content from a web URL
- run_python: Execute Python code

Respond with JSON in this exact format:
{{"recommendations": [
    {{"command": "command_name", "confidence": 0.8, "reasoning": "why this command is useful"}}
]}}

Provide 1-3 most relevant commands with confidence scores between 0.1 and 1.0."""

            # Get response from phi3:mini
            response = llm.invoke(analysis_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response
            try:
                import json
                # Extract JSON from response (handle cases where there might be extra text)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    data = json.loads(json_text)

                    recommendations = []
                    for rec in data.get('recommendations', []):
                        command = rec.get('command', '')
                        confidence = float(rec.get('confidence', 0.5))
                        rec.get('reasoning', '')

                        if command in self._get_available_commands():
                            recommendations.append(CommandRecommendation(
                                command=command,
                                confidence=confidence,
                                description=self._get_command_description(command),
                                category=self._get_command_category(command),
                                required_context=self._context_requirements.get(command, [])
                            ))

                    if recommendations:
                        print(f"✅ phi3:mini analyzed prompt and recommended {len(recommendations)} commands")
                        return recommendations[:5]  # Top 5

            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️  Could not parse phi3:mini response as JSON: {e}")

        except Exception as e:
            print(f"⚠️  phi3:mini analysis failed: {e}")

        return []

    def _analyze_prompt_with_patterns(self, prompt: str) -> list[CommandRecommendation]:
        """Fallback pattern-based analysis."""
        prompt_lower = prompt.lower()
        recommendations = []

        for category, config in self._command_patterns.items():
            confidence = self._calculate_pattern_confidence(prompt_lower, config["patterns"])

            if confidence > 0.1:  # Lower threshold for more recommendations
                for command in config["commands"]:
                    adjusted_confidence = confidence * config["confidence_base"]
                    context_reqs = self._context_requirements.get(command, [])

                    recommendations.append(CommandRecommendation(
                        command=command,
                        confidence=adjusted_confidence,
                        description=self._get_command_description(command),
                        category=category,
                        required_context=context_reqs
                    ))

        # Sort by confidence and return top recommendations
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:5]  # Top 5 recommendations

    def _get_available_commands(self) -> list[str]:
        """Get list of all available commands."""
        return [
            "analyze_repo_structure", "analyze_repo_languages", "analyze_repo_directories",
            "read_file", "write_file", "list_files", "delete_file", "copy_file",
            "generate_documentation", "duck_search", "fetch_url", "run_python"
        ]

    def _get_command_category(self, command: str) -> str:
        """Get category for a command."""
        for category, config in self._command_patterns.items():
            if command in config["commands"]:
                return category
        return "general"

    def _calculate_pattern_confidence(self, prompt: str, patterns: list[str]) -> float:
        """Calculate confidence score based on pattern matching."""
        matches = sum(1 for pattern in patterns if pattern in prompt)
        if matches == 0:
            return 0.0
        # Give higher weight to any matches found
        base_confidence = min(matches / len(patterns), 1.0)
        # Boost confidence if we have any matches at all
        return max(base_confidence, 0.2 if matches > 0 else 0.0)

    def _get_command_description(self, command: str) -> str:
        """Get human-readable description for a command."""
        descriptions = {
            "analyze_repo_structure": "Analyze repository file structure and organization",
            "analyze_repo_languages": "Identify programming languages used in repository",
            "analyze_repo_directories": "Analyze directory structure and contents",
            "read_file": "Read and display file contents",
            "write_file": "Create or modify file with new content",
            "list_files": "List files and directories in a location",
            "delete_file": "Remove a file from the filesystem",
            "copy_file": "Copy a file to a new location",
            "generate_documentation": "Generate documentation for code files",
            "duck_search": "Search the web using DuckDuckGo",
            "fetch_url": "Fetch content from a web URL",
            "run_python": "Execute Python code"
        }
        return descriptions.get(command, f"Execute {command}")

    def generate_lightweight_response(self, prompt: str) -> str:
        """Generate a fast, lightweight response for prompt analysis."""
        recommendations = self.analyze_prompt(prompt, InterceptionMode.LIGHTWEIGHT)

        if not recommendations:
            return "No specific commands identified. This appears to be a general query."

        response_parts = [
            "🔍 **Prompt Analysis Results**\n",
            "**Top Command Recommendations:**\n"
        ]

        for i, rec in enumerate(recommendations[:3], 1):
            response_parts.append(
                f"{i}. **{rec.command}** (confidence: {rec.confidence:.1%})\n"
                f"   ↳ {rec.description}\n"
                f"   ↳ Category: {rec.category}\n"
            )

        if recommendations[0].required_context:
            response_parts.append(f"\n**Required Context:** {', '.join(recommendations[0].required_context)}")

        return "".join(response_parts)

    def run_interactive(self, initial_prompt: str = None) -> None:
        """Run the agent in interactive mode for prompt analysis."""
        print(f"🤖 {self.config.get('name', 'Interceptor Agent')} ready!")
        print("I specialize in analyzing prompts and recommending appropriate commands.")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                if initial_prompt:
                    user_input = initial_prompt
                    initial_prompt = None  # Only use once
                else:
                    user_input = input("💭 Enter prompt to analyze: ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Generate response
                response = self.generate_lightweight_response(user_input)

                # Stream the response
                stream_text(response)
                print("\n" + "="*50 + "\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def process_query(self, query: str, **kwargs) -> str:
        """Process a query and return command recommendations."""
        mode = kwargs.get('mode', InterceptionMode.SMART)

        if mode == InterceptionMode.LIGHTWEIGHT:
            return self.generate_lightweight_response(query)
        else:
            # For full mode, provide more detailed analysis
            recommendations = self.analyze_prompt(query, mode)

            response_parts = [
                "🔍 **Detailed Prompt Analysis**\n",
                f"**Query:** {query}\n\n",
                "**Command Recommendations:**\n"
            ]

            for i, rec in enumerate(recommendations, 1):
                response_parts.append(
                    f"{i}. **{rec.command}** (confidence: {rec.confidence:.1%})\n"
                    f"   ↳ Description: {rec.description}\n"
                    f"   ↳ Category: {rec.category}\n"
                    f"   ↳ Required Context: {', '.join(rec.required_context) if rec.required_context else 'None'}\n\n"
                )

            if not recommendations:
                response_parts.append("No specific commands identified. This appears to be a general query that may require manual analysis.")

            return "".join(response_parts)
