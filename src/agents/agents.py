import json
# Importing the local MCP Server tool for database retrieval (RAG)
from mcp.mcp_server import mcp_server
# Importing the LangChain connector for local Ollama instances
from langchain_community.llms import Ollama

class BotanicalOrchestrator:
    """
    Orchestrates the multi-agent system execution flow.
    Manages local LLM communication, language synchronization, agent handoffs, 
    and routing to local RAG tools.
    """
    def __init__(self, model_name: str = "llama3"):
        """
        Initializes the multi-agent network bounded to a local LLM.
        
        Args:
            model_name (str): The name of the local model running in Ollama.
        """
        self.llm = Ollama(model=model_name)
        print(f"LOG: Multi-agent network bound to local LLM: {model_name}")

    def _build_triage_prompt(self, user_input: str) -> str:
        """
        Builds the prompt for Agent 1 (Triage Agent).
        Its sole purpose is to extract the main keyword/symptom for the database search.
        """
        return (
            f"You are a Triage Agent specializing in plant care routing.\n"
            f"Analyze the following user complaint: '{user_input}'\n"
            f"Identify the primary plant name or symptom mentioned.\n"
            f"Respond strictly with a JSON object using this format: "
            f'{{"query": "plant name or symptom keyword"}}'
        )

    def _build_specialist_prompt(self, user_input: str, retrieved_context: str) -> str:
        """
        Builds the prompt for Agent 2 (Specialist Agent).
        Generates the final recommendation in Portuguese based on the retrieved RAG context.
        Includes strict formatting rules to prevent conversational fillers and mandates a hidden data tag
        for downstream processing.
        """
        return (
            f"You are a strict and objective Botanical Expert.\n"
            f"User Message: {user_input}\n"
            f"Context: {retrieved_context}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Answer ONLY in Portuguese.\n"
            f"2. ABSOLUTELY NO INTRODUCTIONS. Do NOT say 'Vou sugerir', 'Aqui está', or 'Baseado no contexto'. Start your very first word with 'Recomendo' or the plant's name.\n"
            f"3. You MUST identify the main plant you are recommending or diagnosing.\n"
            f"4. At the very end of your text, you MUST append a tag with the plant name exactly like this: [PLANTA: Nome da Planta]\n"
            f"If there is no specific plant, use: [PLANTA: Nenhuma]\n\n"
            f"RESPONSE (Direct to the point):"
        )

    def _build_planner_prompt(self, plant_name: str) -> str:
        """
        Builds the prompt for Agent 3 (Planner Agent).
        Triggered interactively by the user to generate a structured weekly care routine.
        """
        return (
            f"You are a Botanical Planner Agent.\n"
            f"The user needs a practical weekly care schedule for this plant: {plant_name}.\n"
            f"CRITICAL RULES:\n"
            f"1. Respond ENTIRELY in Portuguese.\n"
            f"2. Create a clean, short, bulleted list detailing the care routine (e.g., watering frequency, light exposure, when to fertilize).\n"
            f"3. DO NOT use conversational fillers like 'Aqui está o cronograma'. Start directly with the schedule.\n"
            f"4. Keep it concise. No long paragraphs."
        )

    def execute_workflow(self, user_input: str) -> str:
        """
        Executes the core linear multi-agent task delegation (Agent 1 -> MCP Tool -> Agent 2).
        
        Returns:
            str: The raw LLM response containing the diagnosis/recommendation and the hidden plant tag.
        """
        # Step 1: Agent 1 (Triage) extracts keywords
        print("\nLOG: [Agent 1: Triage Agent] Processing raw user query...")
        triage_prompt = self._build_triage_prompt(user_input)
        triage_response = self.llm.invoke(triage_prompt).strip()
        
        # Fallback mechanism in case the LLM fails to output valid JSON
        try:
            search_args = json.loads(triage_response)
        except Exception:
            search_args = {"query": user_input}

        # Step 2: System calls the MCP Server Tool on behalf of Agent 1 (RAG Retrieval)
        print(f"LOG: [MCP] Executing 'search_clinical_context' tool with args: {search_args}")
        retrieved_context = mcp_server.execute_tool("search_clinical_context", search_args)

        # Step 3: Agent 2 (Specialist) generates final recommendation based on context
        print("LOG: [Agent 2: Specialist Agent] Formulating recommendation...")
        specialist_prompt = self._build_specialist_prompt(user_input, retrieved_context)
        
        final_response = self.llm.invoke(specialist_prompt)
        return final_response

    def generate_routine(self, plant_name: str) -> str:
        """
        Executes Agent 3 (Planner) workflow dynamically upon user interaction.
        
        Args:
            plant_name (str): The specific plant extracted via regex from Agent 2's output.
            
        Returns:
            str: The LLM generated formatted care routine.
        """
        print(f"\nLOG: [Agent 3: Planner Agent] Generating care routine for {plant_name}...")
        planner_prompt = self._build_planner_prompt(plant_name)
        
        return self.llm.invoke(planner_prompt).strip()