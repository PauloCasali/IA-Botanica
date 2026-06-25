import os
import sys
import urllib.parse
import re

# Dynamic path resolution to allow smooth module resolution inside the src/ tree
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importing the multi-agent orchestrator package
from agents.agents import BotanicalOrchestrator

# Importing visual UI components from the Rich library for terminal rendering
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.status import Status

# Initializing global console for standard Rich outputs
console = Console()

def print_header():
    """
    Renders a stylized text-only header for the terminal application UI.
    Provides academic and architectural metadata to the viewer.
    """
    header_text = Text("SISTEMA DE SUPORTE À BOTÂNICA CLÍNICA", style="bold green", justify="center")
    header_text.append("\n[Arquitetura Multi-Agentes & RAG Local]", style="dim white")
    
    console.print(
        Panel(
            header_text, 
            border_style="bright_blue", 
            expand=False, 
            padding=(1, 5)
        ), 
        justify="center"
    )

def generate_image_link(plant_name: str) -> str:
    """
    Generates a clean Google Images search link based on the extracted plant name.
    Uses URL encoding to prevent system crashes with special characters or spaces.
    
    Args:
        plant_name (str): The cleaned string representing the target plant.
        
    Returns:
        str: A rich-formatted link clickable directly inside modern terminals.
    """
    encoded_query = urllib.parse.quote(plant_name)
    url = f"https://www.google.com/search?tbm=isch&q={encoded_query}"
    return f"\n\n[bold yellow]Visualizar Planta:[/bold yellow] [link={url}][underline cyan]Clique aqui para ver fotos de {plant_name}[/underline cyan][/link]"

def main():
    """
    Main application runtime loop. Wraps the terminal environment, 
    manages agent synchronization (Agent 2 -> User Input -> Agent 3), 
    and handles global runtime anomalies.
    """
    # Clear terminal screen based on OS specification (Windows/Linux/Mac)
    os.system('cls' if os.name == 'nt' else 'clear')
    print_header()
    
    console.print("[bold dim white]LOG:[/bold dim white] Inicializando pipelines do sistema...", style="italic")
    
    # Core Orchestrator Bootstrapping
    try:
        orchestrator = BotanicalOrchestrator(model_name="llama3")
        console.print("[bold green]SUCESSO:[/bold green] Conexão com o banco de dados e agentes estabelecida.\n")
    except Exception as e:
        console.print(Panel(f"[bold red]ERRO CRÍTICO NO SISTEMA:[/bold red] {e}", border_style="red"))
        sys.exit(1)
        
    console.print("[bold yellow]Parabéns! Sistema Pronto.[/bold yellow]")
    console.print("Tire sua dúvida com o [bold green]Atendente Llama[/bold green].")
    console.print("Digite [bold red]exit[/bold red] ou [bold red]quit[/bold red] para encerrar o atendimento.\n")

    # Infinite interaction loop until user explicitly exits
    while True:
        try:
            # Capture user prompt via Rich terminal input
            user_input = Prompt.ask("[bold cyan]Atendente Llama: Como posso ajudar com a sua planta hoje?[/bold cyan]")
            
            # Sanitization of empty commands
            if not user_input.strip():
                continue
            # Graceful exit triggers
            if user_input.lower() in ["exit", "quit"]:
                console.print("\n[bold red]LOG:[/bold red] Atendimento encerrado. Obrigado por usar nosso suporte!", style="italic")
                break
                
            # Step 1 & 2 Execution: Triggers Agent 1 (Triage), RAG Search, and Agent 2 (Specialist)
            with Status("[bold magenta]O Atendente Llama está analisando e consultando os manuais técnicos...[/bold magenta]", spinner="bouncingBar") as status:
                raw_response = orchestrator.execute_workflow(user_input)
            
            # Data Extraction: Uses Regular Expressions to capture the hidden tag [PLANTA: Name]
            plant_match = re.search(r'\[PLANTA:\s*(.+?)\]', raw_response, re.IGNORECASE)
            
            visual_link = ""
            clean_response = raw_response
            
            # If a plant was successfully recognized by Agent 2, extract metadata and format UI
            if plant_match:
                plant_name = plant_match.group(1).strip()
                # Ensure the plant found is not a null token
                if plant_name.lower() not in ["nenhuma", "none", "unknown"]:
                    visual_link = generate_image_link(plant_name)
                # Cleaning pipeline: Removes the raw tag string so the end-user never sees the code logic
                clean_response = re.sub(r'\[PLANTA:\s*(.+?)\]', '', raw_response, flags=re.IGNORECASE).strip()
            
            # Merging LLM text with the dynamically generated terminal hyperlink
            final_display = clean_response + visual_link
            
            # Render the Agent 2 Specialist panel output
            console.print("\n")
            console.print(
                Panel(
                    final_display,
                    title="[bold green]RELATÓRIO CLÍNICO DA PLANTA[/bold green]",
                    title_align="left",
                    border_style="green",
                    padding=(1, 2),
                    subtitle="[dim white]Fonte: Base de Dados Local Verificada & Link Dinâmico[/dim white]"
                )
            )
            console.print("\n" + "—" * console.width + "\n", style="dim")
            
            # Step 3: Interactive Human-in-the-Loop Gateway for Agent 3 (Planner)
            if plant_match:
                plant_name = plant_match.group(1).strip()
                if plant_name.lower() not in ["nenhuma", "none", "unknown"]:
                    # Interactive feedback loop request
                    wants_routine = Prompt.ask(f"[bold cyan]Atendente Llama:[/bold cyan] Gostaria de um cronograma semanal de cuidados para a [bold yellow]{plant_name}[/bold yellow]? (s/n)")
                    
                    # If confirmed, trigger the sequential handoff to Agent 3
                    if wants_routine.lower() in ['s', 'sim', 'y', 'yes']:
                        with Status(f"[bold magenta]O Agente Planejador está montando a rotina ideal para a {plant_name}...[/bold magenta]", spinner="bouncingBar"):
                            routine_response = orchestrator.generate_routine(plant_name)
                        
                        # Render the Agent 3 Routine Planner panel output
                        console.print("\n")
                        console.print(
                            Panel(
                                routine_response,
                                title="[bold blue]📅 CRONOGRAMA DE CUIDADOS[/bold blue]",
                                title_align="left",
                                border_style="blue",
                                padding=(1, 2)
                            )
                        )
                        console.print("\n" + "—" * console.width + "\n", style="dim")
            
        except KeyboardInterrupt:
            # Handles unexpected terminal kill signals gracefully (e.g., Ctrl+C)
            console.print("\n[bold red]LOG:[/bold red] Conexão interrompida manualmente. Saindo.")
            break
        except Exception as e:
            # Global exception safety net to prevent total script crashes during evaluation
            console.print(Panel(f"[bold red]ANOMALIA DETECTADA:[/bold red] {e}", border_style="red", title="Falha no Pipeline"))

if __name__ == "__main__":
    main()