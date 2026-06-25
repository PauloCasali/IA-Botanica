# Sistema de Suporte à Botânica Clínica
> Arquitetura Multi-Agentes Autônoma com RAG Local e Interação Human-in-the-Loop
Este projeto foi feito apenas por Paulo Henrique Casali Pereira e consiste em um sistema inteligente especialista em botânica residencial e clínica de plantas. Ele utiliza uma arquitetura avançada de múltiplos agentes para triagem, diagnóstico e planejamento de rotinas, alimentada por uma base de conhecimento local via RAG (Retrieval-Augmented Generation) acionada através do protocolo MCP (Model Context Protocol), garantindo execução 100% local e privada.

#Porque esse projeto foi feito?
O cultivo de plantas em ambientes residenciais e apartamentos é algo comum hoje em dia, entretanto, muitas pessoas trocam regularmente de plantas ou perdem suas plantas devido a falta de conhecimento técnico. Sintomas de rega excessiva, falta de nutrientes ou iluminação inadequada são frequentemente confundidos, levando à morte das plantas. 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Objetivo da Solução

1. Eliminar a alucinação botânica através do uso de um banco de dados vetorial local verificado.
2. Dividir a carga cognitiva da tarefa entre agentes especializados (Triagem, Especialista e Planejador).
3. Integrar o conceito de Human-in-the-Loop para acionamento de serviços sob demanda (geração de cronogramas).
4. Fornecer recursos de acessibilidade visual injetando hiperlinks dinâmicos em tempo real para identificação das plantas indicadas.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Descrição da Arquitetura Multi-Agente

O sistema adota uma topologia de execução linear e condicional controlada por um Orquestrador central (BotanicalOrchestrator). Em vez de um único prompt tentar resolver todo o problema, o fluxo de trabalho é segmentado em fases bem definidas:

[Usuário] -> (Input) -> [Agente 1: Triagem] -> (JSON) -> [MCP Server / ChromaDB]
                                                               |
[Painel Terminal] <- (Output) <- [Agente 2: Especialista] <- (Contexto Técnico)
       |
 (Deseja Cronograma? s/n)
       v
[Agente 3: Planejador] -> [Painel de Cronograma Semanal]
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Papel de cada Agente

1. Agente 1: Triage Agent (Agente de Triagem)
   Papel: Atua na camada de processamento de linguagem natural inicial. Sua única responsabilidade é analisar a queixa bruta do usuário em português e extrair a entidade principal (nome da planta ou sintoma patológico) convertendo-a estritamente em um objeto estruturado JSON.

2. Agente 2: Specialist Agent (Agente Botânico Especialista)
   Papel: Atua como o núcleo clínico. Ele recebe a pergunta do usuário e o contexto técnico fidedigno recuperado do banco vetorial. Sua função é redigir um relatório direto, curto e objetivo em português, sem introduções robóticas. Ele também é encarregado de embutir uma meta-tag estruturada ([PLANTA: Nome]) que será interceptada pelo sistema de backend.

3. Agente 3: Planner Agent (Agente Planejador)
   Papel: Atua de forma reativa sob demanda (Human-in-the-Loop). Se o usuário confirmar o interesse, este agente entra em ação isoladamente para transformar o diagnóstico anterior em uma rotina semanal detalhada em formato de checklist de cuidados (rega, luz e adubação).
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Descrição das Tools Disponíveis e Uso do MCP

O sistema utiliza o Model Context Protocol (MCP), um padrão aberto industrial para conectar LLMs a fontes de dados de forma segura e desacoplada. 

1. Tool Disponível: search_clinical_context
2. Funcionamento: O ecossistema possui um servidor MCP rodando localmente (mcp_server.py). Este servidor expõe a ferramenta de busca ao ecossistema LangChain. Quando o Agente 1 define a query de busca, o sistema invoca programmaticamente a ferramenta search_clinical_context através do protocolo MCP, abstraindo a complexidade de conexão com o banco de dados e permitindo que qualquer agente ou modelo consumisse a mesma ferramenta padronizada.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Explicação da Estratégia de RAG

A estratégia de Geração Aumentada por Recuperação (RAG) segue as seguintes etapas:
1. Recuperação Semântica: A consulta gerada pelo Agente 1 é convertida em um vetor denso.
2. Busca por Proximidade: O banco vetorial realiza o cálculo de similaridade de cosseno para trazer os fragmentos de manuais técnicos que possuem maior afinidade semântica com a dúvida do usuário.
3. Injeção de Contexto: Os dados retornados são acoplados ao prompt do Agente 2 sob uma regra rígida: Se a informação não estiver no contexto fornecido, o agente deve declarar que não possui esse conhecimento em sua base local, mitigando completamente o risco de alucinação.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Origem, Natureza da Base de Conhecimento e Tecnologia Vetorial

1. Origem e Natureza: A base de conhecimento é composta por arquivos textuais Markdown (data/) estruturados e categorizados por nichos botânicos (árvores_frutíferas.md, plantas_de_interior.md, plantas_carnívoras.md, etc.). Ela mapeia mais de 300 espécies com informações técnicas de solo, rega, umidade, fotoperíodo e tratamento de pragas.
2. Tecnologia de Embeddings: É utilizado o modelo de código aberto sentence-transformers (all-MiniLM-L6-v2), que converte os blocos textuais em vetores matemáticos de 384 dimensões capazes de capturar o real significado das palavras.
3. Armazenamento Vetorial: Adotou-se o ChromaDB, um banco de dados vetorial de alto desempenho embutido, que armazena os vetores e metadados no arquivo local chroma.sqlite3, dispensando a necessidade de servidores em nuvem ou infraestruturas complexas.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Modelo Local Utilizado e Forma de Execução

O sistema utiliza o modelo Llama 3 (8B) executado de forma 100% local através do Ollama. 
A execução local garante latência controlada, privacidade total dos dados (nenhuma pergunta ou manual técnico é enviado para servidores externos) e custo zero de infraestrutura de APIs. A comunicação do ecossistema Python com o modelo local é mediada pelos conectores nativos da biblioteca LangChain.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Dependências do Projeto

1. rich: Renderização visual avançada e gerenciamento de painéis/cores no terminal.
2. langchain e langchain-community: Orquestração, encadeamento de prompts e conectores do LLM local.
3. chromadb: Mecanismo de armazenamento e busca vetorial.
4. mcp: Protocolo de comunicação para o gerenciamento de ferramentas e contexto de dados.
5. sentence-transformers: Geração local de embeddings semânticos.
6. markdown: Processamento e leitura adequada dos arquivos de dados.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Instruções Completas para Instalação e Execução

--> Pré-requisitos:
1. Ter o Python 3.10 ou superior instalado.
2. Ter o Ollama instalado e rodando na máquina.

--> Requisitos Mínimos de Hardware
1. Memória RAM: Mínimo de 8 GB (Recomendado: 16 GB para evitar engasgos com o sistema operacional).
2. Armazenamento: Pelo menos 10 GB de espaço livre em disco (preferencialmente SSD) para o Ollama e o arquivo de pesos do Llama 3 (~4.7 GB).
3. Processador (CPU): Intel Core i5/i7 (8ª geração ou superior) ou AMD Ryzen 5/7.
4. Placa de Vídeo (GPU dedicada) recomendado pelo menos 6gb ou mais de Vram Caso a máquina não possua GPU dedicada compatível, o Ollama executará o processamento em modo de emulação por CPU, o que reduzirá a velocidade de geração das respostas.

Passo a Passo de Configuração:

1. Clonar o Repositório e Acessar a Raiz
   Windows e Linux/Mac:
   git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
   cd NOME_DO_REPOSITORIO

2. Criar o Ambiente Virtual (venv)
   Windows:
   python -m venv venv
   
   Linux/Mac:
   python3 -m venv venv

3. Ativar o Ambiente Virtual (venv)
   Windows:
   .\venv\Scripts\activate
   
   Linux/Mac:
   source venv/bin/activate

4. Instalar as Dependências
   Windows e Linux/Mac:
   pip install -r requirements.txt
   
5. Caso não tiver o Ollama, na raiz do projeto execute o comando :irm https://ollama.com/install.ps1 | iex

6. Também baixe o modelo Llama3  na raiz do projeto, execute o comando :ollama run llama3

7. Indexar a Base de Conhecimento (Criar o Banco Vetorial)
   Windows e Linux/Mac:
   python src/database/indexer.py

8. Executar a Aplicação Principal
   Windows e Linux/Mac:
   python src/main.py

9. Ao iniciar o sistema, você será recepcionado por um painel interativo, pergunte ao agente o que você quer saber.
