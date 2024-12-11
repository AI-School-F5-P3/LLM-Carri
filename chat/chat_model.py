import arxiv
import json
import networkx as nx
import os
import requests
import time
import yfinance as yf

from datetime import datetime
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL
from typing import Dict, Union, Optional, List, Tuple

from .models import ConversationTracker, ScientificArticle

load_dotenv()

class ContentValidator:
    def __init__(self):
        # Umbrales de validación para el contenido
        self.MIN_WORDS = 20
        self.MAX_WORDS = 1000
        # Define las secciones requeridas para cada plataforma
        self.REQUIRED_SECTIONS = {
            "twitter": ["text", "hashtags"],
            "linkedin": ["text", "hashtags", "image_prompt"],
            "instagram": ["text", "hashtags", "image_prompt"],
            "facebook": ["text"],
            "tiktok": ["text", "hashtags"],
            "financial": ["text", "hashtags"],
            "scientific": ["text", "hashtags"]
        }
        # Marcadores específicos que deben aparecer en el contenido según la plataforma
        self.SECTION_MARKERS = {
            "linkedin": ["🎯", "💡", "🤔"],
            "financial": ["📊", "📈", "💡", "⚠️"],
            "scientific": ["🔬", "🤔", "💡", "🌟", "📚"]
        }

    def validate_structure(self, content: dict, platform: str) -> Tuple[bool, str]:
        # Verifica que el contenido tenga todas las secciones requeridas
        required = self.REQUIRED_SECTIONS.get(platform, ["text"])
        missing = [field for field in required if field not in content]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"

        # Valida que el texto tenga una longitud apropiada
        word_count = len(content['text'].split())
        if word_count < self.MIN_WORDS or word_count > self.MAX_WORDS:
            return False, f"Text length ({word_count} words) outside allowed range"

        # Verifica que el contenido incluya los marcadores de sección necesarios
        if platform in self.SECTION_MARKERS:
            missing_markers = [
                marker for marker in self.SECTION_MARKERS[platform]
                if marker not in content['text']
            ]
            if missing_markers:
                return False, f"Missing section markers: {', '.join(missing_markers)}"

        return True, "Content validation passed"

    def validate_content(self, content: str) -> bool:
        # Verifica si el contenido contiene patrones sospechosos de alucinaciones del modelo
        suspicious_patterns = [
            "As an AI", "I am an AI", "I apologize",
            "I cannot", "I don't have", "I'm unable"
        ]
        return not any(pattern in content for pattern in suspicious_patterns)

class EnhancedScientificRAG:
    def __init__(self):
        # Inicializa el modelo de embeddings para procesar texto científico
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Crea un grafo RDF para almacenar relaciones semánticas
        self.kg = Graph()
        # Crea un grafo NetworkX para visualización y análisis
        self.graph = nx.Graph()
        # Inicializa la base de conocimientos con conceptos científicos
        self._init_knowledge_base()
        
    def _init_knowledge_base(self):
        # Define conceptos científicos fundamentales y sus relaciones
        # Cada entrada contiene: (concepto, definición, conceptos relacionados)
        concepts = [
            ("quantum_mechanics", "Theory describing nature at atomic scale", 
             ["wave_function", "uncertainty_principle"]),
            ("wave_function", "Mathematical description of quantum state", 
             ["quantum_mechanics", "probability_amplitude"]),
            ("uncertainty_principle", "Fundamental limit of precision in measurements",
             ["quantum_mechanics", "wave_function"])
        ]
        
        # Construye el grafo añadiendo nodos y aristas
        for concept, definition, related in concepts:
            # Añade el nodo con su definición
            self.graph.add_node(concept, definition=definition)
            # Crea conexiones con conceptos relacionados
            for rel in related:
                self.graph.add_edge(concept, rel)

    def fetch_papers(self, query: str, max_results: int = 3):
        # Inicializa el cliente de ArXiv
        client = arxiv.Client()
        # Realiza la búsqueda en ArXiv con los parámetros especificados
        search = arxiv.Search(query=query, max_results=max_results)
        # Retorna los resultados como una lista
        return [result for result in client.results(search)]

    def get_context(self, query: str) -> Dict:
        # Obtiene artículos relevantes de ArXiv
        papers = self.fetch_papers(query)
        papers_context = []
        # Procesa cada artículo para extraer información relevante
        for paper in papers:
            papers_context.append({
                'title': paper.title,
                'summary': paper.summary[:500],  # Limita el resumen a 500 caracteres
                'url': paper.entry_id
            })

        # Identifica conceptos relevantes en la consulta que existen en el grafo
        concepts = [word.lower() for word in query.split() 
                   if word.lower() in self.graph.nodes]
        
        graph_data = None
        if concepts:
            # Construye un subgrafo con los conceptos relevantes y sus vecinos
            relevant_nodes = set(concepts)
            for concept in concepts:
                relevant_nodes.update(self.graph.neighbors(concept))
            
            # Crea un subgrafo para visualización
            subgraph = self.graph.subgraph(relevant_nodes)
            # Formatea los datos del grafo para la respuesta
            graph_data = {
                "nodes": [{
                    "id": node,
                    "definition": self.graph.nodes[node].get("definition", "")
                } for node in subgraph.nodes()],
                "links": [{"source": u, "target": v} 
                         for u, v in subgraph.edges()]
            }

        # Retorna el contexto completo con artículos y datos del grafo
        return {
            "papers": papers_context,
            "graph_data": graph_data
        }

def translate_to_english(text: str) -> str:
    try:
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def search_pexels_image(query: str) -> Optional[str]:
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    if not PEXELS_API_KEY:
        print("Warning: PEXELS_API_KEY not found in .env")
        return None
    
    english_query = translate_to_english(query)
    
    headers = {
        "Authorization": PEXELS_API_KEY
    }

    try:
        response = requests.get(
            f'https://api.pexels.com/v1/search?query={english_query}&per_page=1',
            headers = headers
        )
        response.raise_for_status()

        data = response.json()
        if data["photos"]:
            return data["photos"][0]["src"]["original"]
        return None
    except requests.exceptions.RequestException:
        return None

def get_stock_data(symbol: str) -> Dict:
    # Extraemos informacion financiera con yfinance
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        history = stock.history(period="1d")
        
        return {
            "symbol": symbol,
            "name": info.get('longName', symbol),
            "price": history['Close'].iloc[-1],
            "change": history['Close'].iloc[-1] - history['Open'].iloc[0],
            "change_percent": ((history['Close'].iloc[-1] - history['Open'].iloc[0]) / history['Open'].iloc[0]) * 100,
            "volume": info.get('volume', 0),
            "market_cap": info.get('marketCap', 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": f"Could not fetch data for {symbol}: {str(e)}"}


def detect_platform(prompt: str) -> str:
    # Diccionario que contiene palabras clave asociadas a cada plataforma para identificación
    PLATFORM_KEYWORDS = {
        "twitter": ["twitter", "tweet", "x platform", "x post"],
        "instagram": ["instagram", "insta", "ig", "reels"],
        "linkedin": ["linkedin", "professional", "business network"],
        "facebook": ["facebook", "fb", "meta", "facebook post"],
        "tiktok": ["tiktok", "tik tok", "short video"],
        "financial": ["stock", "market", "trading", "price", "shares", "ticker", 
                     "nasdaq", "nyse", "dow", "sp500", "financial"],
        "scientific": ["science", "quantum", "physics", "biology", "chemistry",
        "research", "study", "theory", "experiment", "scientific",
        "papers", "arxiv", "academic", "explain"]
    }
    
    # Convertimos el prompt a minúsculas para hacer la comparación sin distinción de mayúsculas/minúsculas
    prompt_lower = prompt.lower()

    # Verificamos si es contenido financiero buscando símbolos de acciones con formato $SYMBOL
    if any(keyword in prompt_lower for keyword in PLATFORM_KEYWORDS["financial"]):
        # Extraemos símbolos de acciones que comienzan con $
        symbols = [word.strip('$') for word in prompt.split() if word.startswith('$')]
        if symbols:
            return "financial"
    
    # Iteramos sobre cada plataforma y sus palabras clave para encontrar coincidencias
    # Si encontramos una coincidencia, retornamos esa plataforma
    for platform, keywords in PLATFORM_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return platform
    
    # Si no encontramos coincidencias, retornamos "general" como plataforma por defecto
    return "general"

def get_scientific_context(rag: EnhancedScientificRAG, prompt: str) -> Dict:
    """Obtiene contexto científico usando el sistema RAG (Retrieval Augmented Generation)"""
    try:
        # Obtiene el contexto inicial usando el sistema RAG
        context = rag.get_context(prompt)
        
        # Formatea la información de los artículos científicos en un texto estructurado
        # Incluye título, resumen y fuente de cada artículo
        papers_context = "\n".join([
            f"Title: {paper['title']}\n"
            f"Summary: {paper['summary']}\n"
            f"Source: {paper['url']}\n"
            for paper in context['papers']
        ])

        # Retorna un diccionario con el contexto formateado y los datos del grafo
        # Si no hay artículos, retorna una cadena vacía como contexto
        return {
            "text_context": papers_context if papers_context else "",
            "graph_data": context['graph_data']
        }
    except Exception as e:
        # En caso de error, imprime el mensaje y retorna un contexto vacío
        print(f"Scientific RAG Error: {str(e)}")
        return {"text_context": "", "graph_data": None}

def generate_with_model(model: str, prompt: str, history: list = None, profile_data = None, user = None, session = None) -> str:
    # Inicializa el validador de contenido y el rastreador de chat
    validator = ContentValidator()
    tracker = ChatModelTracker()
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    url = f"{ollama_host}/api/generate"
    max_retries = 3
    
    # Procesa el historial de conversación si existe
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in history:
            history_text += f"{msg['role']}: {msg['content']}\n"

    # Construye el contexto del perfil de la empresa si está disponible
    profile_context = ""
    if profile_data:
        profile_context = f"""
        Company Context:
        - Company Name: {profile_data['company_name']}
        - Company Description: {profile_data['company_description']}
        - Job Description: {profile_data['job_description']}
        
        Please consider this company context when generating responses.
        """

    # Platform-specific prompt templates
    PLATFORM_TEMPLATES = {
        "twitter": """You are a social media expert for Twitter/X posts.
        Create engaging content following this format:
        {
            "text": "Main tweet content (max 280 chars)",
            "hashtags": "space-separated hashtags (max 3)",
            "image_prompt": "description for image generation"
        }
        Guidelines:
        - Use emojis sparingly (1-2 max)
        - Include line breaks with \n
        - Create viral-style hooks
        - Keep it concise""",
        
        "linkedin": """You are a professional LinkedIn content strategist.
        Create content following this exact format:
        {
            "text": "Post content with following structure:
            • Hook (1-2 lines)
            \n\n
            🎯 Main Point
            [2-3 paragraphs with professional insights]
            \n\n
            💡 Key Takeaways:
            • Bullet point 1
            • Bullet point 2
            • Bullet point 3
            \n\n
            🤔 Thought-provoking question
            \n\n
            [Call to action]",
            "hashtags": "3-5 professional hashtags",
            "image_prompt": "professional image description"
        }""",
        
        "instagram": """You are an Instagram content creator.
        Create content following this format:
        {
            "text": "Caption with structure:
            ✨ Attention-grabbing first line
            \n\n
            [Main content with emojis]
            \n\n
            💫 Key points:
            • Point 1
            • Point 2
            • Point 3
            \n\n
            [Call to action + question]",
            "hashtags": "up to 30 relevant hashtags",
            "image_prompt": "instagram-worthy image description"
        }""",
        
        "facebook": """You are a Facebook content creator.
        Create content following this format:
        {
            "text": "Post with structure:
            [Engaging opening]
            \n\n
            [Story or main content with emojis]
            \n\n
            👉 Key message
            \n\n
            [Question to encourage comments]",
            "hashtags": "2-3 relevant hashtags",
            "image_prompt": "shareable image description"
        }""",
        
        "tiktok": """You are a TikTok content strategist.
        Create content following this format:
        {
            "text": "Script format:
            🎬 Hook (3 seconds):
            [Attention grabber]
            \n\n
            📱 Main Content:
            [Point 1] ⚡
            [Point 2] 💫
            [Point 3] 🔥
            \n\n
            🎵 Sound suggestion: [trending sound type]
            \n\n
            [Call to action]",
            "hashtags": "3-5 trending hashtags",
            "image_prompt": "vertical format thumbnail description"
        }""",
        
        "general": """You are a versatile social media content creator.
        Create content following this format:
        {
            "text": "Content with structure:
            [Engaging title]
            \n\n
            [Main content with appropriate formatting]
            \n\n
            [Call to action]",
            "hashtags": "relevant hashtags",
            "image_prompt": "appropriate image description"
        }""", 
        "financial": """You are a financial market expert.
        Create content following this format:
        {
            "text": "Market analysis with structure:
            📊 Market Update
            [Latest price and changes]
            \n\n
            📈 Technical Analysis
            [Key technical indicators and patterns]
            \n\n
            💡 Key Insights:
            • Point 1
            • Point 2
            • Point 3
            \n\n
            ⚠️ Risks and Considerations
            [Key risk factors]
            \n\n
            [Disclaimer]",
            "hashtags": "relevant financial hashtags",
            "image_prompt": "stock chart or financial visualization"
        }""",
        "scientific": """You are a science communicator expert.
        Create accessible scientific content, explaining in layman's terms, following this format:
        {
            "text": "Explanation with structure:
            🔬 Simple Title
            [Hook connecting to everyday life]
            \n\n
            🤔 The Big Question
            [Frame the scientific concept simply]
            \n\n
            💡 Key Points:
            • [Simple explanation 1]
            • [Simple explanation 2]
            • [Simple explanation 3]
            \n\n
            🌟 Real-World Impact
            [Practical applications]
            \n\n
            📚 Learn More:
            [Simplified paper references]",
            "hashtags": "relevant science hashtags",
            "image_prompt": "scientific visualization"
        }"""
    }
    
    # Detect platform from prompt
    platform = detect_platform(prompt)
    
    # Get template and combine with prompt
    system_prompt = PLATFORM_TEMPLATES.get(platform, PLATFORM_TEMPLATES["general"])
    full_prompt = f"""{system_prompt}
{profile_context}
{history_text}


Current request: {prompt}

Please continue the conversation while maintaining context from previous messages."""
    
    # Procesa datos financieros si es contenido relacionado con el mercado
    if platform == "financial":
        symbols = [word.strip('$') for word in prompt.split() if word.startswith('$')]
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = get_stock_data(symbol)
        
        full_prompt += f"\n\nCurrent Market Data:\n{json.dumps(market_data, indent=2)}"
    
    # Procesa datos científicos si es contenido relacionado con ciencia
    if platform == "scientific":
        rag = EnhancedScientificRAG()
        scientific_data = get_scientific_context(rag, prompt)
        
        if scientific_data["text_context"]:
            full_prompt += f"\n\nScientific Context:\n{scientific_data['text_context']}"
            full_prompt += "\nExplain this in simple terms for a general audience."

    # Implementa sistema de reintentos para generación de contenido
    for attempt in range(max_retries):
        try:
            # Configura y envía la solicitud al modelo
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            content = response.json()['response']
                
            try:
                # Intenta parsear la respuesta como JSON
                parsed_content = json.loads(content)
                
                # Busca una imagen relacionada con el contenido
                image_query = parsed_content.get('image_prompt') or parsed_content.get('text').split('\n')[0]
                image_url = search_pexels_image(image_query)
                
                # Estructura la respuesta final
                response_data = {
                    "text": parsed_content['text'],
                    "hashtags": parsed_content.get('hashtags', ''),
                    "image_url": image_url
                }
                
                # Valida la estructura y contenido de la respuesta
                is_valid_structure, message = validator.validate_structure(response_data, platform)
                is_valid_content = validator.validate_content(response_data['text'])
                    
                if is_valid_structure and is_valid_content:
                    # Registra la generación exitosa si hay usuario y sesión
                    if user and session:
                        tracker.track_conversation(
                            user=user,
                            session=session,
                            prompt=prompt,
                            response=json.dumps(response_data),
                            model=model,
                            platform=platform
                        )
                    return response_data
                
                print(f"Attempt {attempt + 1} failed validation: {message}")
                
            except json.JSONDecodeError:
                # Maneja respuestas que no son JSON
                image_url = search_pexels_image(content[:100])
                response_data = {
                    "text": content,
                    "hashtags": "",
                    "image_url": image_url
                }
                
                if validator.validate_content(content):
                    return response_data
                    
        except Exception as e:
            print(f"Generation attempt {attempt + 1} failed: {str(e)}")
                
    # Retorna respuesta por defecto si todos los intentos fallan
    return {
        "text": "I apologize, but I couldn't generate valid content. Please try rephrasing your request.",
        "hashtags": "",
        "image_url": None
    }


class ChatModelTracker:
    def __init__(self):
        # Historial de mensajes para mantener el contexto de la conversación
        self.message_history = ChatMessageHistory()
        # Variable para rastrear el tiempo de inicio de la generación de respuestas
        self.start_time = None
    
    def _track_metrics(self, **kwargs):
        # Calcula y retorna métricas importantes de la conversación:
        # - timestamp: momento exacto de la generación
        # - generation_time: tiempo total de generación
        # - prompt_tokens: cantidad aproximada de tokens en el prompt
        # - response_tokens: cantidad aproximada de tokens en la respuesta
        metrics = {
            'timestamp': time.time(),
            'generation_time': time.time() - self.start_time if self.start_time else 0,
            'prompt_tokens': len(kwargs.get('prompt', '').split()),
            'response_tokens': len(kwargs.get('response', '').split()),
        }
        return metrics

    def track_conversation(self, user, session, prompt, response, model, platform):
        # Inicia el cronómetro para medir el tiempo de generación
        self.start_time = time.time()
        # Obtiene las métricas de la conversación actual
        metrics = self._track_metrics(prompt=prompt, response=response)
        
        # Guarda la conversación en la base de datos incluyendo:
        # - Información del usuario y sesión
        # - Prompt y respuesta
        # - Modelo utilizado y plataforma
        # - Métricas calculadas
        ConversationTracker.objects.create(
            user=user,
            session=session,
            prompt=prompt,
            response=response,
            model_used=model,
            platform=platform,
            metrics=metrics
        )
        
        # Actualiza el historial de mensajes para mantener el contexto
        self.message_history.add_user_message(prompt)
        self.message_history.add_ai_message(response)
        
        return metrics
    
