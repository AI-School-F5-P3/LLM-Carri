import requests
import json
import time
import os
import yfinance as yf
from typing import Dict, Union, Optional
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackManager
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from .models import ConversationTracker
from datetime import datetime, timedelta

load_dotenv()

def search_pexels_image(query: str) -> Optional[str]:
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    if not PEXELS_API_KEY:
        print("Warning: PEXELS_API_KEY not found in .env")
        return None
    
    headers = {
        "Authorization": PEXELS_API_KEY
    }

    try:
        response = requests.get(
            f'https://api.pexels.com/v1/search?query={query}&per_page=1',
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
    """Fetch stock market data using yfinance"""
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
    # Platform keywords dictionary
    PLATFORM_KEYWORDS = {
        "twitter": ["twitter", "tweet", "x platform", "x post"],
        "instagram": ["instagram", "insta", "ig", "reels"],
        "linkedin": ["linkedin", "professional", "business network"],
        "facebook": ["facebook", "fb", "meta", "facebook post"],
        "tiktok": ["tiktok", "tik tok", "short video"],
        "financial": ["stock", "market", "trading", "price", "shares", "ticker", 
                     "nasdaq", "nyse", "dow", "sp500", "financial"]
    }
    
    # Convert prompt to lowercase for case-insensitive matching
    prompt_lower = prompt.lower()

    if any(keyword in prompt_lower for keyword in PLATFORM_KEYWORDS["financial"]):
        # Extract stock symbols (assumed to be in $SYMBOL format)
        symbols = [word.strip('$') for word in prompt.split() if word.startswith('$')]
        if symbols:
            return "financial"
    
    # Check for platform mentions
    for platform, keywords in PLATFORM_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return platform
    
    return "general"

def generate_with_model(model: str, prompt: str, history: list = None, profile_data = None, user = None, session = None) -> str:
    tracker = ChatModelTracker()
    url = "http://localhost:11434/api/generate"
    
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in history:
            history_text += f"{msg['role']}: {msg['content']}\n"

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
    
    if platform == "financial":
        symbols = [word.strip('$') for word in prompt.split() if word.startswith('$')]
        market_data = {}
        for symbol in symbols:
            market_data[symbol] = get_stock_data(symbol)
        
        # Add market data to prompt
        full_prompt += f"\n\nCurrent Market Data:\n{json.dumps(market_data, indent=2)}"
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        content =  response.json()['response']
    
        try:
            parsed_content = json.loads(content)
            
            # Search for image based on image_prompt or text content
            image_query = parsed_content.get('image_prompt') or parsed_content.get('text').split('\n')[0]
            image_url = search_pexels_image(image_query)
            
            response_data = {
                "text": parsed_content['text'],
                "hashtags": parsed_content.get('hashtags', ''),
                "image_url": image_url
            }
                
        except json.JSONDecodeError:
            # If response isn't JSON, treat as plain text
            image_url = search_pexels_image(content[:100])  # Use first 100 chars as query
            response_data = {
                "text": content,
                "image_url": image_url
            }
        
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
                
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    

class ChatModelTracker:
    def __init__(self):
        self.message_history = ChatMessageHistory()
        self.start_time = None
    
    def _track_metrics(self, **kwargs):
        metrics = {
            'timestamp': time.time(),
            'generation_time': time.time() - self.start_time if self.start_time else 0,
            'prompt_tokens': len(kwargs.get('prompt', '').split()),
            'response_tokens': len(kwargs.get('response', '').split()),
        }
        return metrics

    def track_conversation(self, user, session, prompt, response, model, platform):
        self.start_time = time.time()
        metrics = self._track_metrics(prompt=prompt, response=response)
        
        ConversationTracker.objects.create(
            user=user,
            session=session,
            prompt=prompt,
            response=response,
            model_used=model,
            platform=platform,
            metrics=metrics
        )
        
        self.message_history.add_user_message(prompt)
        self.message_history.add_ai_message(response)
        
        return metrics