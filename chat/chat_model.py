import requests
import json
import time
import os
from typing import Dict, Union, Optional
from dotenv import load_dotenv

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



def detect_platform(prompt: str) -> str:
    # Platform keywords dictionary
    PLATFORM_KEYWORDS = {
        "twitter": ["twitter", "tweet", "x platform", "x post"],
        "instagram": ["instagram", "insta", "ig", "reels"],
        "linkedin": ["linkedin", "professional", "business network"],
        "facebook": ["facebook", "fb", "meta", "facebook post"],
        "tiktok": ["tiktok", "tik tok", "short video"]
    }
    
    # Convert prompt to lowercase for case-insensitive matching
    prompt_lower = prompt.lower()
    
    # Check for platform mentions
    for platform, keywords in PLATFORM_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            return platform
    
    return "general"

def generate_with_model(model: str, prompt: str, history: list = None, profile_data = None) -> str:
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
            
            return {
                "text": parsed_content['text'],
                "hashtags": parsed_content.get('hashtags', ''),
                "image_url": image_url
            }
                
        except json.JSONDecodeError:
            # If response isn't JSON, treat as plain text
            image_url = search_pexels_image(content[:100])  # Use first 100 chars as query
            return {
                "text": content,
                "image_url": image_url
            }
                
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}