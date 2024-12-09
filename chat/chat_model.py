import requests
import json
import time

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

def generate_with_model(model: str, prompt: str, history: list = None) -> str:
    url = "http://localhost:11434/api/generate"
    
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in history:
            history_text += f"{msg['role']}: {msg['content']}\n"

    # Platform-specific prompt templates
    PLATFORM_TEMPLATES = {
        "twitter": """You are a social media expert specialized in Twitter/X posts.
        Create engaging content that:
        - Fits within character limits
        - Uses appropriate hashtags
        - Has a conversational tone
        - Encourages engagement
        - Is trendy and relevant
        Format: Main text + Hashtags""",
        
        "instagram": """You are an Instagram content creator.
        Create engaging content that:
        - Has an attention-grabbing first line
        - Uses storytelling elements
        - Includes relevant emoji
        - Has structured paragraphs
        - Ends with a call to action
        - Includes hashtag suggestions
        Format: Caption + Hashtags (separated)""",
        
        "linkedin": """You are a professional LinkedIn content strategist.
        Create content that:
        - Starts with a hook
        - Shares professional insights
        - Uses business-appropriate tone
        - Includes industry expertise
        - Encourages professional discussion
        - Ends with a thought-provoking question
        Format: Professional post with paragraphs""",
        
        "facebook": """You are a Facebook content creator.
        Create engaging content that:
        - Is personal and relatable
        - Encourages comments and shares
        - Uses appropriate emoji
        - Includes call to action
        Format: Post with engaging question""",
        
        "tiktok": """You are a TikTok content strategist.
        Create script/content that:
        - Has a hook in first 3 seconds
        - Is trendy and engaging
        - Uses popular audio references
        - Includes hashtag suggestions
        Format: Script + Hashtags""",
        
        "general": """You are a versatile social media content creator.
        Create engaging content that:
        - Is clear and concise
        - Uses appropriate tone
        - Engages the audience
        - Includes relevant hashtags"""
    }
    
    # Detect platform from prompt
    platform = detect_platform(prompt)
    
    # Get template and combine with prompt
    system_prompt = PLATFORM_TEMPLATES.get(platform, PLATFORM_TEMPLATES["general"])
    full_prompt = f"""{system_prompt}
    
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
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"