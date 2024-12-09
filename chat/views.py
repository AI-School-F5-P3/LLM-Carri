from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from .models import ChatSession
from django.views.decorators.csrf import csrf_exempt
from .test_models import generate_with_model
import json

def landing_page(request):
    return render(request, 'chat/landing.html')

def chat_page(request):
    return render(request, 'chat/chat.html')

@csrf_exempt
def process_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message')
            selected_model = data.get('model')
            
            model_mapping = {
                'llama3.2-3b': 'llama3.2',
                'mistral-7b': 'mistral'
            }
            
            model_response = generate_with_model(
                model_mapping.get(selected_model, 'llama3.2'), 
                user_message
            )
            
            return JsonResponse({
                'response': model_response,
                'model': selected_model
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)