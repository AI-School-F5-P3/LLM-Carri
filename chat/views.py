from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from .models import ChatSession

def landing_page(request):
    return render(request, 'chat/landing.html')

def chat_page(request):
    return render(request, 'chat/chat.html')

def process_message(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        model = request.POST.get('model')
        # Here you would implement actual model interaction
        response = f"Response from {model}: {message}"
        return JsonResponse({'response': response})
    return JsonResponse({'error': 'Invalid request'}, status=400)