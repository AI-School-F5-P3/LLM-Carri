from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import ChatSession, CompanyProfile
from django.views.decorators.csrf import csrf_exempt
from .chat_model import generate_with_model
import json
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

import logging
logger = logging.getLogger(__name__)

def landing_page(request):
    return render(request, 'chat/landing.html')

def chat_page(request):
    return render(request, 'chat/chat.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('home')  # Change to your home view name
        else:
            messages.error(request, 'Invalid username or password')
    
    return render(request, 'chat/login.html')

def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        company_name = request.POST.get('company_name')
        company_description = request.POST.get('company_description')
        job_description = request.POST.get('job_description')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return render(request, 'chat/signup.html')

        user = User.objects.create_user(username=username, password=password)
        CompanyProfile.objects.create(
            user=user,
            company_name=company_name,
            company_description=company_description,
            job_description=job_description
        )
        login(request, user)
        return redirect('chat')

    return render(request, 'chat/signup.html')

def logout_view(request):
    logout(request)
    return redirect('landing')

@login_required(login_url='login')
def chat_page(request):
    return render(request, 'chat/chat.html')

@login_required(login_url='login')

@csrf_exempt
def process_message(request):
    if request.method == 'POST':
        try:
            logger.debug(f"Request body: {request.body}")

            data = json.loads(request.body)
            user_message = data.get('message')
            selected_model = data.get('model')
            session_id = data.get('session_id')

            if not user_message or not selected_model:
                return JsonResponse({
                    'error': 'Missing required fields: message and model'
                }, status=400)
            
            try:
                if session_id:
                    session = ChatSession.objects.get(id=session_id)
                else:
                    session = ChatSession.objects.create(model_choice=selected_model)
            except ChatSession.DoesNotExist:
                session = ChatSession.objects.create(model_choice=selected_model)

            session.add_message('user', user_message)

            history = session.get_history()

            model_mapping = {
                'llama3.2-3b': 'llama3.2',
                'mistral-7b': 'mistral'
            }
            
            model_response = generate_with_model(
                model_mapping.get(selected_model, 'llama3.2'),
                user_message,
                history
            )

            session.add_message('assistant', model_response)
            
            return JsonResponse({
                'response': model_response,
                'model': selected_model,
                'session_id': session.id
            })
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=400)