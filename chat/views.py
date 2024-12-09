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
from.models import ConversationTracker

import logging
logger = logging.getLogger(__name__)

def landing_page(request):
    return render(request, 'chat/landing.html')

def chat_page(request):
    return render(request, 'chat/chat.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username').lower()
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('chat')  # Change to your home view name
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
def profile_view(request):
    profile = CompanyProfile.objects.get(user=request.user)
    
    if request.method == 'POST':
        # Update user data
        request.user.username = request.POST.get('username').lower()
        request.user.save()
        
        # Update profile data
        profile.company_name = request.POST.get('company_name')
        profile.company_description = request.POST.get('company_description')
        profile.job_description = request.POST.get('job_description')
        profile.save()
        
        messages.success(request, 'Profile updated successfully')
        return redirect('profile')
        
    return render(request, 'chat/profile.html', {'profile': profile})

@csrf_exempt
def process_message(request):
    if request.method == 'POST':
        try:
            logger.debug(f"Request body: {request.body}")

            data = json.loads(request.body)
            user_message = data.get('message')
            selected_model = data.get('model')
            session_id = data.get('session_id')

            session = ChatSession.objects.get(id=data.get('session_id')) if data.get('session_id') else ChatSession.objects.create()

            profile = CompanyProfile.objects.get(user=request.user)
            profile_data = {
                'company_name': profile.company_name,
                'company_description': profile.company_description,
                'job_description': profile.job_description
            }

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
                history,
                profile_data,
                user = request.user,
                session = session
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

@login_required(login_url='login')
def history_view(request):
    # Get user's conversation history
    conversations = ConversationTracker.objects.filter(user=request.user).order_by('-timestamp')
    
    # Group by session
    sessions = {}
    for conv in conversations:
        if conv.session_id not in sessions:
            sessions[conv.session_id] = {
                'date': conv.timestamp.strftime("%Y-%m-%d"),
                'messages': []
            }
        
        try:
            response_data = json.loads(conv.response)
            response_text = response_data.get('text', conv.response)
        except json.JSONDecodeError:
            response_text = conv.response
            
        sessions[conv.session_id]['messages'].append({
            'prompt': conv.prompt,
            'response': response_text,
            'platform': conv.platform,
            'timestamp': conv.timestamp.strftime("%H:%M:%S"),
            'model': conv.model_used
        })
    
    return render(request, 'chat/history.html', {
        'sessions': sessions
    })