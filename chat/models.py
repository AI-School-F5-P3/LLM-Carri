from django.db import models
import json
from django.contrib.auth.models import User

class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    model_choice = models.CharField(max_length=50)
    conversation_history = models.TextField(default='[]')  # Store as JSON string

    def add_message(self, role: str, content: str):
        history = json.loads(self.conversation_history)
        history.append({"role": role, "content": content})
        self.conversation_history = json.dumps(history)
        self.save()

    def get_history(self):
        return json.loads(self.conversation_history)
    
class CompanyProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company_name = models.CharField(max_length=200)
    company_description = models.TextField()
    job_description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s Company Profile"