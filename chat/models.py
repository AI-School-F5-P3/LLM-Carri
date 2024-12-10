from django.db import models
import json
from django.contrib.auth.models import User
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

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
    

class ConversationTracker(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    prompt = models.TextField()
    response = models.TextField()
    model_used = models.CharField(max_length=50)
    platform = models.CharField(max_length=20)
    metrics = models.JSONField(default=dict)

    class Meta:
        ordering = ['-timestamp']

class ScientificArticle(models.Model):
    title = models.CharField(max_length=500)
    abstract = models.TextField()
    authors = models.JSONField()
    arxiv_id = models.CharField(max_length=50, unique=True)
    categories = models.JSONField()
    vector_embedding = models.JSONField(null=True)
    added_date = models.DateTimeField(auto_now_add=True)


