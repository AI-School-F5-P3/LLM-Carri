from django.db import models

# Create your models here.
class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    model_choice = models.CharField(max_length=50)

class Message(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    content = models.TextField()
    is_user = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)