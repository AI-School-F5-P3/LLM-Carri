from django.db import models
import json

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