from django.db import models
import json
from django.contrib.auth.models import User
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

class ChatSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)  # Fecha y hora de creación automática
    model_choice = models.CharField(max_length=50)  # Modelo de lenguaje seleccionado
    conversation_history = models.TextField(default='[]')  # Almacena el historial como cadena JSON

    def add_message(self, role: str, content: str):
        """
        Agrega un nuevo mensaje al historial de conversación
        Args:
            role (str): Rol del mensaje (usuario o asistente)
            content (str): Contenido del mensaje
        """
        history = json.loads(self.conversation_history)
        history.append({"role": role, "content": content})
        self.conversation_history = json.dumps(history)
        self.save()

    def get_history(self):
        """
        Recupera el historial de conversación completo
        Returns:
            list: Lista de mensajes en formato JSON
        """
        return json.loads(self.conversation_history)
    
class CompanyProfile(models.Model):
    """
    Modelo que representa el perfil de una empresa.
    Almacena información específica sobre la empresa y su relación con el usuario.
    """
    # Relación uno a uno con el modelo User de Django
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Nombre de la empresa (límite de 200 caracteres)
    company_name = models.CharField(max_length=200)
    # Descripción general de la empresa
    company_description = models.TextField()
    # Descripción del puesto o posición laboral
    job_description = models.TextField()
    # Fecha y hora de creación del perfil
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        """
        Retorna una representación en cadena del perfil de la empresa
        """
        return f"{self.user.username}'s Company Profile"
    

class ConversationTracker(models.Model):
    """
    Modelo para rastrear y almacenar las conversaciones individuales entre usuarios y el chatbot.
    Registra detalles específicos de cada interacción para análisis y seguimiento.
    """
    # Relación con el usuario que inició la conversación
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # Vinculación con la sesión de chat correspondiente
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    # Marca temporal de cuando se registró la conversación
    timestamp = models.DateTimeField(auto_now_add=True)
    # Texto de entrada proporcionado por el usuario
    prompt = models.TextField()
    # Respuesta generada por el modelo
    response = models.TextField()
    # Identificador del modelo de IA utilizado
    model_used = models.CharField(max_length=50)
    # Plataforma desde donde se realizó la interacción
    platform = models.CharField(max_length=20)
    # Métricas adicionales de la conversación en formato JSON
    metrics = models.JSONField(default=dict)

    class Meta:
        # Ordenar las conversaciones por timestamp en orden descendente
        ordering = ['-timestamp']

class ScientificArticle(models.Model):
    """
    Modelo para almacenar artículos científicos y sus metadatos asociados.
    Incluye información bibliográfica y embeddings vectoriales para búsquedas semánticas.
    """
    # Título del artículo científico
    title = models.CharField(max_length=500)
    # Resumen o abstract del artículo
    abstract = models.TextField()
    # Lista de autores almacenada en formato JSON
    authors = models.JSONField()
    # Identificador único del artículo en arXiv
    arxiv_id = models.CharField(max_length=50, unique=True)
    # Categorías o áreas temáticas del artículo en formato JSON
    categories = models.JSONField()
    # Embedding vectorial del artículo para búsquedas semánticas
    vector_embedding = models.JSONField(null=True)
    # Fecha y hora en que se agregó el artículo a la base de datos
    added_date = models.DateTimeField(auto_now_add=True)


