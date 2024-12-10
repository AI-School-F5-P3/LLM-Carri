# Generated by Django 5.1.4 on 2024-12-10 00:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0006_conversationtracker'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScientificArticle',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=500)),
                ('abstract', models.TextField()),
                ('authors', models.JSONField()),
                ('arxiv_id', models.CharField(max_length=50, unique=True)),
                ('categories', models.JSONField()),
                ('vector_embedding', models.JSONField(null=True)),
                ('added_date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
