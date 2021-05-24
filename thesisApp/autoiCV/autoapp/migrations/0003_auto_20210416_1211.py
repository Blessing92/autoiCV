# Generated by Django 3.2 on 2021-04-16 12:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('autoapp', '0002_contact'),
    ]

    operations = [
        migrations.AddField(
            model_name='contact',
            name='ip_address',
            field=models.GenericIPAddressField(null=True),
        ),
        migrations.AlterField(
            model_name='contact',
            name='email',
            field=models.EmailField(max_length=254),
        ),
    ]
