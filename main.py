import time
from langchain_google_genai import ChatGoogleGenerativeAI
import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
from utils import helpers
import asyncio

from models import BaseTextModel

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
#GUILD = os.getenv('DISCORD_GUILD')
#intents = discord.Intents.all()

discord_bot = discord.Bot()
base_model = BaseTextModel()
# Replaced with nosql db later
user_histories = {}

@discord_bot.event
async def on_ready():
    guild = discord_bot.guilds[0]
    print(
        f'{discord_bot.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )

@discord_bot.slash_command(id=[1246755720561692723])
async def general_chat(ctx, question):
    await ctx.defer()

    user_id = ctx.author.id
    if user_id not in user_histories:
        user_histories[user_id] = []

    response, chat_history = base_model.invoke(question, user_histories[user_id])
    user_histories[user_id] = chat_history

    split_response = helpers.split_message(response)
    for message in split_response:
        # This will ensure if user prompts something having no line break
        if len(message) > 2000:
            await ctx.respond("Response exceeds 2000 characters")
            break
        else:
            await ctx.respond(message)

discord_bot.run(TOKEN)

# Task can giai quyet:
# Too long response
# satety failed - ok (time out)