import time
from langchain_google_genai import ChatGoogleGenerativeAI
import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import aiohttp

from models import BaseTextModel

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API')
TOKEN = os.getenv('DISCORD_TOKEN')
#GUILD = os.getenv('DISCORD_GUILD')
#intents = discord.Intents.all()

discord_bot = discord.Bot()
base_model = BaseTextModel(model_name="gemini-1.5-pro")
# Replaced with nosql db later
user_histories = {}

@discord_bot.event
async def on_ready():
    guild = discord_bot.guilds[0]
    print(
        f'{discord_bot.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )

@discord_bot.slash_command()
async def general_chat(ctx, question):
    await ctx.defer()

    user_id = ctx.author.id
    if user_id not in user_histories:
        user_histories[user_id] = []

    response, chat_history = base_model.invoke(question, user_histories[user_id])
    user_histories[user_id] = chat_history

    await ctx.respond(response)

discord_bot.run(TOKEN)