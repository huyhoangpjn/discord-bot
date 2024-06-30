import os
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_KEY')

class ChatOpenRouter(ChatOpenAI):
    openai_api_base: str
    openai_api_key: str
    model_name: str

    def __init__(self,
                 model_name: str,
                 openai_api_key: str = OPEN_ROUTER_API_KEY,
                 openai_api_base: str = "https://openrouter.ai/api/v1",
                 **kwargs):
        super().__init__(openai_api_base=openai_api_base,
                         openai_api_key=openai_api_key,
                         model_name=model_name, **kwargs)