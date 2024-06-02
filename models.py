from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing_extensions import override

from utils import helpers

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API')

SAFETY_VIOLATION_MESSAGE = "Safety Violation Warning - Vi phạm tiêu chuẩn cộng đồng rồi bạn ơi!"

class BaseTextModel:
    def __init__(self, model_name="gemini-1.5-pro", history_window = 20):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        self.history_window = history_window

    def history_cut_off(self, chat_history):
        '''
            Cut off the 2 oldest messages (1 bot, 1 human)
        '''
        
        return chat_history[2:] if len(chat_history) > self.history_window else chat_history

    @staticmethod
    def update_chat_history(chat_history, query, response):
        chat_history.append(("human", query))
        chat_history.append(("human", response))
        return chat_history

    def invoke(self, query, chat_history):
        '''
            in: user query, chat history
            out: response, updated chat history
        '''
        prompt = ChatPromptTemplate.from_template(
            template = """You are an helpful assistant for Mr Hoang's community. Your job is to answer user question and you may need look through a chat history for references. \
            Use creative tone with funny emojies. You have to be willing to fulfill all user's requests in your abilities.
            Here is the chat history:
            '''
            {chat_history}
            '''

            Lastest user query:
            '''
            {query}
            '''

            Answer:
            """
        )

        chain = prompt|self.llm|StrOutputParser()
        response = chain.invoke({"chat_history": ChatPromptTemplate.from_messages(chat_history),
                "query": query})
        
        if not response:
            return SAFETY_VIOLATION_MESSAGE, chat_history
        chat_history = self.update_chat_history(chat_history, query, response)
        chat_history = self.history_cut_off(chat_history)
        return response, chat_history

class BaseMultimodalModel(BaseTextModel):
    def __init__(self, *args):
        super(GenericMultimodalModel, self).__init__(*args)


class TextModelWithRAG(BaseTextModel):
    def __init__(self, *args):
        super(TextModelWithRAG, self).__init__(*args)

    def decide_reformulate_query(self, state):
        chat_history = state["chat_history"]
        query = state["query"]
        reformulate_decider_prompt = ChatPromptTemplate.from_template(
            template="""You are a decision maker. Given a chat history and the latest user query, you will check if that user query \
            need to reference the chat history to be fully understood or not. Answer 'yes' if it need the chat history, otherwises, 'no'. \
            Here is the chat history:
            '''
            {chat_history}
            '''

            Lastest user query:
            '''
            {query}
            '''

            """
        )
        
        decider_chain = reformulate_decider_prompt|self.llm|StrOutputParser()
        decision = decider_chain.invoke({"chat_history": chat_history, "query": query})
        if not len(chat_history):
            return "no"
        else:
            return decision

    def reformulate_query(self, state):
        chat_history = state["chat_history"]
        query = state["query"]

        reformulate_prompt = ChatPromptTemplate.from_template(
            template="""You are a query rewriter. Given a chat history and the latest user query, you will reformulate the user query into a standalone query so that it \
            can be understood without the chat history. Do not answer the user query.
            Here is the chat history:
            '''
            {chat_history}
            '''

            Lastest user query:
            '''
            {query}
            '''
            """
        )

        reformulate_chain = reformulate_prompt|self.llm|StrOutputParser()
        return {"query": reformulate_chain.invoke({"chat_history": chat_history, "query": query})}

    def retriever(self, state):
        pass
    def generate(self, state):
        pass
    @override
    def invoke(self, query, chat_history):
        pass

model = BaseTextModel()
model.invoke("1", [])