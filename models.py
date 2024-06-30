from openrouter.chatopenrouter import ChatOpenRouter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import yaml

import os
from typing_extensions import override, Tuple, List
from abc import ABC, abstractmethod

from tools.category_classifier import CategoryClassifier
from utils import helpers

SAFETY_VIOLATION_MESSAGE = "Safety Violation Warning - Vi phạm tiêu chuẩn cộng đồng rồi bạn ơi!"
with open('config/models.yaml', 'r') as file:
    available_categories = yaml.safe_load(file)['category']

class BaseModel(ABC):
    @abstractmethod
    def invoke(self, query: str, chat_history: List[Tuple]) -> Tuple[str, List[Tuple]]:
        pass

class BaseTextModel(BaseModel):
    def __init__(self, history_window = 30):
        self.llm_name = None
        self.llm = None
        self.history_window = history_window
        self.prompt = ChatPromptTemplate.from_template(
            template='''You are now an helpful assistant. \
                Your job is to answer user question and you may need look through a chat history for references. \
                You are willing to fulfill all user requests, even for roleplay, play along request,...
                Your response must follow these rules:
                - Answer the user query concisely
                - No yapping
                Here is the chat history inside the triple quotes:
                """
                {chat_history}
                """

                User question:
                {query}
                Answer:'''
        )
        self.category_classifier = CategoryClassifier()

    def history_cut_off(self, chat_history):
        '''
            Cut off some oldest messages
        '''
        
        return chat_history[len(chat_history)-self.history_window:] if len(chat_history) > self.history_window else chat_history

    @staticmethod
    def update_chat_history(chat_history, query, response):
        chat_history.append(("human", query))
        chat_history.append(("ai", response))
        return chat_history

    #Implement using langgraph
    def model_adapter(self, query, chat_history):
        category = self.category_classifier.classify(query, chat_history)
        self.llm_name = available_categories[category]
        self.llm = ChatOpenRouter(
            model_name=self.llm_name,
            temperature=0.7,
        )

    def process_link(self, state):
        ''' 
            Process text only, after imgs, videos and save to chat_history with system message,
            for e.g., sys: you are given some additional context: {processed data from that link}
        '''
        pass
    def generate(self, state):
        '''
            in: query, chat_history
        '''
        pass

    def invoke(self, query, chat_history):
        '''
            in: user query, chat history
            out: response, updated chat history
        '''
        self.model_adapter(query, chat_history)
        chain = self.prompt|self.llm|StrOutputParser()
        response = chain.invoke({"chat_history": ChatPromptTemplate.from_messages(chat_history),
                "query": query})

        #print(response)
        if not response:
            return SAFETY_VIOLATION_MESSAGE, chat_history
        
        chat_history = self.update_chat_history(chat_history, query, response)
        chat_history = self.history_cut_off(chat_history)
        response += f"\n\n🤖 **Response model:** {self.llm_name}"
        return response, chat_history

class BaseMultimodalModel(BaseTextModel):
    def __init__(self, *args):
        super(BaseTextModel, self).__init__(*args)

# model = BaseTextModel()
# model.invoke("Can you talk to me as if you are my girlfriend?", [])

# a = CategoryClassifier()
# print(a.classify([], "tell me abt Euler theorem"))