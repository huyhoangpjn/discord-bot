import vertexai
from langchain_google_vertexai import VertexAI, HarmBlockThreshold, HarmCategory
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing_extensions import override

from utils import helpers

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API')

vertexai.init(project=os.getenv('PROJECT_ID'), location="us-central1")

SAFETY_VIOLATION_MESSAGE = "Safety Violation Warning - Vi phạm tiêu chuẩn cộng đồng rồi bạn ơi!"

class BaseTextModel:
    def __init__(self, model_name="gemini-1.5-pro", history_window = 30):
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.llm = VertexAI(
            model_name=model_name,
            temperature=0.2,
            safety_settings=safety_settings,
        )
        
        self.history_window = history_window
        self.prompt = ChatPromptTemplate.from_messages(
            [   
                ('system', '''You are an helpful assistant for Mr Hoang's community. Your job is to answer user question and you may need look through a chat history for references.

                Here is the chat history inside the triple quote:
                """
                {chat_history}
                """

                Lastest user question inside the triple code:
                """
                {query}
                """

                Your response must follow these rules:
                - Answer the user query directly
                - No yapping
                ''')
            ]
        )

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
    def detect_link_in_query(self, state):
        pass
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
        chain = self.prompt|self.llm|StrOutputParser()
        response = chain.invoke({"chat_history": ChatPromptTemplate.from_messages(chat_history),
                "query": query})

        #print(response)
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

#model = BaseTextModel()
#model.invoke("print python code generate random number", [])