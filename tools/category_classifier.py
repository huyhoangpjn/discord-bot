from openrouter.chatopenrouter import ChatOpenRouter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
import yaml

class CategoryClassifier:
    def __init__(self, model_name="anthropic/claude-3.5-sonnet:beta"):
        with open('config/models.yaml', 'r') as file:
            self.category = [category for category in yaml.safe_load(file)['category']]
        
        self.classifier = ChatOpenRouter(model_name=model_name)
        self.prompt = ChatPromptTemplate.from_template(
            template='''You are an expert in topic classification. You will be given a conversation history \
                between human and AI, and a new question.
                Your task is to classify the lastest question category.
                There are 7 categories to be classify: {category}.
                Note that you do not assess the entire chat history, only the latest question to be categorized. 
                The chat history is given in case the lastest question need to reference it to be fully understood.
                Answer the category of the question only.

                Here is the chat history inside the triple quotes:
                """
                {chat_history}
                """
                
                Latest question:
                {question}

                Answer:'''
        )

    def classify(self, question, chat_history):
        chain = self.prompt|self.classifier|StrOutputParser()
        return chain.invoke({"category": self.category, "chat_history": chat_history, "question": question})