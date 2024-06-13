# import langchain
# from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from pydantic import BaseModel
from typing import Any, Mapping, Optional, List, Dict
import markdown
import vertexai
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from google.cloud import aiplatform
from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from supabase.client import  create_client
from google.cloud import texttospeech
# from google.cloud import speech_v1p1beta1
from google.cloud import speech
# from google.cloud import speech
from libs.vertexAI.AudioConversion import mp3_to_wav_bytes
import traceback
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools

# Load environment variables
load_dotenv()
dotenv_result = load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# os.environ['SERPER_API_KEY'] = "85a3148783d1539b6f59b5eca1968edd4d66f0d1"
tools = load_tools(["google-serper"])

# client = texttospeech.TextToSpeechClient()

print("Dotenv Result:", dotenv_result)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# cred = service_account.Credentials.from_service_account_file("google-credentials.json")
# print(cred)
# client = speech.SpeechClient()


PROJECT_ID = "travel-407110"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI SDK
vertexai.init(project="travel-407110", location="us-central1")

embeddings = OpenAIEmbeddings()



class GenerateSuggestions:
    def __init__(self):

        self.llm = VertexAI(
            model_name='text-bison',
            max_output_tokens=1024,
            temperature=0.1,
            top_p=0.8,
            top_k=40,
            verbose=True,
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")

  

    def generate(self, input_text):

  
        agent = initialize_agent(tools, self.llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  verbose=True, memory = self.memory )
    # agent.input_keys= 
        final = agent.run(input = input_text)
        return final
