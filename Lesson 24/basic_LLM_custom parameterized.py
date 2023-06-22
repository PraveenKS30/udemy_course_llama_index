from llama_index import LLMPredictor, GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.chat_models import ChatOpenAI
import os

os.environ['OPENAI_API_KEY'] = ""

# define LLM
# OpenAI, Cohere, AI21 -- max_tokens (default 256)
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=512))


service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

documents = SimpleDirectoryReader('input/text').load_data()
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)


query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)