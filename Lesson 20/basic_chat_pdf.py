from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os

os.environ['OPENAI_API_KEY'] = ""

documents = SimpleDirectoryReader('pdf').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Explain the concept of encoders and decoders")
print(response)