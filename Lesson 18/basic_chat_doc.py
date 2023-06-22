from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os

os.environ['OPENAI_API_KEY'] = ""

documents = SimpleDirectoryReader('doc').load_data()
index = GPTVectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("Please provide me example of applications which uses Leader and Follower concept?")
print(response)
