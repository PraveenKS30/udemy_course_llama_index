from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os

os.environ['OPENAI_API_KEY'] = ""

# load data and generate documents object
documents = SimpleDirectoryReader('data').load_data()

# print document id 
print(documents)

# create vector store index
index = GPTVectorStoreIndex.from_documents(documents)
print(index)


query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)


# print formatted sources
print(response.get_formatted_sources())
