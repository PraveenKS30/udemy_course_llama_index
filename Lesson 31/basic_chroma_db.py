import chromadb
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')

# get openai api key
os.environ['OPENAI_API_KEY'] = config["openai"]["api_key"]


# initialize chroma client
chroma_client = chromadb.Client()

# create chroma collection
chroma_collection = chroma_client.create_collection("quickstart")

documents = SimpleDirectoryReader('input/text').load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

# print embed model
print(index.service_context.embed_model)

query_engine = index.as_query_engine(
    chroma_collection=chroma_collection
)
response = query_engine.query("What did the author do growing up?")

print(chroma_collection.peek())
print(response)