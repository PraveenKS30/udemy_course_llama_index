from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import os

os.environ['OPENAI_API_KEY'] = ""

def construct_index(directory_path):

    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTVectorStoreIndex.from_documents(documents)

    # by default, data is stored in-memory. To persist to disk (under ./storage)
    index.storage_context.persist()

    return index 

def query_index(query):

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")

    # load index
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    print(response)

if __name__ == "__main__":
    construct_index("input/text")
