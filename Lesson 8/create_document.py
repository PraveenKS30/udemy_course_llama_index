from llama_index import Document
from llama_index import SimpleDirectoryReader

# auto create Documet using load_data() function
documents1 = SimpleDirectoryReader("data").load_data();
print(documents1)


# manuall creating Document
text_list = ["This is first sentence", "This is second sentence"]
documents = [Document(t) for t in text_list]
print(documents)
