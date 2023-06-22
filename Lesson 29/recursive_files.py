from llama_index import SimpleDirectoryReader

def mul_file_load(input_dir):
    # load multiple files
    documents = SimpleDirectoryReader(input_dir).load_data()

    # print document ids
    for doc in documents:
        print(doc.doc_id)

def rec_file_load(input_dir):
    # load files from the subfolders recursively
    documents = SimpleDirectoryReader(input_dir = input_dir, recursive=True, filename_as_id=True).load_data()

    # print document ids
    for doc in documents:
        print(doc.doc_id)

if __name__ == '__main__':
    mul_file_load("input/text")
    rec_file_load("input")