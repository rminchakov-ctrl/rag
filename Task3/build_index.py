import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Конфигурация
DATA_PATH = "./../Task2/knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MODEL_NAME = "BAAI/bge-m3"
INDEX_SAVE_PATH = "./faiss_index"

def load_documents(data_path):
    documents = []
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        loader = TextLoader(file_path)

        try:
            print(f"[INFO] Загружается файл: {file_name}")
            docs = loader.load()
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_name
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки {file_name}: {e}")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"[INFO] Документы разбиты на {len(chunks)} чанков.")
    return chunks

# Основной процесс
print("[INFO] Загрузка документов...")
raw_documents = load_documents(DATA_PATH)
if not raw_documents:
    raise ValueError(f"Не найдено ни одного документа в папке {DATA_PATH}.")

print("[INFO] Разбиение на чанки...")
chunks = split_documents(raw_documents)

print("[INFO] Загрузка модели эмбеддингов...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("[INFO] Создание векторного индекса...")
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"[INFO] Сохранение индекса в папку '{INDEX_SAVE_PATH}'...")
vector_store.save_local(INDEX_SAVE_PATH)

print("[SUCCESS] Процесс завершен! Индекс и метаданные сохранены.")