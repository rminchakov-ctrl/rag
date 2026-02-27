import os
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

# Конфигурация
DATA_PATH = "./../Task2/knowledge_base"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MODEL_NAME = "BAAI/bge-m3"
# MODEL_NAME = "intfloat/multilingual-e5-large"
INDEX_SAVE_PATH = "./faiss_index"

def extract_title(content):
    """Извлекает заголовок из первых строк текста"""
    first_lines = content.strip().split('\n')[:3]
    for line in first_lines:
        if line.strip() and len(line.strip()) > 3:
            # Ищем паттерны заголовков
            if re.match(r'^#', line) or re.match(r'^[A-ZА-Я][^.!?]*$', line):
                return line.strip().replace('#', '').strip()
    return "Без названия"

def clean_content(content):
    """Очищает контент от лишних переносов и пробелов"""
    # Заменяем множественные переносы на одинарные
    content = re.sub(r'\n\s*\n', '\n\n', content)
    # Убираем лишние пробелы
    content = re.sub(r'[ \t]+', ' ', content)
    return content.strip()

def load_documents(data_path):
    documents = []
    for file_name in os.listdir(data_path):
        if not file_name.endswith(('.txt', '.md', '.text')):
            continue
            
        file_path = os.path.join(data_path, file_name)
        try:
            print(f"[INFO] Загружается файл: {file_name}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            
            for doc in docs:
                cleaned_content = clean_content(doc.page_content)
                new_doc = Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata.copy()
                )
                new_doc.metadata['source'] = file_name
                new_doc.metadata['title'] = extract_title(cleaned_content)
                
                documents.append(new_doc)
                
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки {file_name}: {e}")
    
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "。", "！", "？", " "]  # Добавил Unicode разделители
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        chunk.metadata["chunk_id"] = i
        
        # Добавляем информацию о позиции в исходном документе
        if 'start_index' in chunk.metadata:
            chunk.metadata['position'] = f"{chunk.metadata['start_index']}-{chunk.metadata['start_index'] + len(chunk.page_content)}"
    
    print(f"[INFO] Документы разбиты на {len(chunks)} чанков.")
    print(f"[INFO] Средняя длина чанка: {sum(len(chunk.page_content) for chunk in chunks) // len(chunks)} символов")
    return chunks

def filter_low_quality_chunks(chunks, min_length=50, max_length=1200):
    """Фильтрует слишком короткие или слишком длинные чанки"""
    filtered_chunks = []
    removed_count = 0
    
    for chunk in chunks:
        content_length = len(chunk.page_content.strip())
        if min_length <= content_length <= max_length:
            filtered_chunks.append(chunk)
        else:
            removed_count += 1
    
    print(f"[INFO] Отфильтровано {removed_count} чанков по длине")
    return filtered_chunks

print("[INFO] Загрузка документов...")
raw_documents = load_documents(DATA_PATH)
if not raw_documents:
    raise ValueError(f"Не найдено ни одного документа в папке {DATA_PATH}.")

print("[INFO] Разбиение на чанки...")
chunks = split_documents(raw_documents)

print("[INFO] Фильтрация чанков по качеству...")
chunks = filter_low_quality_chunks(chunks)

print("[INFO] Загрузка модели эмбеддингов...")
embeddings = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("[INFO] Создание векторного индекса...")
vector_store = FAISS.from_documents(chunks, embeddings)

print(f"[INFO] Сохранение индекса в папку '{INDEX_SAVE_PATH}'...")
os.makedirs(INDEX_SAVE_PATH, exist_ok=True)
vector_store.save_local(INDEX_SAVE_PATH)

print("[SUCCESS] Процесс завершен!")
print(f"[INFO] Размер индекса: {vector_store.index.ntotal} векторов")
print(f"[INFO] Метadata: {list(chunks[0].metadata.keys()) if chunks else 'Нет чанков'}")