import os
import json
import logging
import hashlib

from datetime import datetime
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

CONFIG = {
    "data_path": "./knowledge_base",
    "index_save_path": "./faiss_index",
    "manifest_file": "./index_manifest.json",
    "log_file": "./logs/update_index.log",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "model_name": "intfloat/multilingual-e5-large",
    "supported_extensions": [".txt", ".md", ".text"]
}

def setup_logging():
    os.makedirs(os.path.dirname(CONFIG['log_file']), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG['log_file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def compute_file_hash(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Ошибка вычисления хеша для {file_path}: {e}")
        return None

def load_manifest():
    if os.path.exists(CONFIG["manifest_file"]):
        try:
            with open(CONFIG["manifest_file"], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки манифеста: {e}")
            return {}
    return {}

def save_manifest(manifest):
    try:
        with open(CONFIG["manifest_file"], 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Манифест сохранен: {CONFIG['manifest_file']}")
    except Exception as e:
        logger.error(f"Ошибка сохранения манифеста: {e}")

def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        try:
            logger.info(f"Загрузка файла: {os.path.basename(file_path)}")
            loader = TextLoader(file_path)
            docs = loader.load()
            
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = os.path.basename(file_path)
            documents.extend(docs)
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
    
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["processing_time"] = datetime.now().isoformat()
    
    logger.info(f"Документы разбиты на {len(chunks)} чанков.")
    return chunks

def get_new_files():
    manifest = load_manifest()
    new_files = []
    
    data_path = Path(CONFIG["data_path"])
    
    for file_path in data_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in CONFIG["supported_extensions"]:
            file_hash = compute_file_hash(file_path)
            if not file_hash:
                continue
                
            relative_path = str(file_path.relative_to(data_path))
            
            # Проверяем, новый ли это файл или измененный
            if relative_path not in manifest or manifest[relative_path] != file_hash:
                new_files.append(str(file_path))
                # Обновляем манифест
                manifest[relative_path] = file_hash
    
    return new_files, manifest

def update_index():
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("ЗАПУСК ОБНОВЛЕНИЯ ИНДЕКСА")
    logger.info(f"Время начала: {start_time}")
    
    if not os.path.exists(CONFIG["data_path"]):
        logger.error(f"Папка с данными не найдена: {CONFIG['data_path']}")
        return False
    
    new_files, manifest = get_new_files()
    
    if not new_files:
        logger.info("Новых или измененных файлов не обнаружено.")
        save_manifest(manifest)
        return True
    
    logger.info(f"Найдено новых/измененных файлов: {len(new_files)}")
    
    try:
        documents = load_documents(new_files)
        if not documents:
            logger.warning("Не удалось загрузить ни одного документа из новых файлов")
            return False
        
        chunks = split_documents(documents)
        
        logger.info("Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=CONFIG["model_name"],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if os.path.exists(CONFIG["index_save_path"]):
            logger.info("Загрузка существующего индекса...")
            vector_store = FAISS.load_local(
                CONFIG["index_save_path"], 
                embeddings
            )
            logger.info("Добавление новых документов в индекс...")
            vector_store.add_documents(chunks)
        else:
            logger.info("Создание нового индекса...")
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        logger.info(f"Сохранение индекса в '{CONFIG['index_save_path']}'...")
        vector_store.save_local(CONFIG["index_save_path"])
        
        save_manifest(manifest)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"ОБНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        logger.info(f"Время выполнения: {duration:.2f} секунд")
        logger.info(f"Обработано файлов: {len(new_files)}")
        logger.info(f"Добавлено чанков: {len(chunks)}")
        logger.info(f"Общий размер индекса: {vector_store.index.ntotal} векторов")
        
        return True
        
    except Exception as e:
        logger.error(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        return False

if __name__ == "__main__":
    logger = setup_logging()
    
    success = update_index()
    
    exit(0 if success else 1)