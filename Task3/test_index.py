from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS

MODEL_NAME = "BAAI/bge-m3"
#MODEL_NAME = "intfloat/multilingual-e5-large"
INDEX_SAVE_PATH = "./faiss_index"

# Загружаем модель эмбеддингов
embeddings = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("[INFO] Загрузка индекса...")
vector_store = FAISS.load_local(
    INDEX_SAVE_PATH,
    embeddings
)

#query = "Моллари"
#query = "Импрессионизм"
#query = "Поросенок Фунтик"
query = "Паровоз"
#query = "Иван Грозный"
#query = "Павлик Морозов"
print(f"[INFO] Выполняется поиск по запросу: '{query}'")
results = vector_store.similarity_search_with_score(f"научная фантастика: {query}", k=3)

print(f"\n[RESULTS] Найдено {len(results)} наиболее релевантных чанка:")
for i, (doc, score) in enumerate(results):
    print(f"\n--- Результат #{i+1} ---")
    print(f"Текст: {doc.page_content[:200]}...")
    print(f"Релевантность: {score:.6f}")
    print(f"Источник: {doc.metadata.get('source')}")
    print(f"ID чанка: {doc.metadata.get('chunk_id', 'N/A')}")