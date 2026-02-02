from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

MODEL_NAME = "BAAI/bge-m3"
INDEX_SAVE_PATH = "faiss_index"

# Загружаем модель эмбеддингов
embeddings = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("[INFO] Загрузка индекса...")
try:
    # Пробуем загрузить с параметром (для новых версий)
    vector_store = FAISS.load_local(
        INDEX_SAVE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
except TypeError:
    print("[INFO] Попытка загрузки без allow_dangerous_deserialization...")
    vector_store = FAISS.load_local(
        INDEX_SAVE_PATH,
        embeddings
    )

query = "Расскажи про Моллари"
print(f"[INFO] Выполняется поиск по запросу: '{query}'")
results = vector_store.similarity_search(query, k=3)

print(f"\n[RESULTS] Найдено {len(results)} наиболее релевантных чанка:")
for i, doc in enumerate(results):
    print(f"\n--- Результат #{i+1} ---")
    print(f"Текст: {doc.page_content[:200]}...")
    print(f"Источник: {doc.metadata.get('source')}")
    print(f"ID чанка: {doc.metadata.get('chunk_id', 'N/A')}")