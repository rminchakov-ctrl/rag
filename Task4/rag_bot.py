from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
from local_llm_client import get_llm_client

class RAGBot:
    def __init__(self, index_path: str, model_path: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = self.load_vector_store(index_path)
        self.llm_client = get_llm_client(model_path)
    
    def load_vector_store(self, index_path: str):
        try:
            return FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        except TypeError:
            return FAISS.load_local(index_path, self.embeddings)
        except Exception as e:
            print(f"Ошибка загрузки векторного хранилища: {e}")
            return None
    
    def get_few_shot_examples(self, current_query: str, n_examples: int = 2) -> str:
        example_queries = [
            "кто такая Иванова",
            "планета Вода", 
            "население Сникерса"
        ]
        
        examples = []
        
        for i, query in enumerate(example_queries, 1):
            if len(examples) >= n_examples:
                break
                
            if self._is_too_similar(query, current_query):
                continue
                
            try:
                results = self.search_documents(query, k=1)
                
                if not results:
                    continue
                    
                doc, score = results[0]                
                if score > 0.5 and self._is_valid_example(doc.page_content, query):
                    example_text = self._format_example(query, doc.page_content)
                    examples.append(example_text)
                    
            except Exception as e:
                continue
        
        result = "\n\n".join(examples)
        return result

    def _format_example(self, question: str, context: str) -> str:
        sentences = context.split('.')
        answer = sentences[0].strip() + '.' if sentences else context[:100].strip()        
        return f"В: {question}\nО: {answer}"

    def _is_valid_example(self, context_text: str, query: str) -> bool:
        if not context_text or not query:
            return False
            
        # Базовая проверка: текст должен быть достаточно длинным
        if len(context_text.strip()) < 25:
            return False
            
        # Простая проверка на дублирование вопроса в ответе
        query_lower = query.lower()
        context_lower = context_text.lower()
        
        # Если ответ начинается с вопроса - плохой пример
        if context_lower.startswith(query_lower):
            return False
            
        # Если больше 50% слов вопроса есть в ответе - плохой пример
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())
        
        if query_words:
            common_words = query_words.intersection(context_words)
            if len(common_words) / len(query_words) > 0.5:
                return False
        
        return True

    def _is_too_similar(self, example_query: str, current_query: str) -> bool:
        from difflib import SequenceMatcher
        if not current_query.strip():
            return False
        similarity = SequenceMatcher(None, example_query.lower(), current_query.lower()).ratio()
        return similarity > 0.7

    def search_documents(self, query: str, k: int = 3):
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except:
            results = self.vector_store.similarity_search(query, k=k)
            return [(doc, 1.0) for doc in results]
    
    def generate_prompt(self, query: str, results: List) -> str:
        # Форматируем контекст
        context = ""
        for i, (doc, score) in enumerate(results, 1):
            context += f"Документ {i} (релевантность {score:.3f}):\n{doc.page_content}\n\n"
        
        few_shot_examples = self.get_few_shot_examples(query)
        
        prompt = f"""Ответь на вопрос используя предоставленные документы.
    # ПРИМЕРЫ ДЛЯ ОБУЧЕНИЯ (не показывать в ответе)
    {few_shot_examples}
    # ТЕКУЩАЯ ЗАДАЧА
    Контекст для анализа:
    {context}
    Вопрос: 
    {query}

    Ответь, следуя этим шагам:
    1. Проанализируй вопрос
    2. Используй информацию из документов  
    3. Сформулируй четкий ответ

    # ТВОЙ ОТВЕТ (только это показывать пользователю):"""
        return prompt

    def process_query(self, query: str):
        if not self.vector_store:
            return "Векторное хранилище не загружено."
        
        results = self.search_documents(query)        
        if not results:
            return "Релевантные документы не найдены."
        
        prompt = self.generate_prompt(query, results)
        
        # Запрос к локальной LLM
        llm_response = self.llm_client.generate(
            prompt=prompt,
            max_tokens=400,
            temperature=0.1
        )
        
        response = f"Вопрос: {query}\n\n"
        response += f"Найдено документов: {len(results)}\n"
        response += f"Самый релевантный документ: {results[0][1]:.3f}\n\n"
        response += "Ответ LLM:\n"
        response += "=" * 60 + "\n"
        response += llm_response + "\n"
        response += "=" * 60 + "\n\n"
        response += "Источники информации:\n\n"
        
        for i, (doc, score) in enumerate(results, 1):
            response += f"  Документ {i} (relevance: {score:.3f}):\n"
            response += f"  Источник: {doc.metadata.get('source', 'unknown')}\n"
            if 'page' in doc.metadata:
                response += f"  Страница: {doc.metadata['page']}\n"
            response += f"  Содержание: {doc.page_content[:200]}...\n\n"
        
        return response

def main():
    INDEX_SAVE_PATH = "./../Task3/faiss_index"
    MODEL_PATH = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    bot = RAGBot(INDEX_SAVE_PATH, MODEL_PATH)
    if not bot.vector_store:
        print("Не удалось загрузить векторное хранилище.")
        return
    
    model_info = bot.llm_client.get_model_info()
    print(f"Статус LLM: {model_info['status']}")
    
    print("Бот готов! Введите запрос (или 'quit' для выхода):")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Выход...")
                break
            
            if not user_input:
                continue
            
            response = bot.process_query(user_input)
            print("\n" + response)
            
        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()