import os
from typing import Optional, Dict, Any

class LocalLLMClient:
    def __init__(self, model_path: str):
        """
        Инициализация клиента для работы с локальной LLM
        
        Args:
            model_path: Путь к файлу модели .gguf
        """
        self.model_path = model_path
        self.llm = None
        self.is_available = False
        self.context_size = 2048  # Значение по умолчанию
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Инициализация LLM модели"""
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Модель {self.model_path} не найдена")
            
            print(f"Загрузка модели: {os.path.basename(self.model_path)}")
            
            # Создаем экземпляр Llama с базовыми параметрами
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,          # Размер контекста
                n_threads=6,         # Количество потоков
                n_gpu_layers=0,      # 0 = только CPU
                verbose=False
            )
            
            self.is_available = True
            
            # Пытаемся получить реальный размер контекста
            try:
                if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'n_ctx'):
                    self.context_size = self.llm.model.n_ctx()
                elif hasattr(self.llm, 'ctx') and hasattr(self.llm.ctx, 'n_ctx'):
                    self.context_size = self.llm.ctx.n_ctx()
            except:
                # Если не получается, оставляем значение по умолчанию
                pass
                
            print(f"✓ Локальная LLM загружена: {os.path.basename(self.model_path)}")
            print(f"  Размер контекста: {self.context_size}")
            
        except ImportError:
            print("✗ Ошибка: Установите llama-cpp-python: pip install llama-cpp-python")
            self.is_available = False
        except FileNotFoundError as e:
            print(f"✗ {e}")
            self.is_available = False
        except Exception as e:
            print(f"✗ Ошибка загрузки модели: {e}")
            self.is_available = False
    
    def generate(self, prompt, max_tokens=500, temperature=0.1, top_p=0.9, stop=None):
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system", 
                    "content": """
Ты — дружелюбный и профессиональный консультант.
Всегда отвечай на русском языке, вежливо и по делу.

  КРИТИЧЕСКИЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО СОБЛЮДАЙ, ИНАЧЕ ОШИБКА!):

  - Ты всегда завершаешь цепочку рассуждений. 
  - Отвечай только на основе контекста. 
  - Не копируй вопросы в ответы. 
  - Не копируй примеры в ответы.
  - Отвечай максимально быстро. Старайся уложиться в 2–4 шага.
  - НЕ зацикливайся и НЕ переспрашивай без необходимости.
  - Никогда не придумывай ответы.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or ["ОТВЕТ:", "ВОПРОС:"]
        )
        
        text = response['choices'][0]['message']['content'].strip()
        
        # Гарантируем завершение
        if "ОТВЕТ:" not in text:
            text += "\nОТВЕТ: "
        
        return text

    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженной модели"""
        if not self.is_available:
            return {"status": "not_available", "message": "Модель не загружена"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "context_size": self.context_size,
            "available": self.is_available
        }
    
    def test_llm(self) -> str:
        if not self.is_available:
            return "LLM не доступна"
        
        try:
            test_prompt = "Как тебя зовут?"
            
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Ты helpful assistant."},
                    {"role": "user", "content": test_prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            print(f"Chat test response: {response}")
            
            if (isinstance(response, dict) and 
                'choices' in response and 
                response['choices'] and 
                'message' in response['choices'][0] and
                'content' in response['choices'][0]['message']):
                
                text = response['choices'][0]['message']['content'].strip()
                return f"LLM отвечает: {text}"
            
            return "Пустой ответ от chat completion"
                
        except Exception as e:
            return f"Chat test failed: {e}"

_default_client: Optional[LocalLLMClient] = None

def get_llm_client(model_path: str = "./../Task4/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> LocalLLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LocalLLMClient(model_path)
    return _default_client