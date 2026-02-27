import os
import sys
import logging
from typing import Optional, Dict, Any

class LocalLLMClient:
    def __init__(self, model_path: str):
        self.model_path = os.path.abspath(model_path)
        self.llm = None
        self.is_available = False
        self.context_size = ChildProcessError
        self.logger = self._setup_logging()
        self._initialize_llm()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('LocalLLMClient')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_llm(self):
        try:
            from llama_cpp import Llama
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Модель {self.model_path} не найдена")
            
            self.logger.info(f"Загрузка модели: {os.path.basename(self.model_path)}")
            
            n_threads = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
            
            # Подавляем вывод при инициализации
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=n_threads,
                    n_gpu_layers=0,
                    verbose=False,
                    n_batch=512,
                    seed=42
                )
                
                sys.stdout = old_stdout
            
            self.is_available = True
            self.logger.info(f"Модель загружена: {os.path.basename(self.model_path)}")
            
        except ImportError:
            self.logger.error("Установите llama-cpp-python: pip install llama-cpp-python")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации модели: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 400, temperature: float = 0.1, top_p: float = 0.9) -> str:
        """
        Генерация текста
        
        Args:
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации (0.0-2.0)
            top_p: Top-p sampling (0.0-1.0)
        """
        if not self.is_available or self.llm is None:
            return "Модель LLM не доступна"
        
        try:
            # Подавляем вывод во время генерации
            with open(os.devnull, 'w') as fnull:
                old_stdout = sys.stdout
                sys.stdout = fnull
                
                response = self.llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system", 
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["\n\n", "###", "Информация не найдена", "Контекст:", "Вопрос:"],
                    stream=False
                )
                
                sys.stdout = old_stdout
            
            return self._extract_response_text(response)
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка генерации ответа: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        return """
Ты — помощник, который отвечает на вопросы на основе предоставленной информации.
Всегда отвечай на русском языке, четко и по делу.

КРИТИЧЕСКИЕ ПРАВИЛА:
1. Отвечай только на основе предоставленного контекста
2. Не придумывай информацию
3. Если информации нет - так и скажи
4. Будь кратким и информативным
5. Завершай ответ, не обрывай на полуслове"""
    
    def _extract_response_text(self, response: Dict) -> str:
        """Извлекает текст из ответа llama.cpp"""
        try:
            if (isinstance(response, dict) and 
                'choices' in response and 
                response['choices'] and 
                'message' in response['choices'][0] and
                'content' in response['choices'][0]['message']):
                
                text = response['choices'][0]['message']['content'].strip()
                # Убираем возможные артефакты начала ответа
                if text.startswith('ОТВЕТ:'):
                    text = text[6:].strip()
                return text
            
            return "Пустой ответ от модели"
                
        except Exception as e:
            self.logger.error(f"Ошибка обработки ответа: {e}")
            return "Ошибка обработки ответа модели"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженной модели"""
        if not self.is_available:
            return {
                "status": "not_available", 
                "message": "Модель не загружена",
                "model_path": self.model_path
            }
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "model_name": os.path.basename(self.model_path),
            "context_size": self.context_size,
            "available": self.is_available
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Тестирование подключения к модели"""
        if not self.is_available:
            return {"status": "error", "message": "Модель не загружена"}
        
        try:
            test_response = self.generate(
                "Привет! Как тебя зовут?",
                max_tokens=50,
                temperature=0.1
            )
            
            return {
                "status": "success",
                "response": test_response,
                "model": os.path.basename(self.model_path)
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Глобальный клиент с ленивой инициализацией
_default_client: Optional[LocalLLMClient] = None

def get_llm_client(model_path: str = "./../Task4/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf") -> LocalLLMClient:
    """Получение или создание клиента LLM"""
    global _default_client
    
    if _default_client is None:
        _default_client = LocalLLMClient(model_path)
    elif not _default_client.is_available:
        # Переинициализируем если клиент недоступен
        _default_client = LocalLLMClient(model_path)
    
    return _default_client