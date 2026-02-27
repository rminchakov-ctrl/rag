import json
import csv

from datetime import datetime

class GoldenTestRunner:
    def __init__(self, rag_bot):
        self.bot = rag_bot
        self.test_results = []
        self.golden_set = self.load_golden_set()
        
    def load_golden_set(self):
        """Золотой набор вопросов для тестирования"""
        return [
            # Вопросы на известные темы (бот ДОЛЖЕН ответить)
            {
                "question": "Кто такой Моллари?",
                "expected_answer": True,
                "category": "персонажи"
            },
            {
                "question": "Расскажи про Шеридана",
                "expected_answer": True,
                "category": "персонажи"
            },
            {
                "question": "Что такое Протомолекула?",
                "expected_answer": True,
                "category": "концепты"
            },
            {
                "question": "Опиши корабль Буцефал",
                "expected_answer": True,
                "category": "технологии"
            },
            {
                "question": "Что такое Ремень?",
                "expected_answer": True,
                "category": "локации"
            },
            {
                "question": "Где находится станция Громко?",
                "expected_answer": True,
                "category": "локации"
            },
            {
                "question": "Кто такие ОПА?",
                "expected_answer": True,
                "category": "организации"
            },            
            # Вопросы на удаленные темы (бот НЕ должен ответить)
            {
                "question": "Кто такая Иванова?",
                "expected_answer": True,
                "category": "удаленные_персонажи"
            },
            {
                "question": "Расскажи про Леньера",
                "expected_answer": True,
                "category": "удаленные_персонажи"
            },
            {
                "question": "Что такое двигатель Эпштейна",
                "expected_answer": False,
                "category": "удаленные_технологии"
            },
            {
                "question": "Кто такой Xarn Velgor?",
                "expected_answer": False,
                "category": "удаленные_сущности"
            },
            {
                "question": "Что такое Synth Flux?",
                "expected_answer": False, 
                "category": "удаленные_концепты"
            },
            {
                "question": "Опиши планету VoidCore",
                "expected_answer": False,
                "category": "удаленные_локации"
            },
            {
                "question": "Как работает двигатель на Dark Matter?",
                "expected_answer": False,
                "category": "удаленные_технологии"
            }
        ]
    
    def run_tests(self):
        """Запуск всех тестов"""
        print("Запуск тестирования золотого набора...")
        
        for i, test_case in enumerate(self.golden_set, 1):
            print(f"\n[{i}/{len(self.golden_set)}] Тест: {test_case['question']}")
            response = self.bot.process_query(test_case['question'])
            
            result = self.analyze_response(test_case, response)
            self.test_results.append(result)
            
            print(f"    Ожидалось: {'Ответ' if test_case['expected_answer'] else 'Нет ответа'}")
            print(f"    Получено: {'Ответ' if result['response_found'] else 'Нет ответа'}")
            print(f"    Статус: {'PASS' if result['is_correct'] else 'FAIL'}")
        
        # Сохраняем результаты
        self.save_results()
        return self.test_results
    
    def analyze_response(self, test_case, response):
        """Анализирует ответ бота"""
        # Простая эвристика: считаем, что ответ есть если он достаточно длинный
        # и не содержит фраз о отсутствии информации
        response_text = response.lower()
        has_answer = (
            len(response_text) > 50 and 
            "не найдено" not in response_text and
            "не знаю" not in response_text and
            "нет информации" not in response_text
        )
        
        # Проверяем, соответствует ли результат ожидаемому
        is_correct = has_answer == test_case['expected_answer']
        
        return {
            "timestamp": datetime.now().isoformat(),
            "question": test_case['question'],
            "category": test_case['category'],
            "expected_answer": test_case['expected_answer'],
            "response_text": response,
            "response_length": len(response),
            "response_found": has_answer,
            "is_correct": is_correct
        }
    
    def save_results(self):
        """Сохраняет результаты в JSON и CSV"""
        # JSON для детального анализа
        with open('golden_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        # CSV для табличного анализа
        with open('golden_test_results.csv', 'w', newline='', encoding='utf-8') as f:
            if self.test_results:
                writer = csv.DictWriter(f, fieldnames=self.test_results[0].keys())
                writer.writeheader()
                writer.writerows(self.test_results)
        
        print(f"\nРезультаты сохранены в golden_test_results.json и golden_test_results.csv")

def main():
    INDEX_SAVE_PATH = "./faiss_index"
    MODEL_PATH = "../Task4/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    from rag_bot import RAGBot
    bot = RAGBot(INDEX_SAVE_PATH, MODEL_PATH)

    tester = GoldenTestRunner(bot)
    results = tester.run_tests()
    
    # Статистика
    total = len(results)
    passed = sum(1 for r in results if r['is_correct'])
    coverage = passed / total * 100 if total > 0 else 0
    
    print(f"\n  ИТОГИ:")
    print(f"    Тестов пройдено: {passed}/{total}")
    print(f"    Покрытие: {coverage:.1f}%")
    
    # Анализ по категориям
    from collections import defaultdict
    category_stats = defaultdict(lambda: {'total': 0, 'passed': 0})
    
    for result in results:
        cat = result['category']
        category_stats[cat]['total'] += 1
        if result['is_correct']:
            category_stats[cat]['passed'] += 1
    
    print(f"\nПо категориям:")
    for category, stats in category_stats.items():
        cat_coverage = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"   {category}: {stats['passed']}/{stats['total']} ({cat_coverage:.1f}%)")

if __name__ == "__main__":
    main()