import json
import pandas as pd

def analyze_coverage():
    # Загружаем результаты тестов
    with open('golden_test_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    df = pd.DataFrame(results)
    
    # Анализ покрытия
    print("АНАЛИЗ ПОКРЫТИЯ БАЗЫ ЗНАНИЙ")
    print("=" * 50)
    
    # Общая статистика
    total_tests = len(df)
    passed_tests = len(df[df['is_correct'] == True])
    coverage = passed_tests / total_tests * 100
    
    print(f"Общее покрытие: {passed_tests}/{total_tests} ({coverage:.1f}%)")
    
    # Анализ по категориям
    print("\nПО КАТЕГОРИЯМ:")
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        cat_total = len(cat_data)
        cat_passed = len(cat_data[cat_data['is_correct'] == True])
        cat_coverage = cat_passed / cat_total * 100 if cat_total > 0 else 0
        
        print(f"  {category}: {cat_passed}/{cat_total} ({cat_coverage:.1f}%)")
    
    # Выявление слепых зон
    failed_tests = df[df['is_correct'] == False]
    if not failed_tests.empty:
        print("\nСЛЕПЫЕ ЗОНЫ (требуют дополнения базы):")
        for _, row in failed_tests.iterrows():
            if row['expected_answer']:  # Должны были получить ответ, но не получили
                print(f"  - {row['question']}")    

if __name__ == "__main__":
    analyze_coverage()