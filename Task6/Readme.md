# Запуск и тестирование

## Установка зависимостей
``` bash
pip3 install -r requirements.txt
```
## Запуск и тестирование
``` bash
python3 update_index.py
```

# Настройка периодического запуска (cron)
## Ежедневный запуск в 6:00 утра
``` bash
0 6 * * * cd /Users/rminchakov/go/src/github.com/rag/Task5/update_index.py && python update_index.py >> logs/cron.log 2>&1
```
## Для тестирования - каждые 30 минут

``` bash
*/30 * * * * cd /Users/rminchakov/go/src/github.com/rag/Task5/update_index.py && python update_index.py >> logs/cron.log 2>&1
```