# Melchior — Dual-Brain ML Auto-Research Agent

> **Melchior** — автономный ИИ-агент, который самостоятельно строит решения машинного обучения под различные задачи и улучшает их на основе результатов экспериментов.

В отличие от существующих систем (WECO AIDE, Epires, AIRA-dojo), Melchior использует **двухуровневую архитектуру (Dual-Brain)**:
1. **System 1 (Jev / OpenJev)** — ультрабыстрая модель решений (70–500ms, калиброванные вероятности по шкале RLCD):
   - Выбор стратегии (`explore` / `exploit` / `ablate` / `ensemble`)
   - Фильтрация техник перед генерацией
   - Ранжирование кандидатов по пилотным запускам
   - Early-stop детекция плато кривых обучения
   - Классификация рантайм ошибок и авто-фикс
2. **System 2 (LLM)** — генеративный интеллект:
   - Формулировка гипотез и написание чистого, воспроизводимого кода `solution.py`
   - Самокалибровка: предсказание ожидаемой метрики до запуска

---

## Быстрый старт

### 1. Установка

```bash
git clone https://github.com/0himera/epires-melchior.git
cd epires-melchior

# Создать venv и установить
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Запуск автономного исследования

```bash
# Локально:
melchior run examples/iris_classification.yaml --output artifacts/

# Или через Docker (оптимизированный multi-stage образ):
docker build -t melchior:latest .
docker run --rm -v $(pwd)/artifacts:/workspace/artifacts melchior:latest run examples/iris_classification.yaml

# Или через Docker Compose:
docker compose run melchior
```

### 3. Артефакты работы
По итогам работы агент создаёт:
- `artifacts/best_solution.py` — автономный воспроизводимый скрипт лучшей модели
- `artifacts/journal.jsonl` — полный аудит всех попыток с трекингом калибровки
- `artifacts/report.md` — итоговая исследовательская монография с таблицей абляций

---

## Архитектура: 9 точек решений Jev

```
   Task.yaml
       │
       ▼
 ┌───────────┐
 │ STRATEGIST│ ──① Jev.Choice: Стратегия (explore/exploit/ablate/ensemble)
 │           │ ──② Jev.Noul: Фильтрация техник по домену и истории
 │           │ ──③ LLM (System 2): Генерация гипотезы + кода solution.py
 └─────┬─────┘
       │ (1–3 кандидата)
       ▼
 ┌───────────┐
 │  FILTER   │ ──④ AST синтаксический анализ и проверка безопасности
 │           │ ──⑤ Пилотный прогон (1 эпоха / 5% данных)
 │           │ ──⑥ Jev.Score: Оценка качества пилотной кривой
 └─────┬─────┘
       │ (Top-1 кандидат)
       ▼
 ┌───────────┐
 │ EXECUTOR  │ ── Полноценное обучение с dense feedback
 │           │ ──⑦ Jev.Noul: Ранняя остановка при детекции плато
 │           │ ──⑧ Jev.Choice: Классификация ошибки (OOM, shape, data)
 └─────┬─────┘
       │
       ▼
 ┌───────────┐
 │  JOURNAL  │ ── Логирование в JSONL + калибровка (predicted vs actual)
 │           │ ──⑨ Jev.Noul: Оценка бюджета ("стоит ли продолжать?")
 └───────────┘
```

---

## Запуск тестов

```bash
pytest tests/ -v
```
