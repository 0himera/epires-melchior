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

### 3. Генерация датасета решений (Melchior Crucible)

Для запуска ночного конвейера эмпирического Self-Play на сервере (20 vCPU + локальный vLLM с Qwen 27B):

```bash
# Запуск 20 асинхронных воркеров с подключением к vLLM:
melchior crucible \
  --concurrency 20 \
  --url http://localhost:8000/v1 \
  --model qwen \
  --output data/crucible_dataset

# Результат:
# data/crucible_dataset/openjev_ml_nli.jsonl  (OpenJev 3-way NLI: 0/1/2)
# data/crucible_dataset/melchior_dpo_pairs.jsonl (DPO pairs: chosen/rejected)
```

Crucible проверяет наличие кода и гипотезы у обоих кандидатов. При ошибке API,
неполном или обрезанном ответе выполняется до двух попыток на backend.
Режимы `api` и `opencode` сообщают об ошибке после исчерпания попыток;
`auto` может перейти с API на доступный OpenCode. Синтетические решения
генерируются только при явном `--mode mock`. Лимит `--pairs N` ограничивает
число попыток пар, включая неудачные; ошибки пишутся в `crucible_errors.jsonl`.

Для контрольного прогона:

```bash
melchior canary --tasks 50 --concurrency 8 --output data/canary_run_01
```

Canary продолжает работу после ошибки отдельной задачи и сохраняет результаты
сразу после её завершения:

- `canary_evaluations.jsonl` — все оценённые пары с кодом, гипотезами и результатами;
- `canary_errors.jsonl` — ошибки задач с seed и описанием причины;
- `holdout_validation_50.jsonl` — пары с определённым победителем;
- `canary_summary.json` — текущая сводка и статус прогона.

Новый запуск Canary перезаписывает эти файлы: используйте отдельный каталог
для каждого прогона. При отмене завершённые записи сохраняются, а запущенные
дочерние процессы завершаются.

### 4. Артефакты работы
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
uv sync --extra dev
uv run pytest tests/ -v
```

Тесты используют HTTP-заглушки и реальные локальные subprocess, без обращения
к сервисам моделей. Регрессии отмены проверяются в отдельном интерпретаторе
с внешним таймаутом, чтобы обнаруживать зависание самого `asyncio.run()`.
