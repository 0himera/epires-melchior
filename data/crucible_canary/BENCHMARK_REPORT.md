# Melchior Crucible — Frozen Holdout Benchmark & Canary Calibration Report

**Дата фиксации среза**: 29 сентября 2026 г.  
**Среда выполнения**: Remote GPU Server (`129.212.176.219`), AMD MI300X (192GB VRAM), 20 vCPU  
**Модели**:
* **System 2 (Генератор)**: `Qwen/Qwen3.8-27B-FP8` через vLLM (MTP-2, AITer/CK kernels)
* **System 1 (Арбитр)**: `AlexWortega/openjev` (`qwen3.5-4b-nli-v5`)

---

## 1. Сводные метрики калибровки (Canary Run)

| Метрика | Значение | Оценка |
|---|---:|---|
| **Всего задач в срезе** | 27 задач (54 скрипта-кандидата) | — |
| **Pass Rate кода от Qwen 27B** | **64.8%** (35 из 54) | В рабочем диапазоне (50–75%) |
| **Timeout Rate (лимит 14.0s)** | **0.0%** (0 таймаутов) | Отлично (14с достаточно для всех сред) |
| **Decisive Rate ($\Delta \ge 0.01$)** | **74.1%** (20 из 27 пар) | Высокое разнообразие гипотез |
| **Доля ничьих ($\Delta < 0.01$)** | **3.7%** (1 задача) | Модель избегает тривиальных дублей |
| **Доля сбоев обоих кандидатов** | **22.2%** (6 задач) | Служит негативными NLI-примерами |
| **OpenJev Baseline Accuracy** | **55.0%** (11 из 20 угадано) | **Чистая монетка (Random prior ~50%)** |

---

## 2. Зафиксированный контрольный тест (20 Decisive Pairs)

Датасет сохранён в: [`data/crucible_canary/holdout_validation_benchmark.jsonl`](./holdout_validation_benchmark.jsonl)

| № | Task ID | Тип задачи & Метрика | Фактический победитель | Эмпирическая маржа ($\Delta$) | Предсказание Jev v5 | Результат |
|---|---|---|:---:|---:|:---:|:---:|
| 1 | `crucible_task_001001` | Imbalanced Binary (`f1`) | **A** | 0.022 | A | ✓ Верно |
| 2 | `crucible_task_001002` | Non-linear Regression (`r2`) | **B** | 0.049 | A | ✗ Ошибка |
| 3 | `crucible_task_001004` | Tabular Benchmark (`accuracy`) | **A** | 0.044 | B | ✗ Ошибка |
| 4 | `crucible_task_001007` | Imbalanced Binary (`f1`) | **B** | 1.000 (нокаут) | B | ✓ Верно |
| 5 | `crucible_task_001008` | Non-linear Regression (`r2`) | **B** | 0.034 | B | ✓ Верно |
| 6 | `crucible_task_001009` | Multiclass (`accuracy`) | **A** | 1.000 (нокаут) | B | ✗ Ошибка |
| 7 | `crucible_task_001011` | Digits Image Feats (`accuracy`) | **B** | 1.000 (нокаут) | B | ✓ Верно |
| 8 | `crucible_task_001012` | High-dim Binary (`roc_auc`) | **A** | 0.083 | B | ✗ Ошибка |
| 9 | `crucible_task_001013` | Imbalanced Binary (`f1`) | **B** | 1.000 (нокаут) | B | ✓ Верно |
| 10 | `crucible_task_001014` | Non-linear Regression (`r2`) | **B** | 0.124 | A | ✗ Ошибка |
| 11 | `crucible_task_001016` | Tabular Benchmark (`accuracy`) | **A** | 1.000 (нокаут) | B | ✗ Ошибка |
| 12 | `crucible_task_001018` | High-dim Binary (`roc_auc`) | **A** | 0.092 | A | ✓ Верно |
| 13 | `crucible_task_001019` | Imbalanced Binary (`f1`) | **A** | 0.013 | A | ✓ Верно |
| 14 | `crucible_task_001020` | Non-linear Regression (`r2`) | **B** | 0.062 | A | ✗ Ошибка |
| 15 | `crucible_task_001021` | Multiclass (`accuracy`) | **A** | 1.000 (нокаут) | A | ✓ Верно |
| 16 | `crucible_task_001022` | Tabular Benchmark (`roc_auc`) | **A** | 0.108 | B | ✗ Ошибка |
| 17 | `crucible_task_001023` | Digits Image Feats (`accuracy`) | **B** | 1.000 (нокаут) | B | ✓ Верно |
| 18 | `crucible_task_001025` | Imbalanced Binary (`f1`) | **B** | 1.000 (нокаут) | B | ✓ Верно |
| 19 | `crucible_task_001026` | Non-linear Regression (`r2`) | **B** | 0.025 | A | ✗ Ошибка |
| 20 | `crucible_task_001027` | Multiclass (`accuracy`) | **A** | 0.122 | A | ✓ Верно |

**Итог Baseline OpenJev**: 11 верных решений из 20 (**55.0%**).

---

## 3. Как запустить утреннюю валидацию дообученной модели

После завершения ночной генерации и дообучения OpenJev:

```bash
# Запуск автоматической валидации против замороженного среза:
python3 scripts/evaluate_holdout.py --endpoint http://129.212.176.219:8080/v1/systemone
```

Если точность возрастёт с **55.0%** до **75–80%+**, гипотеза полностью подтверждена: создана первая в своём роде калиброванная эмпирическая система принятия решений System 1 для автономных ML-агентов.
