# Агент обработки обращений граждан

Учебный CLI-агент для сотрудника, разбирающего обращения по вопросам ФНС. Определяет тему и приоритет, находит справочные материалы и готовит черновик с источниками. Передача человеку создаёт локальную задачу после подтверждения. Данные синтетические, официальные ответы не отправляются.

## Установка

Нужны Python 3.11–3.13 и Git.

```bash
git clone https://github.com/alexey077/agent_systems.git
cd agent_systems
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
cp .env.example .env
```

В `.env` укажите Authorization Key GigaChat в `GIGACHAT_CREDENTIALS`. Файл исключён из Git. В Windows PowerShell окружение активируется командой `.venv\Scripts\Activate.ps1`.

## Запуск

```bash
appeals doctor --live
appeals demo
appeals demo --fault
appeals run --text 'Не получил налоговое уведомление. Где его найти?' --trace
appeals queue
```

`demo` показывает обычную обработку, `demo --fault` — таймаут поиска и передачу человеку. Для создания задачи введите `ПОДТВЕРЖДАЮ`. Результаты и трассы сохраняются в `runtime/`.

## Инструменты и конфигурация

| Инструмент | Назначение |
|---|---|
| `search_knowledge_base` | Поиск публичных материалов ФНС |
| `get_routing_rules` | Учебная очередь и правила приоритета |
| `escalate_to_operator` | Задача ручного разбора с подтверждением |

Провайдер и модель задаются в `config/*.toml`. По умолчанию используется GigaChat. Альтернативный конфиг выбирается без изменения кода: `appeals --config config/ollama.toml demo`. Для него нужен запущенный совместимый сервер и `COMPATIBLE_API_KEY`.

## Проверки

```bash
ruff check .
ruff format --check .
pytest -q
appeals --config config/offline.toml evaluate --repeats 3 --output runtime/offline/report.json
python -m build --no-isolation
```

Offline-прогон проверяет выполнение инструментов без LLM. Реальные результаты и ограничения описаны в [оценке качества](docs/evaluation.md).

[Спецификация](docs/specification.md) · [Контракты](docs/tool_contracts.md) · [Архитектура](docs/architecture.md) · [Демонстрация](docs/demo.md) · [Источники](docs/sources.md) · [TLS и подключение](docs/tls.md)
