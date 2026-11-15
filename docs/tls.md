# TLS и диагностика подключения GigaChat

Приложение сохраняет системные доверенные сертификаты, добавляет публичные корни `certifi` и, только для native GigaChat, публичный Russian Trusted Root CA. Это устраняет зависимость от ручной установки корня GigaChat в Python на Mac. Проверка цепочки и имени сервера обязательна; системное хранилище и файлы `certifi` не изменяются.

## Происхождение публичного сертификата

- [Официальная инструкция GigaChat](https://developers.sber.ru/docs/ru/gigachat/certificates).
- [Ссылка на загрузку корня из официальной инструкции](https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt).
- Получен 15.11.2026 по HTTPS с проверкой TLS; включён в `src/citizen_agent/certs/russian_trusted_root_ca.pem` и в wheel/sdist.
- Subject/issuer: `C=RU, O=The Ministry of Digital Development and Communications, CN=Russian Trusted Root CA`.
- Срок действия: 01.03.2022 21:04:15 UTC — 27.02.2032 21:04:15 UTC.
- SHA-256 DER: `D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31`.

При обновлении файла нужно повторно получить его из официального источника, проверить subject, issuer, сроки и SHA-256; затем обновить ожидаемый отпечаток в тесте. Не скачивать сертификаты автоматически во время запуска и не отключать TLS при скачивании.

## Проверка на Mac

```bash
cd ~/Desktop/agent_systems
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
appeals doctor --live
```

Ключ уже находится в `.env`, повторный ввод не требуется. `doctor --live` печатает JSON и при неудаче завершается с кодом 2:

| Поле/тип | Значение и следующий шаг |
|---|---|
| `oauth_ok=true` | API выдал токен; ключ принят |
| `models_ok=true` | Получен список моделей |
| `configured_model_available=true` | Модель из конфига есть в списке |
| `live_stage` | Этап: `oauth`, `models` или `complete` |
| `live_error.type=tls` | Сертификат не прошёл проверку. Проверить дату компьютера; при корпоративном прокси указать его доверенный корень через `GIGACHAT_CA_BUNDLE` |
| `dns` / `network` / `proxy` / `timeout` | Проверить сеть, DNS, VPN, прокси и доступ к порту OAuth 9443 |
| `authentication` / HTTP 401 | Сервер отклонил авторизацию: проверить Authorization Key и scope |
| `forbidden` / HTTP 403 | Проверить права API, scope и баланс |
| `service_unavailable` / HTTP 5xx | Сервис или сетевой шлюз недоступен; это не подтверждение неверного ключа |
| `model_unavailable` | Выбрать `model` из `available_models` в `config/gigachat.toml` |

Значение ключа, токена, заголовков и содержимое ошибок сервера в диагностике не выводятся. Ошибка сертификата не повторяется: повторы не меняют доверие. Временные ошибки повторяются в пределах `max_attempts`.

Дополнительный доверенный PEM, например для корпоративного прокси, можно задать в локальном `.env` через `GIGACHAT_CA_BUNDLE=/абсолютный/путь/ca.pem`. Он добавляется к остальным корням. Альтернативный адаптер не получает корень GigaChat автоматически.
