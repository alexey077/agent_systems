# Working rules for this repository

This is homework 1 of «Практическое применение агентских систем».

- Before every change, consult `docs/requirements.md` and the relevant tool card in `docs/tool_contracts.md`.
- Keep the requirement-to-implementation-to-test mapping current. Do not mark live checks as passed without a real provider response.
- Preserve exactly three domain tools. Model/provider adapters are infrastructure, not extra tools.
- The LLM chooses its next tool. Application code validates arguments, permissions, confirmation, evidence, limits and results.
- Request identity, organization, operator identity, confirmation and idempotency keys come from trusted application state, never model arguments.
- Never commit secrets, `.env`, private citizen data, local databases or unredacted traces. TLS verification remains enabled.
- Use only synthetic appeals and attributed public sources. Routing queues are educational conventions, not claims about actual FNS structure.
- Distinguish offline scripted runs, HTTP-contract tests and real LLM evaluation. Report actual sample sizes and failed runs.
- Test ordinary behavior AND validation, permission, cancellation, timeout, malformed output, loop and duplicate-action failures.
- Preserve native GigaChat `functions_state_id` and translate provider-specific messages at the adapter boundary.
- Before delivery: run lint, unit/integration tests, offline scenario evaluation, package build, and inspect the Git diff. Real-model evaluation and the submission video remain explicit gates if credentials/recording are unavailable.
