# Contributing to sigil-py

Thank you for your interest in contributing to the Micelium Sigil Python SDK.

---

## Contributor License Agreement (CLA)

All contributors must sign the Qwentrix CLA before their first pull request
can be merged. The CLA bot ([cla-assistant.io](https://cla-assistant.io)) will
prompt you automatically when you open a PR. If you are contributing on behalf
of an employer, a Corporate CLA is also required — contact
[legal@qwentrix.com](mailto:legal@qwentrix.com).

---

## Development Setup

```bash
git clone https://github.com/Qwentrix/sigil-py.git
cd sigil-py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Code Style

We use **Black** for formatting and **Ruff** for linting:

```bash
# Format
black sigil/ tests/

# Lint (auto-fix)
ruff check --fix sigil/ tests/
```

CI will reject PRs that fail either check. Configure your editor to run
`black` on save and `ruff` as a linter for the best experience.

---

## Type Checking

We enforce **mypy strict** mode:

```bash
mypy sigil/
```

All public functions and methods must have full type annotations. `Any` is
allowed only in narrow SDK boundary code that genuinely cannot be typed, and
must be accompanied by an inline comment explaining why.

---

## Testing

We use **pytest**. The test suite requires a local `sigil-core` fixture for
contract tests; see `tests/contract/README.md` for setup instructions.

```bash
# Run all tests (unit + integration stubs)
pytest

# Run only unit tests (no sigil-core required)
pytest tests/unit/

# Run with coverage report
pytest --cov=sigil --cov-report=html
```

Coverage threshold is enforced in `pyproject.toml` (80% for unit, 85% for
SDK-critical paths). PRs that drop coverage below threshold will be rejected
by CI.

---

## Pull Request Checklist

- [ ] `black` and `ruff` pass with zero warnings
- [ ] `mypy --strict sigil/` passes with zero errors
- [ ] `pytest` passes with coverage >= threshold
- [ ] New public API has type annotations and a one-line docstring
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] CLA signed (bot will check automatically)

---

## Branching Model

- `main` — latest stable; protected; CI must pass + 1 approving review
- `dev/*` — feature branches; open PRs against `main`
- `fix/*` — bug-fix branches

---

## Reporting Bugs

For security vulnerabilities, see [SECURITY.md](SECURITY.md). For all other
bugs, open a GitHub issue with a minimal reproduction case.

---

## Questions

For general questions, open a GitHub Discussion or e-mail
[sigil-sdk@qwentrix.com](mailto:sigil-sdk@qwentrix.com).
