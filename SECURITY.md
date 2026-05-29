# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `0.x` (pre-release) | Yes — rolling fixes on `main` |

Once `1.0.0` is released, the two most recent minor releases receive security
patches.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security issues by e-mail to:

```
security@qwentrix.com
```

Include as much detail as possible:

- Package version and Python version
- A minimal reproduction case (sanitised of any real credentials or PII)
- Observed vs expected behaviour
- Your assessment of severity

We aim to acknowledge reports within **2 business days** and to publish a
patch or advisory within **14 calendar days** of a confirmed vulnerability.

If you require encrypted communication, request our PGP key by replying to
the acknowledgement e-mail.

## Disclosure Policy

We follow coordinated disclosure. We will:

1. Confirm receipt and assess severity.
2. Work on a fix in a private fork.
3. Notify you before public disclosure so you can review the fix.
4. Credit you in the advisory (unless you prefer to remain anonymous).

## Scope

This policy covers the `sigil-py` Python package and its published PyPI
artifact. For vulnerabilities in the server-side `sigil-core` service, please
use the same e-mail address; those are triaged by the Qwentrix platform
security team.
