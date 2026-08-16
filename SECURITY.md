# Security Policy

## Reporting a vulnerability

Do not open a public issue containing credentials, private portfolio data,
provider tokens, or vulnerability details. Use GitHub's private security
advisory reporting for this repository when it is available; otherwise contact
the repository owner privately through GitHub.

Please include a concise description, reproduction steps, likely impact, and
any suggested mitigation. Do not include real secrets or private financial
data.

## Scope

This research application does not execute trades or hold brokerage
credentials. Treat uploaded configs, cached market data, generated reports, and
deployment environment variables as potentially sensitive. Rotate any secret
that is accidentally committed and remove it from deployment settings.

Supported security fixes target the latest branch. No formal SLA is currently
offered.
