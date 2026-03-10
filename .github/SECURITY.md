# Security Policy

## Security Scope

This repository is a research artifact and benchmark prototype. It is not marketed as an audited production cryptography implementation.

That means:

- correctness bugs are still important
- security issues are still worth reporting privately
- production-grade hardening, side-channel review, and formal audits are out of scope for the claims made by this repository

## Supported Versions

Only the current `main` branch is considered maintained for security-related fixes.

## Reporting A Vulnerability

Please do not open a public GitHub issue for suspected vulnerabilities.

Preferred reporting path:

1. Use GitHub private vulnerability reporting if it is enabled for this repository.
2. If that is not available, email `chenxinzhe132@126.com` with:
   - affected path or component
   - reproduction steps
   - impact assessment
   - any proposed mitigation

I will try to acknowledge reports promptly, but no SLA is promised for this research artifact.

## What To Expect

- triage for reproducible issues in first-party code
- clarification if the report only affects bundled third-party or baseline snapshot code
- a fix, documentation update, or explicit non-fix decision depending on scope and severity
