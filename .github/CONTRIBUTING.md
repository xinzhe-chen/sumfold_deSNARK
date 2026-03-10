# Contributing

Thanks for considering changes to this repository.

This project is maintained as a public research artifact first. Changes should preserve reproducibility, keep the benchmark surface understandable, and avoid widening the repository's claims beyond that scope.

## Before Opening A Pull Request

Run the repository checks locally:

```bash
make release-check
```

That command is the expected baseline for contributions touching first-party code or public-facing documentation.

## Contribution Priorities

- fix correctness bugs in first-party code
- improve benchmark reproducibility
- tighten documentation around artifact setup and limitations
- keep public interfaces and CSV schemas stable unless there is a strong reason to change them

## Out Of Scope For Routine PRs

- claiming production readiness or audited security properties
- large benchmark-result dumps without provenance
- replacing vendored third-party trees without documenting the source and license impact
- broad vendored-code refactors that do not help correctness, reproducibility, or maintainability

## Benchmark Artifact Policy

Do not commit generated logs or benchmark CSVs unless they are intentionally curated for public release. Curated benchmark files must come with:

- machine or runner description
- execution date
- exact command or script used
- commit hash
- short interpretation of what the file represents

The repository intentionally prefers no benchmark artifact over a benchmark artifact with incomplete provenance.

## Pull Request Notes

- keep changes focused
- document behavioral or documentation changes in the PR description
- call out any changes to public scripts, CLI usage, or CSV formats
- mention if a change touches vendored or baseline code

## Reporting Security Issues

Please follow [SECURITY.md](SECURITY.md) instead of opening a public issue for a suspected vulnerability.
