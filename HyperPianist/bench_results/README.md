# Benchmark Artifact Policy

This directory is reserved for curated benchmark artifacts that are ready to be published with reproducibility metadata.

Historical CSVs that lacked enough provenance for a public artifact were intentionally removed from the main branch during repository cleanup.

## What To Commit Here

Only commit benchmark CSVs that come with a short provenance record containing:

- machine or runner description
- execution date
- exact command or wrapper script invocation
- commit hash
- short description of the workload represented by the file

## Suggested Header For Curated Results

Use a sibling Markdown file or a short section in a directory README with fields such as:

- `File`
- `Commit`
- `Date`
- `Machine`
- `Command`
- `Meaning`

If you do not have this information, prefer not committing the result.
