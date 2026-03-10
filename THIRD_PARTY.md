# Third-Party Provenance

This repository includes first-party code plus bundled external material needed for a reproducible public research artifact.

## Bundled Components

| Path | Role | Upstream source | License | Local modifications |
| --- | --- | --- | --- | --- |
| `third_party/algebra/` | Vendored arkworks algebra crates used by the main workspace | <https://github.com/arkworks-rs/algebra> | MIT OR Apache-2.0 | Yes. Kept as an in-tree vendored snapshot and patched for this workspace's dependency graph. |
| `third_party/curves/` | Vendored arkworks curve crates used by the main workspace | <https://github.com/arkworks-rs/curves> | MIT OR Apache-2.0 | Yes. Kept as an in-tree vendored snapshot and patched for this workspace's dependency graph. |

## Licensing Notes

- First-party code at the repository root and in the main workspace crates is released under the MIT license in [LICENSE](LICENSE).
- Vendored code keeps its upstream notices and license files in place.
- When there is any conflict between the root MIT notice and a vendored subdirectory's own license files, the more specific subdirectory license applies to that subdirectory.

## What Counts As First-Party Here

For this public release, the following directories are treated as first-party artifact code:

- `arithmetic/`
- `deNetwork/`
- `deSnark/`
- `hyperplonk/`
- `scripts/`
- `subroutines/`
- `transcript/`
- `util/`
- repository-level docs and GitHub metadata

Everything under `third_party/` should be read with the provenance table above in mind.
