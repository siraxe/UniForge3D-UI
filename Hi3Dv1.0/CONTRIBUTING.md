# Contributing

Contributions to Hi3D v1.0 are welcome. Use repository issues for bug reports
and feature requests, and pull requests for proposed changes.

## Developer Certificate of Origin

Every commit must be signed off. Append a `Signed-off-by` line that names you
and an email you control:

```text
Signed-off-by: Your Name <you@example.com>
```

`git commit -s` writes that line. By signing off you certify the
[Developer Certificate of Origin](https://developercertificate.org/), version
1.1.

## Before you send a change

- Run `python3 -m ruff check .`. The rule set is pinned in `pyproject.toml`.
- Match the docstring house style already in the tree: one-line module
  summaries, Google-style function docstrings (`Args` / `Returns` / `Raises`),
  PEP 257 attribute docstrings on fields and on constants that need a rationale,
  double backticks for code references, and an imperative first sentence.

For changes to inference, include the model revision, environment, inputs,
resolution, and validation results. Follow [the validation guide](docs/reproduction.md)
for mesh checks and geometric comparisons. No automated evaluation harness is
included in this repository.

This contribution guide is adapted from the SPARC3D contribution guidelines for
Hi3D v1.0.
