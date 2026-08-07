# Documentation Guidance

Applies to files under `docs/`.

Run the commands below from the repository root.

- Documentation should focus on behavior. Include implementation details only for complex behavior
  or private functions.
- Keep prose terse and accurate, and link to relevant external resources, project documentation,
  or API types when useful.
- Use `docs/Makefile` as the canonical interface to Sphinx. Do not invoke `sphinx-build`
  directly.
- Before handing off a documentation change, run the canonical clean build and doctest flow:

  ```bash
  uv run --all-packages make -C docs check
  ```

- The `check` target cleans the Sphinx output, builds HTML, and runs doctests. Use focused Make
  targets such as `html`, `doctest`, or `serve` while iterating; `make -C docs help` lists the
  available targets.
- Do not run Rust checks for changes confined to RST, Markdown, or Sphinx configuration.
