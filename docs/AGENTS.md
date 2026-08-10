# Documentation Guidance

Applies to files under `docs/`.

- Documentation should focus on behavior. Include implementation details only for complex behavior
  or private functions.
- Keep prose terse and accurate, and link to relevant external resources, project documentation,
  or API types when useful.
- Follow the [documentation development workflow](../CONTRIBUTING.md#documentation) in the
  contributor guide.
  Keep that contributor guide as the source of truth for Sphinx commands shared by humans and
  agents, and do not invoke `sphinx-build` directly.
- Do not run Rust checks for changes confined to RST, Markdown, or Sphinx configuration.
