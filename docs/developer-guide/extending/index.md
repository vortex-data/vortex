# Extending Vortex

:::{warning}
This section is under construction. For guidance on extending Vortex, please join the
[Vortex Slack channel](https://vortex.dev/slack)
or start a [GitHub Discussion](https://github.com/spiraldb/vortex/discussions).
:::

Vortex is designed to be extended with custom types, encodings, layouts, and compute functions.
The following topics are planned for this section:

- **Extension DTypes** -- defining custom logical types, serialization, session registration,
  and Arrow interoperability.
- **Writing an Encoding** -- implementing a custom array encoding with compression and
  decompression logic.
- **Writing a Layout** -- implementing the LayoutReader and LayoutWriter traits for custom
  on-disk data organizations.
- **Writing a Compute Function** -- the dispatch model, implementing kernels, vtable
  registration, and testing.

```{toctree}
---
hidden: true
---

extension-dtypes
writing-an-encoding
writing-a-layout
writing-a-compute-fn
```
