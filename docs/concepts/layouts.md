# Vortex Layouts

### Owned vs Viewed

As with other possibly large recursive data structures in Vortex, layouts can be either _owned_ or _viewed_.
Owned layouts are heap-allocated, while viewed layouts are lazily unwrapped from an underlying FlatBuffer
representation.
This allows Vortex to efficiently load and work with very wide schemas without needing to deserialize the full layout
in memory.
