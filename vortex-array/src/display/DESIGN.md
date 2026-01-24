# Tree Display Design

This document describes the design of the unified tree display system for Vortex arrays and layouts.

## Goals

1. **Unified rendering**: Support both text (tree view) and JSON output from the same data model
2. **Lazy evaluation**: Walk the tree during rendering, not upfront
3. **Zero-copy where possible**: Borrow children when they're already stored, box when computed
4. **Pluggable attributes**: Composable attribute extraction without modifying core traits

## Core Types

### `MaybeOwned<'a, T: ?Sized>`

A `Cow`-like type for trait objects that can be either borrowed or owned:

```rust
pub enum MaybeOwned<'a, T: ?Sized> {
    Borrowed(&'a T),
    Owned(Box<T>),
}
```

This solves the lifetime problem where some types (like `DisplayTreeNode`) store their children and can return references, while others (like `Layout`) compute children on demand and need to return owned values.

### `TreeDisplayable` Trait

The core trait for lazy tree rendering:

```rust
pub trait TreeDisplayable {
    fn name(&self) -> Cow<'_, str>;
    fn attrs(&self) -> Vec<Attr>;
    fn nested_attrs(&self) -> Vec<Attr> { vec![] }
    fn children(&self) -> Vec<(Cow<'_, str>, MaybeOwned<'_, dyn TreeDisplayable>)> { vec![] }
    fn to_tree_node(&self) -> DisplayTreeNode { /* walks tree eagerly */ }
}
```

### `DisplayTreeNode`

An eager/owned tree representation suitable for JSON serialization:

```rust
pub struct DisplayTreeNode {
    pub name: String,
    pub attrs: Vec<Attr>,
    pub nested_attrs: Vec<Attr>,
    pub children: IndexMap<String, DisplayTreeNode>,
}
```

## Pluggable Attribute Providers

To make attribute extraction composable without modifying the core `TreeDisplayable` trait, we use a provider pattern.

### `AttrProvider<T>` Trait

```rust
/// Extracts attributes from a value of type T.
pub trait AttrProvider<T>: Send + Sync {
    /// Extract inline attributes (shown on same line as node name).
    fn attrs(&self, value: &T) -> Vec<Attr>;

    /// Extract nested attributes (shown on separate indented lines).
    fn nested_attrs(&self, _value: &T) -> Vec<Attr> {
        vec![]
    }
}
```

### `AttrProviders<T>` Collection

```rust
/// A collection of providers to apply when building tree nodes.
pub struct AttrProviders<T> {
    providers: Vec<Box<dyn AttrProvider<T>>>,
}

impl<T> AttrProviders<T> {
    pub fn new() -> Self {
        Self { providers: vec![] }
    }

    pub fn with<P: AttrProvider<T> + 'static>(mut self, provider: P) -> Self {
        self.providers.push(Box::new(provider));
        self
    }

    pub fn extract_attrs(&self, value: &T) -> Vec<Attr> {
        self.providers.iter().flat_map(|p| p.attrs(value)).collect()
    }

    pub fn extract_nested_attrs(&self, value: &T) -> Vec<Attr> {
        self.providers.iter().flat_map(|p| p.nested_attrs(value)).collect()
    }
}
```

### Example Providers for Layout

```rust
pub struct DTypeProvider;
impl AttrProvider<LayoutRef> for DTypeProvider {
    fn attrs(&self, layout: &LayoutRef) -> Vec<Attr> {
        vec![Attr::new("dtype", layout.dtype().to_string())]
    }
}

pub struct RowCountProvider;
impl AttrProvider<LayoutRef> for RowCountProvider {
    fn attrs(&self, layout: &LayoutRef) -> Vec<Attr> {
        vec![Attr::new("rows", layout.row_count())]
    }
}

pub struct ChildCountProvider;
impl AttrProvider<LayoutRef> for ChildCountProvider {
    fn attrs(&self, layout: &LayoutRef) -> Vec<Attr> {
        let n = layout.nchildren();
        if n > 0 {
            vec![Attr::new("children", n as u64)]
        } else {
            vec![]
        }
    }
}

pub struct SegmentIdsProvider;
impl AttrProvider<LayoutRef> for SegmentIdsProvider {
    fn nested_attrs(&self, layout: &LayoutRef) -> Vec<Attr> {
        let ids = layout.segment_ids();
        if ids.is_empty() {
            vec![]
        } else {
            let list: Vec<AttrValue> = ids.iter()
                .map(|s| AttrValue::UInt(**s as u64))
                .collect();
            vec![Attr::new("segments", AttrValue::List(list))]
        }
    }
}

/// Provider that needs external data (segment sizes from IO).
pub struct BufferSizesProvider {
    segment_sizes: HashMap<SegmentId, Vec<usize>>,
}

impl AttrProvider<LayoutRef> for BufferSizesProvider {
    fn nested_attrs(&self, layout: &LayoutRef) -> Vec<Attr> {
        // Look up buffer sizes for this layout's segments
        // ...
    }
}
```

### Preset Combinations

```rust
impl AttrProviders<LayoutRef> {
    /// All available attributes.
    pub fn verbose() -> Self {
        Self::new()
            .with(DTypeProvider)
            .with(RowCountProvider)
            .with(ChildCountProvider)
            .with(MetadataBytesProvider)
            .with(SegmentIdsProvider)
    }

    /// Minimal attributes for concise output.
    pub fn concise() -> Self {
        Self::new()
            .with(DTypeProvider)
            .with(ChildCountProvider)
    }

    /// Just the encoding name, no attributes.
    pub fn minimal() -> Self {
        Self::new()
    }
}
```

### Usage

```rust
// Build a tree display with custom providers
let providers = AttrProviders::<LayoutRef>::new()
    .with(DTypeProvider)
    .with(RowCountProvider)
    .with(BufferSizesProvider::new(segment_sizes));

let display = LayoutTreeDisplay::new(layout, providers);

// Text output (lazy)
println!("{}", display);

// JSON output (eager)
let json = serde_json::to_string(&display.to_tree_node())?;
```

## Lazy JSON with JSONPath Queries

For advanced use cases, we can wrap `TreeDisplayable` in a lazy JSON type that implements the `Queryable` trait from `jsonpath-rust`:

```rust
pub enum LazyJson<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Cow<'a, str>),
    Array(Vec<LazyJson<'a>>),
    Object(IndexMap<Cow<'a, str>, LazyJson<'a>>),
    /// Lazy node - materializes on access
    Node(LazyNode<'a>),
}

pub struct LazyNode<'a> {
    source: &'a dyn TreeDisplayable,
    cache: OnceCell<IndexMap<Cow<'a, str>, LazyJson<'a>>>,
}
```

This enables JSONPath queries over arrays and layouts without eagerly building the entire tree:

```rust
let lazy = LazyJson::from_tree(&layout);

// Only materializes nodes that the query touches
let results = jsonpath_rust::query(&lazy, "$.children.*.name")?;
```

## Summary

| Component | Purpose |
|-----------|---------|
| `MaybeOwned` | Cow-like type for trait objects (borrow or own) |
| `TreeDisplayable` | Core trait for lazy tree rendering |
| `DisplayTreeNode` | Eager/owned tree for JSON serialization |
| `AttrProvider<T>` | Pluggable attribute extraction |
| `AttrProviders<T>` | Composable collection of providers |
| `LazyJson` | Lazy JSON for JSONPath queries (future) |
