# Writing a Layout

A Vortex layout plugin describes serialized layout metadata and how that layout expands into the
scan runtime. Layout plugins do not implement a separate reader trait. Instead, they implement the
layout vtable, deserialize layout-specific data into `Layout<V>`, and return a `ScanPlan` for
runtime reads.

## Layout Vtable

The layout vtable lives in `vortex_layout::layout_v2`. Its shape follows the same plugin pattern as
the other Vortex vtables:

- `Layout<V>` is the typed layout handle.
- `LayoutRef` is the type-erased layout handle.
- `LayoutParts<V>` constructs typed layouts from common fields plus `V::LayoutData`.
- `DynLayout` is private erased dispatch plumbing.
- `LayoutVTablePlugin` is the registry object used for ID-based deserialization.

Common fields are hoisted out of the plugin-specific data. The vtable receives dtype, row count,
segment IDs, lazy child access, and layout metadata during deserialization, but returns only the
layout-specific `LayoutData`.

```rust
use vortex_layout::layout_v2;
use vortex_layout::{LayoutChildType, LayoutId};
use vortex_layout::scan::plan::ScanPlanRef;
use vortex_layout::scan::plan::request::ScanRequest;
use vortex_session::VortexSession;

#[derive(Clone, Debug)]
pub struct MyLayout;

#[derive(Clone, Debug)]
pub struct MyLayoutData {
    // layout-specific metadata
}

impl layout_v2::VTable for MyLayout {
    type LayoutData = MyLayoutData;

    fn id(&self) -> LayoutId {
        LayoutId::new("example.my_layout")
    }

    fn deserialize(
        &self,
        args: &layout_v2::LayoutDeserializeArgs<'_>,
    ) -> vortex_error::VortexResult<Self::LayoutData> {
        // Parse args.metadata and validate args.segment_ids / args.children.
        Ok(MyLayoutData {})
    }

    fn child_dtype(
        layout: layout_v2::Layout<Self>,
        idx: usize,
    ) -> vortex_error::VortexResult<vortex_array::dtype::DType> {
        // Return the dtype expected for child `idx`.
        Ok(layout.dtype().clone())
    }

    fn child_type(
        _layout: layout_v2::Layout<Self>,
        idx: usize,
    ) -> vortex_error::VortexResult<LayoutChildType> {
        Ok(LayoutChildType::Transparent(format!("child-{idx}").into()))
    }

    fn new_scan_plan(
        layout: layout_v2::Layout<Self>,
        req: &mut ScanRequest,
        session: &VortexSession,
    ) -> vortex_error::VortexResult<ScanPlanRef> {
        // Expand the layout into a runtime ScanPlan.
        todo!()
    }
}
```

## Deserialization

`LayoutDeserializeArgs` contains the common serialized fields:

- `dtype`: the logical dtype of this layout;
- `row_count`: the number of rows in this layout's row domain;
- `metadata`: plugin-specific metadata bytes;
- `segment_ids`: logical segments referenced directly by this layout;
- `children`: lazy child access;
- `array_ctx`: the array read context captured from the file footer.

Use `deserialize` to validate invariants that are local to the layout. For example, a flat layout
requires exactly one segment ID, and a chunked layout verifies that child row counts add up to the
parent row count.

Do not eagerly deserialize children unless the layout metadata itself requires it. Child access is
intentionally lazy so projection and predicate pushdown can avoid unrelated branches of a wide
layout tree.

## Child Contracts

`child_dtype` and `child_type` define the contract between a parent layout and its children. The
scan path calls `layout.child(idx)`, which asks the parent for the expected dtype and then
materializes that child from the footer.

Use `LayoutChildType` to describe how child rows relate to parent rows:

- `Field(name)` for struct fields;
- `Chunk((idx, offset))` for row-range chunks;
- `Transparent(name)` for wrappers whose data child shares the parent row domain;
- `Auxiliary(name)` for metadata or support children such as validity, dictionary values, or zone
  statistics.

These relationships are used by debugging tools, split planning, and scan expansion.

## ScanPlan Expansion

`new_scan_plan` turns a typed layout into an immutable runtime `ScanPlan`. The plan should hold
layout metadata and child plan references, not per-morsel state. Runtime state belongs in prepared
handles or state caches created during preparation.

A `ScanPlan` implementation can specialize:

- `try_push_expr` to route expressions into children or rewrite them into a cheaper row domain;
- `prepare_read` to produce a `PreparedRead` for the plan's root value;
- `prepare_evidence` to produce cheap predicate evidence from metadata or indexes;
- `prepare_stats` and `prepare_aggregate_partial` for metadata-backed answers;
- `split_hints` to expose natural morsel boundaries; and
- `release` to drop caches behind the completed-row frontier.

The layout vtable expands child layouts by calling `child.new_scan_plan(req, session)`. Pass the
same `ScanRequest` through for children in the same row domain, and use a fresh
`ScanRequest::empty()` for children in independent row domains such as dictionary values or zone
statistics. This keeps the layout plugin responsible for its local structure while the scan runtime
owns predicate ordering, morsel execution, and output assembly.

## State Placement

Keep state at the narrowest level that can safely reuse it:

- `ScanPlan` stores immutable structure only.
- `PrepareCtx::shared_state` stores scan/file-level prepared state shared across prepared reads,
  evidence, statistics, and aggregate handles.
- Layouts with independent child row domains can create child-local prepared-state caches so one
  child shares dictionaries, zone tables, or decoded setup without colliding with another child.
- `ReadTask` and `EvidenceTask` own only one morsel's range and masks.
- Segment bytes belong to the segment source and segment cache, not to layout plans.

This separation lets a scan clone and prepare many pushed expressions while still sharing expensive
setup where the row domain is the same.

## Registration

Register layout vtables through the session's layout registry:

```rust
use vortex_layout::LayoutSessionExt;

session.layouts().register_v2(MyLayout);
```

The session resolves serialized layout IDs through this registry when opening a Vortex file.
