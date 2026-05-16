//! Vortex layout binding.
//!
//! Lowers a Vortex `LayoutRef` into an engine-visible operator
//! subgraph. Composite layouts decompose into recursively-bound
//! children plus an explicit combiner; only `vortex.flat` is a
//! leaf operator.
//!
//! - `vortex.struct` → N per-field subgraphs + [`StructAssembler`]
//! - `vortex.chunked` → N per-chunk subgraphs + [`ChunkConcat`]
//! - `vortex.zoned` (legacy "stats") → recurse into the data
//!   child; the zones auxiliary child is dropped pending a
//!   pruning operator
//! - `vortex.dict` → values subgraph + codes subgraph +
//!   [`DictDecode`]
//! - `vortex.flat` → leaf [`FlatLayoutOperator`]
//!
//! Public entry points:
//!
//! - [`bind_into_graph`] attaches a layout's full subgraph to an
//!   `OperatorGraph` and returns the root output's `OperatorId`.
//! - [`bind_field`] / [`bind_field_filtered`] do the same but walk
//!   a struct field path first; the filtered variant lowers
//!   `WHERE` + `SELECT` to a native scan + [`Filter`](crate::operators::Filter)
//!   chain. There is no `*_evaluation` involvement anywhere in
//!   the engine path.
//! - [`read_full_file`] / [`read_layout_via_engine`] are testing
//!   conveniences that bind a layout, attach an
//!   [`ArrayCollectSink`], and drive the resulting graph.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::lit;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::VortexFile;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::Layout;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::flat::Flat;
use vortex_layout::segments::SegmentSource;
use vortex_session::VortexSession;

use crate::operators::ArrayCollectSink;
use crate::EngineError;
use crate::EngineResult;
use crate::OperatorNode;

mod chunk_concat;
mod dict_decode;
mod flat;
mod struct_assembler;
mod zone_map;

pub use chunk_concat::ChunkConcat;
pub use dict_decode::DictDecode;
pub use flat::FlatLayoutOperator;
pub use struct_assembler::StructAssembler;
pub use zone_map::ZoneMapOperator;
pub use zone_map::ZoneMapSink;

/// Bundles a Vortex file with the runtime that drives its segment
/// source.
///
/// The runtime must outlive the file: `VortexOpenOptions::open_path`
/// spawns an I/O driver task on it, and any `LayoutReaderRef` derived
/// from the file requests segments through that driver.
pub struct VortexFileHandle {
    pub file: VortexFile,
    pub runtime: Arc<CurrentThreadRuntime>,
}

impl VortexFileHandle {
    /// The session configured with the file's runtime handle. Used to
    /// build `LayoutReader`s.
    pub fn session(&self) -> VortexSession {
        default_session().with_handle(self.runtime.handle())
    }
}

/// Open a Vortex file using a fresh `CurrentThreadRuntime` and return
/// both the file and the runtime.
pub fn open_vortex_file(path: impl AsRef<Path>) -> EngineResult<VortexFileHandle> {
    let runtime = Arc::new(CurrentThreadRuntime::new());
    let handle = runtime.handle();
    let session = default_session().with_handle(handle);

    let path = path.as_ref().to_path_buf();
    let file = runtime
        .block_on(async move { session.open_options().open_path(path).await })
        .map_err(|e| EngineError::message(format!("open vortex file: {e}")))?;

    Ok(VortexFileHandle { file, runtime })
}

fn default_session() -> VortexSession {
    use vortex::VortexSessionDefault;
    let session = VortexSession::default();
    crate::kernels::install(&session);
    session
}

/// Walk a layout tree depth-first and return the first layout whose
/// encoding matches the predicate.
pub fn find_first<F>(root: &dyn Layout, pred: F) -> Option<LayoutRef>
where
    F: Fn(&dyn Layout) -> bool,
{
    root.depth_first_traversal()
        .flatten()
        .find(|layout| pred(layout.as_ref()))
}

/// Find the first reachable `FlatLayout` (depth-first) under a root
/// layout.
pub fn find_first_flat_layout(
    root: &dyn Layout,
) -> Option<Arc<vortex_layout::layouts::flat::FlatLayout>> {
    let layout = find_first(root, |l| l.is::<Flat>())?;
    Some(layout.into::<Flat>())
}

/// Walk a struct layout and return the child subtree for the named
/// field. Returns `None` if `layout` is not a struct or the field is
/// missing. Skips the leading validity child for nullable structs.
pub fn struct_field_subtree(layout: &dyn Layout, field: &str) -> Option<LayoutRef> {
    use vortex_layout::LayoutChildType;
    for idx in 0..layout.nchildren() {
        if let LayoutChildType::Field(name) = layout.child_type(idx)
            && name.as_ref() == field
        {
            return layout.child(idx).ok();
        }
    }
    None
}

/// Add a sub-graph that reads `layout` to `graph`, returning the
/// `OperatorId` whose output port 0 carries the binding's result.
///
/// This is the *engine-visible* binder: composite layouts produce
/// multiple operators. A struct binds to one source operator per
/// field plus a `StructAssembler`; leaves bind to a single source
/// operator. Caller is responsible for connecting the returned
/// operator's output to whatever consumer.
///
/// Bind-time context threaded through every recursion of
/// [`bind_into_graph`]. Carries:
///
/// - `row_range`: which rows of `layout` are wanted. Default
///   `0..layout.row_count()` (the whole layout). Narrower ranges
///   trigger bind-time pruning at composite boundaries.
/// - `expression`: the analysable expression the binder offers to
///   stat-aware operators (today: `Zoned`). Default
///   `lit(true)`. Composite binders pass it through unchanged
///   because struct/chunked/dict are pure structural
///   re-arrangements; only `Zoned` consumes it.
///
/// The pruning side-channel is *not* carried on `BindContext`. It
/// is a private link between the `ZoneOperator` produced by a
/// `Zoned` binding and the `ChunkConcat` immediately below it on
/// the data side. The `Zoned` binder shares an `Arc<PruningResource>`
/// between exactly those two operators at construction; nothing
/// else in the graph sees it.
///
/// `expression` is the *pushdown candidate*: a predicate the bind
/// layer offers to upstream layouts. Each layout is free to consume
/// some, all, or none of it. What the layout *did not* consume is
/// reported as the `residual_predicate` on the [`BindResult`] —
/// callers that wrap the bound subgraph in a `Filter` use that
/// residual to decide whether (and what) to filter.
///
/// `ordering` is the *consumer's row-order requirement*. It tells
/// the bind layer (and the multi-shard fan-in above it) what
/// guarantee the consumer expects about the output order. Most
/// callers don't care; the default is [`OutputOrdering::Unordered`]
/// so existing call sites are unchanged.
#[derive(Clone)]
pub struct BindContext {
    pub row_range: Range<u64>,
    pub expression: Expression,
    pub ordering: OutputOrdering,
}

impl BindContext {
    /// Convenience: full row range, no filter, unordered output.
    pub fn full(row_count: u64) -> Self {
        Self {
            row_range: 0..row_count,
            expression: lit(true),
            ordering: OutputOrdering::Unordered,
        }
    }

    /// Convenience: full row range, custom expression, unordered.
    pub fn with_expression(row_count: u64, expression: Expression) -> Self {
        Self {
            row_range: 0..row_count,
            expression,
            ordering: OutputOrdering::Unordered,
        }
    }

    /// Builder method: set the row-order requirement.
    pub fn with_ordering(mut self, ordering: OutputOrdering) -> Self {
        self.ordering = ordering;
        self
    }
}

/// What the consumer of a bound subgraph needs (or the bind result
/// reports it produces) about row order.
///
/// Today there are two shapes; finer-grained sort-key tracking
/// (e.g. "ordered by `(country, timestamp)`") is intentionally
/// deferred — it slots in as additional variants when a
/// merge-on-key plan needs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OutputOrdering {
    /// Rows may arrive in any order. Multi-input fan-ins use a
    /// `Union` (arrival-order interleave) which is the cheapest
    /// option — producers race, no per-input buffering, no
    /// synchronisation beyond the channel merge. Right for
    /// count-distinct, group-by aggregate, hash-join probe,
    /// write-to-bag-of-rows.
    #[default]
    Unordered,
    /// Rows must arrive in the order: all rows from input 0, then
    /// all rows from input 1, then …, then all rows from input N-1.
    /// Multi-input fan-ins use a `Concat` operator (drain inputs
    /// in order, seal each before pulling from the next). Right
    /// for ORDER-BY-at-leaf, range-keyed downstream, sequential
    /// writes that need stable ordering.
    OrderedByInputIndex,
}

/// Result of binding a layout subtree into the operator graph.
///
/// `bind_into_graph` returns this struct — not just an `OperatorId`
/// — so the caller can decide whether (and what) to wrap on top.
/// In particular, the `residual_predicate` field reports what part
/// of the [`BindContext`]'s pushdown predicate the bound subgraph
/// did *not* consume. A `Filter` wrapper at the top of the chain
/// only needs to evaluate the residual; if the residual is `None`,
/// the bound subgraph already produces an exact result for the
/// pushdown and the `Filter` can be skipped entirely.
///
/// Today every layout sets `residual_predicate = Some(predicate)`
/// (no layout consumes a predicate exactly) so behavior is
/// unchanged from before this struct existed. Future
/// specializations — dict-eq, exact zone proof, bloom-filter
/// pushdown — will clear the residual when the layout proves it
/// matched the predicate exactly.
#[derive(Clone)]
pub struct BindResult {
    /// The id of the operator whose output port carries the bound
    /// subgraph's stream. Wire this to whatever consumer the caller
    /// builds on top.
    pub op: crate::OperatorId,
    /// Predicate that downstream still needs to evaluate. `None`
    /// means the bound subgraph is exact for the pushed-down
    /// predicate. Defaults to `Some(ctx.expression)` for layouts
    /// that don't consume the predicate.
    pub residual_predicate: Option<Expression>,
    /// Static guarantees the bound subgraph offers about its
    /// output. Currently a small struct — extend as new
    /// specializations land.
    pub capabilities: BindCapabilities,
}

/// Static guarantees a bound subgraph offers to its caller.
///
/// Default values are conservative ("nothing guaranteed"). Layouts
/// that can offer stronger guarantees (e.g. an exact dict-eq match,
/// a sorted output) override the relevant fields. Callers consult
/// these flags to skip wrapping operators that would otherwise be
/// unconditional.
#[derive(Clone, Default, Debug)]
pub struct BindCapabilities {
    /// `true` iff the bound subgraph's output rows are exactly the
    /// rows that satisfy the pushed-down predicate. Equivalent to
    /// `residual_predicate.is_none()` and intentionally redundant
    /// with it for callers that want a one-line check.
    pub output_is_exact_for_pushdown: bool,
    /// What row-order guarantee the bound subgraph offers. A
    /// caller that requested `OutputOrdering::OrderedByInputIndex`
    /// must check this matches before relying on order. The
    /// guarantee is a *superset* relationship: a subgraph that
    /// offers `OrderedByInputIndex` legitimately satisfies a
    /// caller that asked for `Unordered` (extra guarantee is fine);
    /// the reverse is not.
    pub output_ordering: OutputOrdering,
}

/// Composites currently decomposed: `struct` (per-field sources +
/// `StructAssembler`), `chunked` (per-chunk subgraphs +
/// `ChunkConcat`, with bind-time chunk pruning by `row_range`),
/// `zoned` (transparent — recurse into the `data` child; the
/// `zones` auxiliary child is skipped pending pruning hookup),
/// `dict` (values subgraph + codes subgraph + `DictDecode`). Leaf
/// encodings (`flat`) construct a native source operator that
/// reads its segment directly via `SegmentSource` — no
/// `LayoutReader::projection_evaluation` involvement anywhere in
/// the engine path.
///
/// `ctx.row_range` selects a sub-range of the layout's natural
/// rows; passing a narrower range performs bind-time pruning.
/// `ctx.expression` is forwarded unchanged through structural
/// composites and consumed by stat-aware operators (today: zoned
/// pruning).
pub fn bind_into_graph(
    graph: &mut crate::OperatorGraph,
    layout: LayoutRef,
    ctx: BindContext,
    name: impl Into<Arc<str>>,
    runtime: Arc<CurrentThreadRuntime>,
    segment_source: Arc<dyn SegmentSource>,
    session: &VortexSession,
) -> EngineResult<BindResult> {
    use crate::Cardinality;
    use crate::ChannelBuffer;
    use crate::Domain;
    use crate::DomainId;
    use crate::OperatorGraph as Graph;
    use crate::OperatorId;
    use crate::OperatorNode;
    use vortex_layout::LayoutChildType;
    use vortex_layout::layouts::chunked::Chunked;
    use vortex_layout::layouts::dict::Dict;
    use vortex_layout::layouts::flat::Flat;
    use vortex_layout::layouts::struct_::Struct;
    use vortex_layout::layouts::zoned::Zoned;

    let name: Arc<str> = name.into();
    let BindContext {
        row_range,
        expression,
        ordering,
    } = ctx;
    let _ = ordering; // Today every layout reports its own
                      // `output_ordering` capability based on its
                      // shape; the requested ordering doesn't yet
                      // change which physical ops are emitted.
                      // It will when a multi-shard `Concat`-vs-`Union`
                      // choice lives in this layer.
    if row_range.start >= row_range.end {
        return Err(EngineError::message(format!(
            "bind_into_graph: empty row_range {:?} for {name}",
            row_range
        )));
    }
    if row_range.end > layout.row_count() {
        return Err(EngineError::message(format!(
            "bind_into_graph: row_range {:?} exceeds layout row_count {}",
            row_range,
            layout.row_count()
        )));
    }

    // Struct layouts decompose into per-field sources + an assembler.
    if layout.is::<Struct>() {
        // Walk the struct's children, recursively binding each
        // non-validity field into the same graph. (We currently skip
        // the validity child for nullable structs — the assembler
        // emits a non-nullable struct array. Production would route
        // validity through the assembler too.)
        let struct_dtype = layout.dtype().clone();
        let struct_fields = struct_dtype
            .as_struct_fields_opt()
            .ok_or_else(|| EngineError::message("Struct layout has non-struct dtype"))?;
        let row_count = layout.row_count();

        let mut child_outputs: Vec<OperatorId> = Vec::new();
        let mut field_names: Vec<Arc<str>> = Vec::new();
        let mut field_dtypes: Vec<DType> = Vec::new();
        for idx in 0..layout.nchildren() {
            match layout.child_type(idx) {
                LayoutChildType::Field(field_name) => {
                    let field_layout = layout.child(idx).map_err(|e| {
                        EngineError::message(format!("struct child {idx}: {e}"))
                    })?;
                    let field_dtype = struct_fields
                        .field(field_name.as_ref())
                        .ok_or_else(|| {
                            EngineError::message(format!(
                                "struct dtype missing field {field_name}"
                            ))
                        })?;
                    let child_name: Arc<str> =
                        format!("{name}.{field_name}").into();
                    let child_ctx = BindContext {
                        row_range: row_range.clone(),
                        expression: expression.clone(),
                        ordering: OutputOrdering::Unordered,
                    };
                    let child = bind_into_graph(
                        graph,
                        field_layout,
                        child_ctx,
                        Arc::clone(&child_name),
                        Arc::clone(&runtime),
                        Arc::clone(&segment_source),
                        session,
                    )?;
                    child_outputs.push(child.op);
                    field_names.push(field_name.into());
                    field_dtypes.push(field_dtype);
                }
                LayoutChildType::Auxiliary(_) => {
                    // Skip validity / other auxiliary children.
                }
                LayoutChildType::Chunk(_) | LayoutChildType::Transparent(_) => {
                    return Err(EngineError::message(format!(
                        "unexpected child type {:?} under struct",
                        layout.child_type(idx)
                    )));
                }
            }
        }

        let assembler = OperatorNode::new(StructAssembler::new(
            format!("struct:{name}"),
            field_names,
            field_dtypes,
            row_count,
        ));
        let assembler_id = graph.add_operator(assembler);
        for (i, child_id) in child_outputs.into_iter().enumerate() {
            graph.connect(
                Graph::output(child_id),
                vec![Graph::input(assembler_id, i)],
                ChannelBuffer::bounded_bytes(256 << 20),
            );
        }
        return Ok(BindResult {
            op: assembler_id,
            residual_predicate: residual_default(&expression),
            // Struct assembles a row at a time from per-field
            // sources; output is in the same row order as the
            // shared input.
            capabilities: BindCapabilities {
                output_ordering: OutputOrdering::OrderedByInputIndex,
                ..Default::default()
            },
        });
    }

    // Chunked layouts decompose into per-chunk subgraphs + a
    // ChunkConcat that gathers them into one ordered stream over
    // the parent's combined domain. Chunks fully outside
    // `row_range` are pruned at bind time — never bound, never
    // read.
    if layout.is::<Chunked>() {
        let output_columns = column_count_of(layout.dtype());

        let mut child_outputs: Vec<OperatorId> = Vec::new();
        let mut chunk_domains: Vec<Domain> = Vec::new();
        // Offsets within the *output* domain (= sum of selected
        // chunk lengths). Chunks pruned at bind time don't
        // contribute to the output and don't appear in the
        // ChunkConcat input ports at all.
        let mut chunk_offsets: Vec<u64> = Vec::new();
        let mut emitted_rows: u64 = 0;
        for idx in 0..layout.nchildren() {
            match layout.child_type(idx) {
                LayoutChildType::Chunk((chunk_idx, chunk_offset)) => {
                    let chunk_layout = layout.child(idx).map_err(|e| {
                        EngineError::message(format!("chunked child {idx}: {e}"))
                    })?;
                    let chunk_end = chunk_offset + chunk_layout.row_count();
                    // Bind-time pruning: skip chunks fully outside
                    // the requested row range.
                    let intersect_start = row_range.start.max(chunk_offset);
                    let intersect_end = row_range.end.min(chunk_end);
                    if intersect_start >= intersect_end {
                        continue;
                    }
                    // Translate to chunk-local coordinates.
                    let local_start = intersect_start - chunk_offset;
                    let local_end = intersect_end - chunk_offset;
                    let chunk_name: Arc<str> =
                        format!("{name}[{chunk_idx}]").into();
                    let chunk_ctx = BindContext {
                        row_range: local_start..local_end,
                        expression: expression.clone(),
                        ordering: OutputOrdering::OrderedByInputIndex,
                    };
                    let chunk = bind_into_graph(
                        graph,
                        chunk_layout,
                        chunk_ctx,
                        Arc::clone(&chunk_name),
                        Arc::clone(&runtime),
                        Arc::clone(&segment_source),
                        session,
                    )?;
                    let emitted_chunk_rows = local_end - local_start;
                    child_outputs.push(chunk.op);
                    chunk_domains.push(Domain::new(
                        DomainId::new(format!("chunk:{chunk_name}")),
                        Cardinality::Exact(emitted_chunk_rows),
                    ));
                    chunk_offsets.push(emitted_rows);
                    emitted_rows += emitted_chunk_rows;
                }
                LayoutChildType::Auxiliary(_) => {
                    // Skip auxiliary children (e.g. row offset
                    // metadata) — they don't carry data.
                }
                LayoutChildType::Field(_) | LayoutChildType::Transparent(_) => {
                    return Err(EngineError::message(format!(
                        "unexpected child type {:?} under chunked",
                        layout.child_type(idx)
                    )));
                }
            }
        }

        if child_outputs.is_empty() {
            return Err(EngineError::message(format!(
                "bind_into_graph: chunked layout {name} produced no chunks for row_range {:?}",
                row_range
            )));
        }
        let combined_domain = Domain::new(
            DomainId::new(format!("chunked:{name}")),
            Cardinality::Exact(emitted_rows),
        );
        let concat = OperatorNode::new(ChunkConcat::new(
            format!("chunk_concat:{name}"),
            combined_domain,
            chunk_domains,
            chunk_offsets,
            output_columns,
        ));
        let concat_id = graph.add_operator(concat);
        for (i, child_id) in child_outputs.into_iter().enumerate() {
            graph.connect(
                Graph::output(child_id),
                vec![Graph::input(concat_id, i)],
                ChannelBuffer::bounded_bytes(256 << 20),
            );
        }
        return Ok(BindResult {
            op: concat_id,
            residual_predicate: residual_default(&expression),
            // ChunkConcat drains chunks in input-index order; output
            // is in the same order as the chunked layout's chunks.
            capabilities: BindCapabilities {
                output_ordering: OutputOrdering::OrderedByInputIndex,
                ..Default::default()
            },
        });
    }

    // Zoned (legacy "stats") wrapper: child 0 is the data layout
    // (Transparent), child 1 is the zones map (Auxiliary). V1
    // recurses into the data child and drops the zones child on
    // the floor — pruning is a follow-up that re-introduces the
    // zones layout via a sibling operator publishing a pruning
    // resource.
    if layout.is::<Zoned>() {
        use vortex_array::dtype::FieldPath;
        use vortex_array::dtype::FieldPathSet;
        use vortex_array::expr::pruning::checked_pruning_expr;
        use vortex_layout::layouts::zoned::ZonedLayout;

        // Locate the data + zones children.
        let mut data_layout: Option<LayoutRef> = None;
        let mut zones_layout: Option<LayoutRef> = None;
        for idx in 0..layout.nchildren() {
            let child = layout
                .child(idx)
                .map_err(|e| EngineError::message(format!("zoned child {idx}: {e}")))?;
            match layout.child_type(idx) {
                LayoutChildType::Transparent(_) => data_layout = Some(child),
                LayoutChildType::Auxiliary(_) => zones_layout = Some(child),
                LayoutChildType::Chunk(_) | LayoutChildType::Field(_) => {
                    return Err(EngineError::message(format!(
                        "unexpected child type {:?} under zoned",
                        layout.child_type(idx)
                    )));
                }
            }
        }
        let data_layout =
            data_layout.ok_or_else(|| EngineError::message("zoned: no Transparent data child"))?;
        let zones_layout =
            zones_layout.ok_or_else(|| EngineError::message("zoned: no Auxiliary zones child"))?;

        // Pull stats metadata from the typed Zoned layout.
        let zoned: &ZonedLayout = layout.as_::<Zoned>();
        let zone_len = zoned.zone_len() as u64;
        let nzones = zoned.nzones();
        let present_stats = Arc::clone(zoned.present_stats());
        let column_dtype = data_layout.dtype().clone();
        let data_row_count = data_layout.row_count();

        // Lower the bind expression to a pruning predicate over
        // the zone-stat schema. For a flat-column zoned layout
        // (the typical case), each present stat is a top-level
        // field of the zones struct (`min`, `max`, …) — that's
        // what we expose to `checked_pruning_expr` as available.
        let available_stats: FieldPathSet = present_stats
            .iter()
            .map(|stat| FieldPath::from_name(stat.name()))
            .collect();
        let lowered = checked_pruning_expr(&expression, &available_stats);

        // Always bind the data subgraph; pruning is opportunistic.
        let data_name: Arc<str> = format!("{name}.data").into();
        let data_ctx = BindContext {
            row_range: row_range.clone(),
            expression: expression.clone(),
            ordering: OutputOrdering::OrderedByInputIndex,
        };
        let data = bind_into_graph(
            graph,
            data_layout,
            data_ctx,
            data_name,
            Arc::clone(&runtime),
            Arc::clone(&segment_source),
            session,
        )?;
        let data_id = data.op;

        let Some((pruning_predicate, _required_stats)) = lowered else {
            // Predicate not lowerable to a pruning expression
            // (e.g. expression doesn't reference column stats).
            // No pruning operator — return the data subgraph
            // directly. Forward the data subgraph's residual: it's
            // the same shape (no zone-pruning wrapper added).
            return Ok(BindResult {
                op: data_id,
                residual_predicate: data.residual_predicate,
                capabilities: data.capabilities,
            });
        };

        // Allocate zone row offsets.
        let zone_row_offsets: Vec<u64> = (0..=nzones)
            .map(|i| ((i as u64) * zone_len).min(data_row_count))
            .collect();
        let resource = Arc::new(crate::resources::ZoneMapResource::new(
            pruning_predicate,
            zone_row_offsets,
        ));

        // Bind the zones subgraph.
        let zones_row_count = zones_layout.row_count();
        let zones_dtype = zones_layout.dtype().clone();
        let zones_name: Arc<str> = format!("{name}.zones").into();
        let zones = bind_into_graph(
            graph,
            zones_layout,
            BindContext::full(zones_row_count),
            zones_name,
            Arc::clone(&runtime),
            Arc::clone(&segment_source),
            session,
        )?;
        let zones_id = zones.op;

        // ZoneMapSink consumes zones, builds ZoneMap, evaluates
        // predicate, publishes mask to resource.
        let zones_sink_domain = Domain::new(
            DomainId::new(format!("zones:{name}")),
            Cardinality::Exact(zones_row_count),
        );
        drop(zones_dtype); // dtype is implicit on the channel
        let sink = OperatorNode::new(ZoneMapSink::new(
            format!("zone_sink:{name}"),
            zones_sink_domain,
            column_dtype.clone(),
            present_stats,
            zone_len,
            data_row_count,
            session.clone(),
            Arc::clone(&resource),
        ));
        let sink_id = graph.add_operator(sink);
        graph.connect(
            Graph::output(zones_id),
            vec![Graph::input(sink_id, 0)],
            ChannelBuffer::bounded_bytes(64 << 20),
        );

        // ZoneMapOperator forwards data, ANDing the resource's
        // demand into outgoing batches' demand masks.
        let data_input_domain = Domain::new(
            DomainId::new(format!("zoned_in:{name}")),
            Cardinality::Exact(row_range.end - row_range.start),
        );
        let zoned_output_domain = Domain::new(
            DomainId::new(format!("zoned_out:{name}")),
            Cardinality::Exact(row_range.end - row_range.start),
        );
        let output_columns = column_count_of(&column_dtype);
        let zone_op = OperatorNode::new(ZoneMapOperator::new(
            format!("zone_op:{name}"),
            data_input_domain,
            zoned_output_domain,
            output_columns,
            row_range.start,
            Arc::clone(&resource),
        ));
        let zone_op_id = graph.add_operator(zone_op);
        graph.connect(
            Graph::output(data_id),
            vec![Graph::input(zone_op_id, 0)],
            ChannelBuffer::bounded_bytes(256 << 20),
        );
        // Zone pruning is *inexact* for filter purposes: kept zones
        // still contain rows that don't match the predicate. The
        // residual stays = the original predicate; a downstream
        // Filter must still evaluate it. Order: ZoneMapOperator
        // is a pass-through over the data subgraph's ordering.
        return Ok(BindResult {
            op: zone_op_id,
            residual_predicate: residual_default(&expression),
            capabilities: BindCapabilities {
                output_ordering: OutputOrdering::OrderedByInputIndex,
                ..Default::default()
            },
        });
    }

    // Dict: values (auxiliary, typically small) + codes
    // (transparent, one per output row) + a DictDecode that
    // materialises codes against values via `take`.
    if layout.is::<Dict>() {
        let mut values_id: Option<OperatorId> = None;
        let mut codes_id: Option<OperatorId> = None;
        let mut values_dtype: Option<DType> = None;
        let mut values_rows: u64 = 0;
        let mut codes_rows: u64 = 0;
        for idx in 0..layout.nchildren() {
            let child_layout = layout
                .child(idx)
                .map_err(|e| EngineError::message(format!("dict child {idx}: {e}")))?;
            match layout.child_type(idx) {
                LayoutChildType::Auxiliary(child_name) => {
                    let sub_name: Arc<str> = format!("{name}.{child_name}").into();
                    let child_rows = child_layout.row_count();
                    values_dtype = Some(child_layout.dtype().clone());
                    values_rows = child_rows;
                    // Values are a lookup table: always full
                    // range, no predicate to evaluate against
                    // (the predicate analyses output rows, which
                    // are codes-driven).
                    let values_ctx = BindContext::full(child_rows);
                    values_id = Some(
                        bind_into_graph(
                            graph,
                            child_layout,
                            values_ctx,
                            sub_name,
                            Arc::clone(&runtime),
                            Arc::clone(&segment_source),
                            session,
                        )?
                        .op,
                    );
                }
                LayoutChildType::Transparent(child_name) => {
                    let sub_name: Arc<str> = format!("{name}.{child_name}").into();
                    codes_rows = child_layout.row_count();
                    let codes_ctx = BindContext {
                        row_range: row_range.clone(),
                        expression: expression.clone(),
                        ordering: OutputOrdering::OrderedByInputIndex,
                    };
                    let _ = (); // (no pruning resource on dict path V1)
                    codes_id = Some(
                        bind_into_graph(
                            graph,
                            child_layout,
                            codes_ctx,
                            sub_name,
                            Arc::clone(&runtime),
                            Arc::clone(&segment_source),
                            session,
                        )?
                        .op,
                    );
                }
                LayoutChildType::Field(_) | LayoutChildType::Chunk(_) => {
                    return Err(EngineError::message(format!(
                        "unexpected child type {:?} under dict",
                        layout.child_type(idx)
                    )));
                }
            }
        }
        let values_id = values_id
            .ok_or_else(|| EngineError::message("dict layout missing values child"))?;
        let codes_id = codes_id
            .ok_or_else(|| EngineError::message("dict layout missing codes child"))?;
        let values_dtype = values_dtype.expect("set with values_id");
        // Codes are 1:1 with the parent rows; the narrowed
        // output is exactly `row_range`. Values are address-only
        // (lookup table), unaffected by row narrowing.
        let output_rows = row_range.end - row_range.start;
        let _ = codes_rows; // codes' natural size before narrowing

        let output_columns = column_count_of(layout.dtype());
        let output_domain = Domain::new(
            DomainId::new(format!("dict:{name}")),
            Cardinality::Exact(output_rows),
        );
        let values_domain = Domain::new(
            DomainId::new(format!("dict:{name}.values")),
            Cardinality::Exact(values_rows),
        );
        let codes_domain = Domain::new(
            DomainId::new(format!("dict:{name}.codes")),
            Cardinality::Exact(output_rows),
        );
        let decode = OperatorNode::new(DictDecode::new(
            format!("dict_decode:{name}"),
            output_domain,
            values_domain,
            codes_domain,
            values_dtype,
            output_columns,
        ));
        let decode_id = graph.add_operator(decode);
        graph.connect(
            Graph::output(values_id),
            vec![Graph::input(decode_id, 0)],
            ChannelBuffer::bounded_bytes(64 << 20),
        );
        graph.connect(
            Graph::output(codes_id),
            vec![Graph::input(decode_id, 1)],
            ChannelBuffer::bounded_bytes(256 << 20),
        );
        return Ok(BindResult {
            op: decode_id,
            residual_predicate: residual_default(&expression),
            // DictDecode emits one decoded row per code row in
            // codes-input order.
            capabilities: BindCapabilities {
                output_ordering: OutputOrdering::OrderedByInputIndex,
                ..Default::default()
            },
        });
    }

    // Leaf encoding (flat): native source operator that reads its
    // single segment via SegmentSource — no LayoutReader involved.
    if layout.is::<Flat>() {
        let flat = layout
            .as_opt::<Flat>()
            .ok_or_else(|| EngineError::message("flat layout downcast failed"))?
            .clone();
        let node = OperatorNode::new(FlatLayoutOperator::new(
            name.as_ref().to_string(),
            flat,
            row_range,
            segment_source,
            session.clone(),
            runtime,
        ));
        return Ok(BindResult {
            op: graph.add_operator(node),
            residual_predicate: residual_default(&expression),
            // FlatLayoutOperator emits rows in disk order — i.e.,
            // in input-index order over its single segment.
            capabilities: BindCapabilities {
                output_ordering: OutputOrdering::OrderedByInputIndex,
                ..Default::default()
            },
        });
    }

    Err(EngineError::message(format!(
        "bind_into_graph: unsupported layout encoding {} for {name}",
        layout.encoding_id()
    )))
}

/// Default residual: passes the predicate through unchanged unless
/// it's the trivial `lit(true)` placeholder used by
/// [`BindContext::full`]. Layouts that consume the predicate
/// exactly (future work — dict-eq, exact zone proof, bloom-filter
/// pushdown) override the residual to `None` rather than calling
/// this helper. Today every layout uses this default, so
/// [`BindResult::residual_predicate`] reads back as `Some(predicate)`
/// for any non-trivial predicate — the
/// [`bind_field_filtered`] caller adds a `Filter` on top
/// unconditionally, preserving pre-refactor behavior.
fn residual_default(expression: &Expression) -> Option<Expression> {
    if is_trivially_true(expression) {
        None
    } else {
        Some(expression.clone())
    }
}

/// Cheap check for the trivial `lit(true)` placeholder.
fn is_trivially_true(expression: &Expression) -> bool {
    expression.to_string() == "true"
}

/// Bind a single field path through (nested) struct layouts.
///
/// The returned operator reads only that field. This is the
/// column-pruning entry point: a request for `["UserID"]` from the
/// 108-column ClickBench struct binds and reads only the UserID
/// subtree — no other column's segments are touched.
///
/// Each path component must name a struct field of the surrounding
/// layout. Returns `EngineError` if the path cannot be resolved.
///
/// Adds the field's subgraph to `graph` via [`bind_into_graph`] and
/// returns the [`OperatorId`](crate::OperatorId) whose
/// output port 0 carries the bound field's stream. Callers wire that
/// id to whatever consumer they want.
pub fn bind_field(
    graph: &mut crate::OperatorGraph,
    layout: LayoutRef,
    path: &[&str],
    name: impl Into<Arc<str>>,
    runtime: Arc<CurrentThreadRuntime>,
    segment_source: Arc<dyn SegmentSource>,
    session: &VortexSession,
) -> EngineResult<crate::OperatorId> {
    let mut current = layout;
    for component in path {
        let next = struct_field_subtree(current.as_ref(), component).ok_or_else(|| {
            EngineError::message(format!(
                "field '{component}' not found under layout '{}'",
                current.encoding_id()
            ))
        })?;
        current = next;
    }
    let ctx = BindContext::full(current.row_count());
    bind_into_graph(graph, current, ctx, name, runtime, segment_source, session)
        .map(|r| r.op)
}

/// Bind a single field path and apply a predicate + projection on
/// top via the engine-native [`Filter`](crate::operators::Filter)
/// operator. No `filter_evaluation` / `projection_evaluation`
/// involvement — the layout produces full batches and the filter
/// applies the expression natively.
///
/// Adds the scan + filter chain to `graph` and returns the
/// [`OperatorId`](crate::OperatorId) of the filter (whose output
/// is the predicate-true rows of the projection). The output's
/// row count is unknown at bind time (depends on filter
/// selectivity).
///
/// Zone-based pruning (the previous fast path) lives outside the
/// engine path now; bringing it back is the zoned operator
/// decomposition pass — for now the zoned operator forwards to its
/// data child unchanged.
#[allow(clippy::too_many_arguments)]
pub fn bind_field_filtered(
    graph: &mut crate::OperatorGraph,
    layout: LayoutRef,
    path: &[&str],
    predicate: Expression,
    projection: Expression,
    name: impl Into<Arc<str>>,
    runtime: Arc<CurrentThreadRuntime>,
    segment_source: Arc<dyn SegmentSource>,
    session: &VortexSession,
) -> EngineResult<crate::OperatorId> {
    use crate::ChannelBuffer;
    use crate::operators::Filter;

    let name: Arc<str> = name.into();

    // Walk to the field's layout subtree.
    let mut current = layout;
    for component in path {
        let next = struct_field_subtree(current.as_ref(), component).ok_or_else(|| {
            EngineError::message(format!(
                "field '{component}' not found under layout '{}'",
                current.encoding_id()
            ))
        })?;
        current = next;
    }
    let scan_name: Arc<str> = format!("{name}.scan").into();
    // Pass the predicate down so any zoned layouts in the subtree
    // can lower it to per-zone pruning. The Filter operator above
    // still applies it row-by-row for correctness; pruning is a
    // fast path that skips work, not a replacement for the filter.
    let scan_ctx = BindContext::with_expression(current.row_count(), predicate.clone());
    let scan = bind_into_graph(
        graph,
        current.clone(),
        scan_ctx,
        scan_name,
        Arc::clone(&runtime),
        Arc::clone(&segment_source),
        session,
    )?;

    // If the bound subgraph reported it consumed the predicate
    // exactly (residual == None) and the projection is identity,
    // skip the Filter wrapper entirely. Today no layout sets
    // residual to None for non-trivial predicates — this branch
    // is dormant and lights up as future specializations
    // (dict-eq, exact zone proof, bloom-filter pushdown) ship.
    let projection_is_identity = is_trivially_root(&projection);
    if scan.residual_predicate.is_none() && projection_is_identity {
        return Ok(scan.op);
    }

    let residual = scan.residual_predicate.unwrap_or_else(|| predicate.clone());
    let filter = OperatorNode::new(Filter::new(
        format!("filter:{name}"),
        current.dtype().clone(),
        current.row_count(),
        residual,
        projection,
        session.clone(),
    ));
    let filter_id = graph.add_operator(filter);
    graph.connect(
        crate::OperatorGraph::output(scan.op),
        vec![crate::OperatorGraph::input(filter_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );
    Ok(filter_id)
}

/// Cheap structural check for the trivial `root()` projection.
fn is_trivially_root(expression: &Expression) -> bool {
    expression.to_string() == "$"
}

/// Read the entire root layout of a Vortex file end-to-end through
/// the engine. Convenience wrapper that opens the file, binds the
/// root layout, plumbs a single source→sink graph, and returns the
/// emitted arrays.
pub fn read_full_file(
    path: impl AsRef<Path>,
) -> EngineResult<(Vec<ArrayRef>, VortexFileHandle)> {
    use crate::ChannelBuffer;
    use crate::ExecutionMetrics;
    use crate::OperatorGraph;
    use crate::PreparedTask;
    use crate::TaskOptions;

    let handle = open_vortex_file(path)?;
    let session = handle.session();
    let segment_source = handle.file.segment_source();
    let layout = Arc::clone(handle.file.footer().layout());
    let row_count = handle.file.row_count();

    let mut graph = OperatorGraph::new();
    let source_id = bind_into_graph(
        &mut graph,
        layout,
        BindContext::full(row_count),
        "root",
        Arc::clone(&handle.runtime),
        segment_source,
        &session,
    )?
    .op;

    let domain = crate::Domain::new(
        crate::DomainId::new("layout_rows"),
        crate::Cardinality::Exact(row_count),
    );
    let captured: Arc<parking_lot::Mutex<Vec<ArrayRef>>> =
        Arc::new(parking_lot::Mutex::new(Vec::new()));
    let metrics = Arc::new(parking_lot::Mutex::new(ExecutionMetrics::default()));

    let sink_id = graph.add_operator(OperatorNode::new(ArrayCollectSink::new(
        "collect_arrays",
        domain,
        Arc::clone(&captured),
    )));
    graph.connect(
        OperatorGraph::output(source_id),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );

    PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    let arrays = captured.lock().clone();
    Ok((arrays, handle))
}

/// Read a specific layout (anywhere in the tree) end-to-end via a
/// fresh source→sink graph.
///
/// Uses [`bind_into_graph`] so composite layouts (struct, chunked)
/// are decomposed into their per-component subgraphs rather than
/// being absorbed into a single source operator.
pub fn read_layout_via_engine(
    layout: LayoutRef,
    name: impl Into<Arc<str>>,
    runtime: Arc<CurrentThreadRuntime>,
    segment_source: Arc<dyn SegmentSource>,
    session: &VortexSession,
) -> EngineResult<Vec<ArrayRef>> {
    use parking_lot::Mutex;

    use crate::Cardinality;
    use crate::ChannelBuffer;
    use crate::Domain;
    use crate::DomainId;
    use crate::ExecutionMetrics;
    use crate::OperatorGraph;
    use crate::PreparedTask;
    use crate::TaskOptions;

    let row_count = layout.row_count();
    let domain = Domain::new(DomainId::new("layout_rows"), Cardinality::Exact(row_count));
    let captured: Arc<Mutex<Vec<ArrayRef>>> = Arc::new(Mutex::new(Vec::new()));
    let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));

    let mut graph = OperatorGraph::new();
    let source_id = bind_into_graph(
        &mut graph,
        layout,
        BindContext::full(row_count),
        name,
        runtime,
        segment_source,
        session,
    )?
    .op;
    let sink_id = graph.add_operator(OperatorNode::new(ArrayCollectSink::new(
        "collect_arrays",
        domain,
        Arc::clone(&captured),
    )));
    graph.connect(
        OperatorGraph::output(source_id),
        vec![OperatorGraph::input(sink_id, 0)],
        ChannelBuffer::bounded_bytes(256 << 20),
    );

    PreparedTask::prepare(graph, metrics, TaskOptions::default())?.run()?;
    let arrays = captured.lock().clone();
    Ok(arrays)
}

/// Number of leaf columns implied by a layout dtype. Used to size
/// each source operator's output port.
///
/// We use Vortex's own structural notion of "fields": a struct dtype
/// reports its `nfields()`; everything else is treated as a single
/// column. Production will use a typed `OutputContract` instead.
pub(crate) fn column_count_of(dtype: &DType) -> usize {
    match dtype {
        DType::Struct(fields, _) => fields.nfields(),
        _ => 1,
    }
}

/// Approximate byte cost of an array, saturating to `usize::MAX` if
/// the underlying `nbytes()` exceeds the platform pointer size.
pub(crate) fn array_estimated_bytes(array: &ArrayRef) -> usize {
    usize::try_from(array.nbytes()).unwrap_or(usize::MAX)
}


