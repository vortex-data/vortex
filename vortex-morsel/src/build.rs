// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building an [`ExecPlan`] from a layout tree and a query.
//!
//! The plan is the immutable half of the design's split: one blueprint per scan. Each worker
//! instantiates one thread-local [`Arena`], whose node state survives IO suspension and is recycled
//! across that worker's morsels without crossing a thread boundary.
//!
//! Only the layouts and expression shapes named in the P1 scope are accepted. Anything else is a
//! build error rather than a silent fallback, so an unsupported query can never be timed as if
//! the prototype had executed it.

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::analysis::referenced_field_paths;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::chunked::Chunked;
use vortex_layout::layouts::flat::Flat;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::struct_::Struct;
use vortex_layout::layouts::zoned::LegacyStats;
use vortex_layout::layouts::zoned::Zoned;

use crate::io::IoKey;
use crate::io::ProducerId;
use crate::node::Arena;
use crate::node::ExecNode;
use crate::node::NodeId;
use crate::nodes::ChunkedExec;
use crate::nodes::ConjunctExec;
use crate::nodes::ConjunctMode;
use crate::nodes::ConjunctSlot;
use crate::nodes::FilterExec;
use crate::nodes::FlatExec;
use crate::nodes::StructExec;

/// The immutable blueprint of one node.
enum NodeSpec {
    Flat {
        layout: FlatLayout,
        root_offset: u64,
    },
    Chunked {
        chunk_offsets: Arc<[u64]>,
        children: Arc<[NodeId]>,
        dtype: DType,
    },
    Struct {
        names: FieldNames,
        children: Arc<[NodeId]>,
    },
    Conjunct {
        slots: Vec<(NodeId, BoundExpression)>,
        mode: ConjunctMode,
    },
    Filter {
        predicate: Option<NodeId>,
        projection: NodeId,
        expr: BoundExpression,
        dtype: DType,
    },
}

/// A shared, immutable execution plan for one scan.
pub struct ExecPlan {
    nodes: Vec<NodeSpec>,
    root: NodeId,
    output_dtype: DType,
    row_count: u64,
    /// Root-coordinate boundaries at which every column starts a fresh chunk, used as the
    /// default morsel cut.
    natural_splits: Vec<u64>,
}

impl ExecPlan {
    /// The root node of the plan.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The dtype the scan emits.
    pub fn output_dtype(&self) -> &DType {
        &self.output_dtype
    }

    /// The number of rows in the scanned layout.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    /// The union of every column's chunk boundaries, in root coordinates.
    pub fn natural_splits(&self) -> &[u64] {
        &self.natural_splits
    }

    /// Every flat node's stored unit and its root-coordinate row range, one entry per node.
    ///
    /// A segment referenced from two subtrees (a column in both filter and projection) appears
    /// once per referencing node, because each node registers its own use per morsel. This is
    /// the input to the shared-cell lease counts: the count for a unit is the number of
    /// (node, morsel) pairs whose ranges overlap.
    pub fn flat_uses(&self) -> impl Iterator<Item = (IoKey, Range<u64>)> + '_ {
        self.nodes.iter().filter_map(|spec| match spec {
            NodeSpec::Flat {
                layout,
                root_offset,
            } => Some((
                IoKey::Segment(layout.segment_id()),
                *root_offset..*root_offset + layout.row_count(),
            )),
            _ => None,
        })
    }

    /// The number of nodes in the plan.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Instantiate one worker's mutable arena from this blueprint.
    pub fn instantiate(&self) -> Arena {
        let nodes: Vec<Box<dyn ExecNode>> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, spec)| -> Box<dyn ExecNode> {
                match spec {
                    NodeSpec::Flat {
                        layout,
                        root_offset,
                    } => Box::new(FlatExec::new(
                        layout,
                        *root_offset,
                        ProducerId(u32::try_from(idx).unwrap_or(u32::MAX)),
                    )),
                    NodeSpec::Chunked {
                        chunk_offsets,
                        children,
                        dtype,
                    } => Box::new(ChunkedExec::new(
                        Arc::clone(chunk_offsets),
                        Arc::clone(children),
                        dtype.clone(),
                    )),
                    NodeSpec::Struct { names, children } => {
                        Box::new(StructExec::new(names.clone(), Arc::clone(children)))
                    }
                    NodeSpec::Conjunct { slots, mode } => Box::new(ConjunctExec::new(
                        slots
                            .iter()
                            .map(|(input, predicate)| ConjunctSlot {
                                input: *input,
                                predicate: predicate.clone(),
                            })
                            .collect(),
                        *mode,
                    )),
                    NodeSpec::Filter {
                        predicate,
                        projection,
                        expr,
                        dtype,
                    } => Box::new(FilterExec::new(
                        *predicate,
                        *projection,
                        expr.clone(),
                        dtype.clone(),
                    )),
                }
            })
            .collect();
        Arena::new(nodes)
    }
}

/// Build an execution plan for `layout` under `projection` and `filter`.
///
/// The expressions are *unbound*: each conjunct and the projection are re-bound against the
/// narrowed struct dtype of just the fields they reference, which is what lets a subtree read
/// only its own columns without any expression rewriting.
pub fn build_plan(
    layout: &LayoutRef,
    projection: &Expression,
    filter: Option<&Expression>,
    mode: ConjunctMode,
) -> VortexResult<ExecPlan> {
    let root_dtype = layout.dtype().clone();
    let root_fields = root_dtype
        .as_struct_fields_opt()
        .ok_or_else(|| vortex_err!("the morsel executor requires a struct-rooted layout"))?
        .clone();
    if root_dtype.is_nullable() {
        vortex_bail!("the morsel executor does not support a nullable root struct");
    }
    if !layout.is::<Struct>() {
        vortex_bail!(
            "the morsel executor requires a struct root layout, got {}",
            layout.encoding_id()
        );
    }

    let mut builder = Builder {
        nodes: Vec::new(),
        layout: LayoutRef::clone(layout),
        root_fields,
        splits: Vec::new(),
    };

    // The filter: one subtree per conjunct, each over just that conjunct's fields.
    let predicate = match filter {
        None => None,
        Some(filter) => {
            let conjuncts = split_conjuncts(filter);
            let mut slots = Vec::with_capacity(conjuncts.len());
            for conjunct in conjuncts {
                let (input, bound) = builder.build_scoped(&conjunct)?;
                slots.push((input, bound));
            }
            Some(builder.push(NodeSpec::Conjunct { slots, mode }))
        }
    };

    // The projection.
    let (projection_input, projection_bound) = builder.build_scoped(projection)?;
    let output_dtype = projection_bound.dtype().clone();
    let root = builder.push(NodeSpec::Filter {
        predicate,
        projection: projection_input,
        expr: projection_bound,
        dtype: output_dtype.clone(),
    });

    let row_count = layout.row_count();
    let mut natural_splits = builder.splits;
    natural_splits.push(row_count);
    natural_splits.sort_unstable();
    natural_splits.dedup();
    natural_splits.retain(|&split| split > 0 && split <= row_count);

    Ok(ExecPlan {
        nodes: builder.nodes,
        root,
        output_dtype,
        row_count,
        natural_splits,
    })
}

struct Builder {
    nodes: Vec<NodeSpec>,
    layout: LayoutRef,
    root_fields: StructFields,
    splits: Vec<u64>,
}

impl Builder {
    fn push(&mut self, spec: NodeSpec) -> NodeId {
        self.nodes.push(spec);
        NodeId::try_from(self.nodes.len() - 1).vortex_expect("exec plan exceeds u32 nodes")
    }

    /// Build the subtree for one expression: a struct over exactly the top-level fields the
    /// expression reads, plus that expression re-bound against the narrowed struct dtype.
    fn build_scoped(&mut self, expr: &Expression) -> VortexResult<(NodeId, BoundExpression)> {
        let full = expr.bind(self.layout.dtype())?;
        let names = self.referenced_top_level_fields(&full)?;

        let dtypes = names
            .iter()
            .map(|name| {
                self.root_fields
                    .field(name)
                    .ok_or_else(|| vortex_err!("field {name} not found in the scan dtype"))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let narrowed = DType::Struct(
            StructFields::new(FieldNames::from(names.clone()), dtypes),
            Nullability::NonNullable,
        );
        let bound = expr.bind(&narrowed)?;

        let mut children = Vec::with_capacity(names.len());
        for name in &names {
            let idx = self
                .root_fields
                .find(name)
                .ok_or_else(|| vortex_err!("field {name} not found in the scan dtype"))?;
            let field_layout = self.field_layout(idx)?;
            children.push(self.build_layout(&field_layout, 0)?);
        }

        let node = self.push(NodeSpec::Struct {
            names: FieldNames::from(names),
            children: Arc::from(children),
        });
        Ok((node, bound))
    }

    /// The struct layout's child for field `idx`, accounting for the validity slot.
    fn field_layout(&self, idx: usize) -> VortexResult<LayoutRef> {
        self.layout
            .slot(idx + 1)?
            .ok_or_else(|| vortex_err!("struct layout has no child for field {idx}"))
    }

    fn referenced_top_level_fields(&self, expr: &BoundExpression) -> VortexResult<Vec<FieldName>> {
        let paths = referenced_field_paths(expr)?;
        let mut names: Vec<FieldName> = Vec::new();
        let mut all = false;
        for path in paths.iter() {
            if path.is_root() {
                all = true;
                break;
            }
            match &path.parts()[0] {
                Field::Name(name) => {
                    if !names.contains(name) {
                        names.push(name.clone());
                    }
                }
                other => vortex_bail!("unsupported field reference {other:?}"),
            }
        }
        if all {
            names = self.root_fields.names().iter().cloned().collect();
        }
        // Keep the scan dtype's field order so `select` and `pack` see the fields they expect.
        names.sort_by_key(|name| self.root_fields.find(name).unwrap_or(usize::MAX));
        Ok(names)
    }

    /// Build the subtree for one column, recording its chunk boundaries as natural splits.
    fn build_layout(&mut self, layout: &LayoutRef, root_offset: u64) -> VortexResult<NodeId> {
        if layout.is::<Zoned>() || layout.is::<LegacyStats>() {
            let data = layout
                .slot(0)?
                .ok_or_else(|| vortex_err!("zoned layout has no data child"))?;
            return self.build_layout(&data, root_offset);
        }

        if layout.is::<Flat>() {
            self.splits.push(root_offset + layout.row_count());
            let flat = layout.as_::<Flat>().clone();
            return Ok(self.push(NodeSpec::Flat {
                layout: flat,
                root_offset,
            }));
        }

        if layout.is::<Chunked>() {
            let nchunks = layout.nchildren();
            let mut offsets = Vec::with_capacity(nchunks + 1);
            offsets.push(0u64);
            let mut children = Vec::with_capacity(nchunks);
            for idx in 0..nchunks {
                let child = layout
                    .slot(idx)?
                    .ok_or_else(|| vortex_err!("chunked layout has no child {idx}"))?;
                let offset = offsets[idx];
                children.push(self.build_layout(&child, root_offset + offset)?);
                offsets.push(offset + child.row_count());
            }
            return Ok(self.push(NodeSpec::Chunked {
                chunk_offsets: Arc::from(offsets),
                children: Arc::from(children),
                dtype: layout.dtype().clone(),
            }));
        }

        vortex_bail!(
            "the morsel executor supports flat and chunked columns only, got {} at row offset {}",
            layout.encoding_id(),
            root_offset
        )
    }
}

/// Split a conjunction into its conjuncts, mirroring the V1 `FilterExpr` split.
fn split_conjuncts(expr: &Expression) -> Vec<Expression> {
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;

    let mut conjuncts = Vec::new();
    let mut pending = vec![expr.clone()];
    while let Some(expr) = pending.pop() {
        let is_and = expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And);
        if is_and {
            pending.extend(expr.children().iter().rev().cloned());
        } else {
            conjuncts.push(expr);
        }
    }
    conjuncts
}

/// The morsel row ranges for a plan, from its natural splits, coalesced to `target_rows`.
pub(crate) fn cut_morsels(splits: &[u64], target_rows: u64) -> Vec<Range<u64>> {
    let mut morsels = Vec::new();
    let mut start = 0u64;
    for &split in splits {
        if split <= start {
            continue;
        }
        if split - start >= target_rows {
            morsels.push(start..split);
            start = split;
        }
    }
    if let Some(&last) = splits.last()
        && last > start
    {
        morsels.push(start..last);
    }
    morsels
}
