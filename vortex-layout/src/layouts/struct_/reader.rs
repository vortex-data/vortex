// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::future::try_join_all;
use futures::try_join;
use itertools::Itertools;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::RESERVED_VALIDITY_FIELD_NAME;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::analysis::AnnotationFn;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::is_not_null;
use vortex_array::expr::bound::mask;
use vortex_array::expr::bound::not;
use vortex_array::expr::bound::pack;
use vortex_array::expr::make_bound_free_field_annotator;
use vortex_array::expr::transform::BoundPartitionedExpr;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_project::Project;
use vortex_array::scalar_fn::fns::list_project::project_paths_bound;
use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::select::FieldSelection;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderRef;
use crate::LazyReaderChildren;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::partitioned::BoundPartitionedExprEval;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSource;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Item {
    Field(FieldName),
    Validity,
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Field(field_name) => write!(f, "{field_name}"),
            Item::Validity => f.write_str(RESERVED_VALIDITY_FIELD_NAME),
        }
    }
}

impl From<Item> for FieldName {
    fn from(item: Item) -> Self {
        match item {
            Item::Field(field_name) => field_name,
            Item::Validity => FieldName::from(RESERVED_VALIDITY_FIELD_NAME),
        }
    }
}

pub struct StructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,

    /// For a non-nullable struct, pack(f: $.f, ...) for every f in struct
    /// fields. For a nullable struct, mask(pack(f: $.f, ...), is_not_null($))
    /// for every f in fields.
    expanded_root_expr: BoundExpression,

    field_lookup: Option<HashMap<FieldName, usize>>,
    partitioned_expr_cache: DashMap<ExactBoundExpr, Arc<OnceLock<Partitioned>>>,
}

impl StructReader {
    pub(super) fn try_new(
        layout: StructLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: crate::LayoutReaderContext,
    ) -> VortexResult<Self> {
        let struct_dt = layout.struct_fields();

        // NOTE: This number is arbitrary and likely depends on the longest common prefix of field names
        let field_lookup = (struct_dt.nfields() > 80).then(|| {
            struct_dt
                .names()
                .iter()
                .enumerate()
                .map(|(i, n)| (n.clone(), i))
                .collect()
        });

        let nullable = layout.dtype().is_nullable();
        let extra = nullable as usize;

        let mut dtypes: Vec<DType> = Vec::with_capacity(struct_dt.nfields() + extra);
        let mut names: Vec<Arc<str>> = Vec::with_capacity(struct_dt.nfields() + extra);
        if nullable {
            dtypes.push(DType::Bool(Nullability::NonNullable));
            names.push(Arc::from("validity"));
        }
        dtypes.extend(struct_dt.fields());
        names.extend(struct_dt.names().iter().map(|x| Arc::clone(x.inner())));

        let lazy_children = LazyReaderChildren::new(
            Arc::clone(layout.children()),
            dtypes,
            names,
            Arc::clone(&segment_source),
            session.clone(),
            ctx,
        );

        // Create an expanded root expression that contains all fields of the struct.
        let expanded_root_expr = expanded_struct_root(layout.dtype(), struct_dt)?;

        // This is where we need to do some complex things with the scan in order to split it into
        // different scans for different fields.
        Ok(Self {
            layout,
            name,
            session,
            expanded_root_expr,
            lazy_children,
            field_lookup,
            partitioned_expr_cache: DashMap::with_hasher_and_shard_amount(Default::default(), 8),
        })
    }

    /// Return the [`StructFields`] of this layout.
    fn struct_fields(&self) -> &StructFields {
        self.layout.struct_fields()
    }

    /// Return the child reader for the struct coordinate.
    fn field_reader(&self, item: &Item) -> VortexResult<&LayoutReaderRef> {
        let name = match item {
            Item::Validity => {
                return self
                    .validity()?
                    .ok_or_else(|| vortex_err!("Non-nullable struct layout has no validity"));
            }
            Item::Field(name) => name,
        };
        let idx = self
            .field_lookup
            .as_ref()
            .and_then(|lookup| lookup.get(name).copied())
            .or_else(|| self.struct_fields().find(name))
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))?;
        self.field_reader_by_index(idx)
    }

    /// Return the child reader for the field, by index.
    fn field_reader_by_index(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        // Field `idx` always occupies slot `idx + 1`; the layout maps that to a dense child index,
        // accounting for the validity slot when the struct is nullable.
        let child_index = self
            .layout
            .slot_to_child(idx + 1)
            .vortex_expect("struct field slot is always present");
        self.lazy_children.get(child_index)
    }

    /// Return the reader for the struct validity, if present
    fn validity(&self) -> VortexResult<Option<&LayoutReaderRef>> {
        self.layout
            .slot_to_child(0)
            .map(|child_index| self.lazy_children.get(child_index))
            .transpose()
    }

    /// Utility for partitioning an expression over the fields of a struct.
    fn partition_expr(&self, expr: &BoundExpression) -> VortexResult<Partitioned> {
        let key = ExactBoundExpr(expr.clone());

        // Look up the cell under a shared shard lock; only a miss takes the write lock, and
        // only for as long as it takes to insert an empty cell.
        let cell = match self.partitioned_expr_cache.get(&key) {
            Some(entry) => Arc::clone(entry.value()),
            None => Arc::clone(
                self.partitioned_expr_cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceLock::new()))
                    .value(),
            ),
        };
        // All map guards are dropped here, so partitioning runs outside any shard lock.
        // Concurrent misses may compute redundantly; `get_or_init` keeps a single winner.

        if let Some(value) = cell.get() {
            return Ok(value.clone());
        }
        // Cache the bound trees themselves: rebinding would create fresh child identity and defeat
        // downstream `ExactBoundExpr` caches on every split.
        let result = self.compute_partitioned_expr(expr)?;
        Ok(cell.get_or_init(|| result).clone())
    }

    fn compute_partitioned_expr(&self, expr: &BoundExpression) -> VortexResult<Partitioned> {
        if let Some(direct) = self.direct_projection(expr)? {
            return Ok(Partitioned::Direct(Arc::new(direct)));
        }

        // First, we expand the root scope into the fields of the struct to ensure
        // that partitioning works correctly.
        let expr =
            expand_struct_root(expr.clone(), &self.expanded_root_expr, self.struct_fields())?;

        // Partition the expression into expressions that can be evaluated
        // over individual coordinates.
        let fields = self
            .dtype()
            .as_struct_fields_opt()
            .vortex_expect("struct dtype");
        let mut partitioned = partition_bound(expr.clone(), make_struct_item_annotator(fields))?;

        if partitioned.partitions.len() == 1 {
            // If there's only one partition, we step into the field scope of the original
            // expression by replacing any `$.a` with `$`.
            let item = partitioned.partition_annotations[0].clone();
            let partition =
                step_into_struct_field(expr, &item, self.field_reader(&item)?.dtype().clone())?;
            return Ok(Partitioned::Single(item, partition));
        }

        // We now need to process the partitioned expressions to rewrite the root scope
        // to be that of the field, rather than the struct. In other words, "stepping in"
        // to the field scope.
        let partitions = partitioned
            .partitions
            .iter()
            .zip_eq(partitioned.partition_annotations.iter())
            .map(|(expr, item)| {
                step_into_struct_field(expr.clone(), item, self.field_reader(item)?.dtype().clone())
            })
            .try_collect::<_, Vec<_>, _>()?
            .into_boxed_slice();
        partitioned.replace_partitions(partitions)?;

        Ok(Partitioned::Multi(Arc::new(partitioned)))
    }

    /// Recognize projections that pass fields of a non-nullable struct through unchanged: `$`,
    /// `select(..., $)` and `pack(name: $.field, ...)`.
    ///
    /// These read each field with its own root expression and assemble the result directly,
    /// instead of partitioning a `pack` over every field and evaluating it lazily on top of the
    /// field arrays, a per-field cost that dominates scans of wide tables.
    fn direct_projection(&self, expr: &BoundExpression) -> VortexResult<Option<DirectProjection>> {
        if self.dtype().is_nullable() {
            return Ok(None);
        }
        let fields = self.struct_fields();
        let field_index = |name: &FieldName| {
            fields
                .find(name)
                .ok_or_else(|| vortex_err!("Field {name} not found in struct layout"))
        };

        let (names, indices) = if expr.is_root() {
            (fields.names().clone(), (0..fields.nfields()).collect())
        } else if let Some(FieldSelection::Include(names)) = expr.as_opt::<Select>()
            && expr.child(0).is_root()
        {
            let indices = names.iter().map(field_index).try_collect()?;
            (names.clone(), indices)
        } else if let Some(pack) = expr.as_opt::<Pack>()
            && !pack.nullability.is_nullable()
        {
            let Some(indices) = expr
                .children()
                .iter()
                .map(|child| {
                    child
                        .as_opt::<GetItem>()
                        .filter(|_| child.child(0).is_root())
                        .map(field_index)
                })
                .collect::<Option<VortexResult<Vec<_>>>>()
            else {
                return Ok(None);
            };
            (pack.names.clone(), indices?)
        } else {
            return Ok(None);
        };

        let fields = indices
            .into_iter()
            .map(|idx: usize| -> VortexResult<_> {
                let dtype = fields
                    .field_by_index(idx)
                    .ok_or_else(|| vortex_err!("Field index {idx} out of bounds"))?;
                Ok((idx, BoundExpression::new_root(dtype)))
            })
            .try_collect()?;
        Ok(Some(DirectProjection { names, fields }))
    }

    /// Evaluate a [`DirectProjection`] by reading each field and packing the results.
    fn direct_projection_evaluation(
        &self,
        direct: &DirectProjection,
        row_range: &Range<u64>,
        mask_fut: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let field_futs: Vec<_> = direct
            .fields
            .iter()
            .map(|(idx, expr)| {
                self.field_reader_by_index(*idx)?
                    .projection_evaluation(row_range, expr, mask_fut.clone())
                    .map_err(|err| {
                        err.with_context(format!(
                            "While evaluating projection of field {}",
                            self.struct_fields()
                                .field_name(*idx)
                                .vortex_expect("field index is in bounds")
                        ))
                    })
            })
            .try_collect()?;
        let names = direct.names.clone();

        Ok(async move {
            let (fields, mask) = try_join!(try_join_all(field_futs), mask_fut)?;
            Ok(
                StructArray::try_new(names, fields, mask.true_count(), Validity::NonNullable)?
                    .into_array(),
            )
        }
        .boxed())
    }
}

fn expanded_struct_root(
    root_dtype: &DType,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    let root = BoundExpression::new_root(root_dtype.clone());
    let children = fields
        .names()
        .iter()
        .map(|name| get_item(name.clone(), root.clone()))
        .collect::<Vec<_>>();
    let packed = pack(
        fields.names().iter().cloned().zip(children),
        Nullability::NonNullable,
    );
    if root_dtype.is_nullable() {
        Ok(mask(packed, is_not_null(root)))
    } else {
        Ok(packed)
    }
}

fn expand_struct_root(
    expr: BoundExpression,
    expanded_root: &BoundExpression,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
    let (packed, validity) = if expanded_root
        .as_scalar()
        .is_some_and(|scalar_fn| scalar_fn.is::<MaskFn>())
    {
        let children = expanded_root.children();
        (&children[0], Some(&children[1]))
    } else {
        (expanded_root, None)
    };
    let with_validity = |expr: BoundExpression| match validity {
        Some(validity) => mask(expr, validity.clone()),
        None => expr,
    };

    Ok(expr
        .transform_down(|node| {
            if node.is_root() {
                return Ok(Transformed {
                    value: expanded_root.clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            let Some(scalar_fn) = node.as_scalar() else {
                return Ok(Transformed::no(node));
            };
            if !node
                .children()
                .first()
                .is_some_and(BoundExpression::is_root)
            {
                return Ok(Transformed::no(node));
            }

            if let Some(field_name) = scalar_fn.as_opt::<GetItem>() {
                let idx = fields.find(field_name).ok_or_else(|| {
                    vortex_err!("Field {field_name} not found while expanding struct root")
                })?;
                let child = packed.children()[idx].clone();
                return Ok(Transformed {
                    value: with_validity(child),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                let names = selection.normalize_to_included_fields(fields.names())?;
                let children: Vec<_> = names
                    .iter()
                    .map(|name| {
                        let idx = fields.find(name).vortex_expect(
                            "normalized selection fields must exist in the struct root",
                        );
                        packed.children()[idx].clone()
                    })
                    .collect();
                return Ok(Transformed {
                    value: with_validity(pack(
                        names.into_iter().zip(children),
                        Nullability::NonNullable,
                    )),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            if let Some(paths) = scalar_fn.as_opt::<Project>() {
                let mut names = Vec::new();
                let mut children = Vec::new();
                for name in fields.names().iter() {
                    let matching: Vec<_> = paths
                        .0
                        .iter()
                        .filter(|path| path.first() == Some(name))
                        .collect();
                    if matching.is_empty() {
                        continue;
                    }
                    names.push(name.clone());
                    let descendants: Vec<_> = matching
                        .iter()
                        .filter_map(|path| path.get(1..))
                        .filter(|path| !path.is_empty())
                        .map(|path| path.to_vec())
                        .collect();
                    let idx = fields.find(name).vortex_expect("project path field exists");
                    let child = packed.children()[idx].clone();
                    if matching.iter().any(|path| path.len() == 1) || descendants.is_empty() {
                        children.push(child);
                    } else {
                        children.push(project_paths_bound(&descendants, child)?);
                    }
                }
                return Ok(Transformed {
                    value: with_validity(pack(
                        names.into_iter().zip(children),
                        Nullability::NonNullable,
                    )),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            let Some(validity) = validity else {
                return Ok(Transformed::no(node));
            };

            // You can alternatively call optimize_recursive but this is cheaper
            if scalar_fn.is::<IsNotNull>() {
                return Ok(Transformed {
                    value: validity.clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }
            if scalar_fn.is::<IsNull>() {
                return Ok(Transformed {
                    value: not(validity.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }
            Ok(Transformed::no(node))
        })?
        .into_inner())
}

// Same as make_bound_free_field_annotator, but handles is_not_null($) as the
// struct's own validity coordinate
fn make_struct_item_annotator(
    scope: &StructFields,
) -> impl AnnotationFn<BoundExpression, Annotation = Item> {
    let free_fields = make_bound_free_field_annotator(scope);
    move |expr: &BoundExpression| {
        if expr
            .as_scalar()
            .is_some_and(|scalar_fn| scalar_fn.is::<IsNotNull>())
            && expr.children()[0].is_root()
        {
            return vec![Item::Validity];
        }
        free_fields(expr).into_iter().map(Item::Field).collect()
    }
}

fn step_into_struct_field(
    expr: BoundExpression,
    item: &Item,
    field_dtype: DType,
) -> VortexResult<BoundExpression> {
    Ok(expr
        .transform_down(|node| {
            let is_field_access = node.as_scalar().is_some_and(|scalar_fn| match item {
                Item::Field(field_name) => scalar_fn
                    .as_opt::<GetItem>()
                    .is_some_and(|name| name == field_name),
                Item::Validity => scalar_fn.is::<IsNotNull>(),
            }) && node.children()[0].is_root();

            if is_field_access {
                Ok(Transformed {
                    value: BoundExpression::new_root(field_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

/// When partitioning an expression, in the case it only has a single partition we can avoid
/// some cost and just delegate to the child reader directly.
// TODO(joe): this is a duplicate of the Partitioned enum in arrays/expr/vtable/rules
#[derive(Clone)]
enum Partitioned {
    /// An expression which only operates over a single coordinate
    Single(Item, BoundExpression),
    /// An expression which operates over multiple coordinates
    Multi(Arc<BoundPartitionedExpr<Item>>),
    /// A projection that passes fields through unchanged, see
    /// [`StructReader::direct_projection`].
    Direct(Arc<DirectProjection>),
}

/// The fields read by a pass-through projection of a non-nullable struct.
struct DirectProjection {
    /// The output field names.
    names: FieldNames,
    /// The struct field index for each output field, with the root expression it is read with.
    fields: Vec<(usize, BoundExpression)>,
}

impl LayoutReader for StructReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // Register splits for the validity child, if there is one
        if let Some(validity_ref) = self.validity()? {
            validity_ref.register_splits(field_mask, split_range, splits)?;
        }

        self.layout.matching_fields(field_mask, |mask, idx| {
            self.field_reader_by_index(idx)?
                .register_splits(&[mask], split_range, splits)
        })?;

        // In the case of an empty struct, we need to register the end split. Pushed last so it
        // extends the final field's ascending run rather than starting a run of its own.
        splits.push(split_range.root_row_range().end);

        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Partition the expression into expressions that can be evaluated over individual fields
        match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) => {
                let reader = self.field_reader(name)?;
                reader
                    .pruning_evaluation(row_range, partition, mask)
                    .map_err(|err| {
                        err.with_context(format!(
                            "While evaluating pruning filter partition {name}"
                        ))
                    })
            }
            Partitioned::Multi(_) | Partitioned::Direct(_) => {
                // TODO(ngates): if all partitions are boolean, we can use a pruning evaluation. Otherwise
                //  there's not much we can do? Maybe... it's complicated...
                Ok(MaskFuture::ready(mask))
            }
        }
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        // Partition the expression into expressions that can be evaluated over individual fields
        match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) => {
                let reader = self.field_reader(name)?;
                reader
                    .filter_evaluation(row_range, partition, mask)
                    .map_err(|err| {
                        err.with_context(format!("While evaluating filter partition {name}"))
                    })
            }
            Partitioned::Multi(partitioned) => Arc::clone(partitioned).into_mask_future(
                mask,
                |name, expr, mask| {
                    let reader = self.field_reader(name)?;
                    reader
                        .filter_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!("While evaluating filter partition {name}"))
                        })
                },
                |name, expr, mask| {
                    let reader = self.field_reader(name)?;
                    reader
                        .projection_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {name}"
                            ))
                        })
                },
                self.session.clone(),
            ),
            Partitioned::Direct(_) => {
                vortex_bail!(
                    "Cannot evaluate struct projection {expr} as a filter over {}",
                    self.name
                )
            }
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask_fut: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // Partition the expression into expressions that can be evaluated over individual fields
        match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) => {
                let reader = self.field_reader(name)?;
                reader
                    .projection_evaluation(row_range, partition, mask_fut)
                    .map_err(|err| {
                        err.with_context(format!("While evaluating projection partition {name}"))
                    })
            }
            Partitioned::Multi(partitioned) => {
                Arc::clone(partitioned).into_array_future(mask_fut, |name, expr, mask| {
                    let reader = self.field_reader(name)?;
                    reader
                        .projection_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {name}"
                            ))
                        })
                })
            }
            Partitioned::Direct(direct) => {
                self.direct_projection_evaluation(direct, row_range, mask_fut)
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::fixture;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::RESERVED_VALIDITY_FIELD_NAME;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::or;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::expr::select;
    use vortex_array::expr::select_exclude;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;

    use crate::LayoutReaderRef;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;
    use crate::test::new_session;

    #[fixture]
    fn empty_struct() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_new(
                        Vec::<FieldName>::new().into(),
                        vec![],
                        5,
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::from_fields(
                        [
                            ("a", buffer![7, 2, 3].into_array()),
                            ("b", buffer![4, 5, 6].into_array()),
                            ("c", buffer![4, 5, 6].into_array()),
                        ]
                        .as_slice(),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[fixture]
    /// Create a chunked layout with three chunks of primitive arrays.
    fn null_struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_from_iter_with_validity(
                        [
                            ("a", buffer![7, 2, 3].into_array()),
                            ("b", buffer![4, 5, 6].into_array()),
                            ("c", buffer![4, 5, 6].into_array()),
                        ],
                        Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    /// Writes a nested struct layout with the following values:
    ///
    /// |        a         |
    /// |------------------|
    /// |`{"b": {"c": 4 }}`|
    /// |     `NULL`       |
    /// |`{"b": {"c": 6 }}`|
    #[fixture]
    fn nested_struct_layout() -> (Arc<dyn SegmentSource>, LayoutRef) {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::try_from_iter_with_validity(
                        [(
                            "a",
                            StructArray::try_from_iter_with_validity(
                                [(
                                    "b",
                                    StructArray::try_from_iter_with_validity(
                                        [("c", buffer![4, 5, 6].into_array())],
                                        Validity::NonNullable,
                                    )
                                    .unwrap()
                                    .into_array(),
                                )],
                                Validity::Array(
                                    BoolArray::from_iter([true, false, true]).into_array(),
                                ),
                            )
                            .unwrap()
                            .into_array(),
                        )],
                        Validity::NonNullable,
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        (segments, layout)
    }

    #[rstest]
    fn test_struct_layout_or(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let filt = or(
            eq(col("a"), lit(7)),
            or(eq(col("b"), lit(5)), eq(col("a"), lit(3))),
        )
        .bind(reader.dtype())
        .unwrap();
        let result = block_on(|_| {
            reader
                .filter_evaluation(&(0..3), &filt, MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        assert_eq!(result, Mask::from_iter([true, true, true]));
    }

    #[rstest]
    fn test_struct_layout(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = gt(get_item("a", root()), get_item("b", root()))
            .bind(reader.dtype())
            .unwrap();
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })
        .unwrap();
        let expected = BoolArray::from_iter([true, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[rstest]
    fn test_struct_layout_row_mask(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = gt(get_item("a", root()), get_item("b", root()))
            .bind(reader.dtype())
            .unwrap();
        let result = block_on(|_| {
            reader
                .projection_evaluation(
                    &(0..3),
                    &expr,
                    MaskFuture::ready(Mask::from_iter([true, true, false])),
                )
                .unwrap()
        })
        .unwrap();

        let expected = BoolArray::from_iter([true, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[rstest]
    fn test_struct_layout_select(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = array_session().create_execution_ctx();
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = pack(
            [("a", get_item("a", root())), ("b", get_item("b", root()))],
            Nullability::NonNullable,
        )
        .bind(reader.dtype())
        .unwrap();
        let result = block_on(|_| {
            reader
                .projection_evaluation(
                    &(0..3),
                    &expr,
                    // Take rows 0 and 1, skip row 2, and anything after that
                    MaskFuture::ready(Mask::from_iter([true, true, false])),
                )
                .unwrap()
        })
        .unwrap();

        assert_eq!(result.len(), 2);

        let expected_a = PrimitiveArray::from_iter([7i32, 2]);
        let result_struct_a = result.clone().execute::<StructArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            result_struct_a.unmasked_field_by_name("a").unwrap(),
            expected_a,
            &mut ctx
        );

        let expected_b = PrimitiveArray::from_iter([4i32, 5]);
        let result_struct_b = result.execute::<StructArray>(&mut ctx).unwrap();
        assert_arrays_eq!(
            result_struct_b.unmasked_field_by_name("b").unwrap(),
            expected_b,
            &mut ctx
        );
    }

    /// Field pass-through projections take the direct path; the rest must match it.
    #[rstest]
    #[case::root(root(), &[("a", [7, 2]), ("b", [4, 5]), ("c", [4, 5])])]
    #[case::select(select(["c", "a"], root()), &[("c", [4, 5]), ("a", [7, 2])])]
    #[case::select_exclude(select_exclude(["b"], root()), &[("a", [7, 2]), ("c", [4, 5])])]
    #[case::pack_renamed(
        pack([("x", get_item("b", root())), ("y", get_item("b", root()))], Nullability::NonNullable),
        &[("x", [4, 5]), ("y", [4, 5])],
    )]
    #[case::pack_nullable(
        pack([("a", get_item("a", root()))], Nullability::Nullable),
        &[("a", [7, 2])],
    )]
    fn test_struct_layout_field_projection(
        #[from(struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
        #[case] expr: Expression,
        #[case] expected: &[(&str, [i32; 2])],
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let expr = expr.bind(reader.dtype())?;
        let future = reader.projection_evaluation(
            &(0..3),
            &expr,
            MaskFuture::ready(Mask::from_iter([true, true, false])),
        )?;
        let result = block_on(|_| future)?;

        assert_eq!(result.dtype(), expr.dtype());
        assert_eq!(result.len(), 2);
        let result = result.execute::<StructArray>(&mut ctx)?;
        assert_eq!(result.names().len(), expected.len());
        for (idx, (name, values)) in expected.iter().enumerate() {
            assert_eq!(result.names()[idx].as_ref(), *name);
            assert_arrays_eq!(
                result.unmasked_field(idx),
                PrimitiveArray::from_iter(values.iter().copied()),
                &mut ctx
            );
        }
        Ok(())
    }

    #[rstest]
    fn test_struct_layout_nulls(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        // Read the layout source from the top.
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = get_item("a", root()).bind(reader.dtype()).unwrap();
        let project = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
            .unwrap();

        let result = block_on(move |_| project).unwrap();
        // Result should be the primitive array with a single field.
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I32, Nullability::Nullable)
        );

        // ...and the result is masked with the validity of the parent StructArray
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::null(result.dtype().clone()),
        );
        assert_nth_scalar!(result, 1, 2, &mut ctx);
        assert_nth_scalar!(result, 2, 3, &mut ctx);
    }

    #[rstest]
    fn test_is_not_null_over_nullable_struct(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let filter = is_not_null(root()).bind(reader.dtype())?;
        let result = block_on(|_| {
            reader
                .filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))
                .unwrap()
        })?;
        assert_eq!(result, Mask::from_iter([false, true, true]));
        Ok(())
    }

    #[rstest]
    fn test_predicate_over_nullable_struct(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let filter = eq(get_item("a", root()), lit(7)).bind(reader.dtype())?;
        let result = block_on(|_| {
            reader
                .filter_evaluation(&(0..3), &filter, MaskFuture::new_true(3))
                .unwrap()
        })?;
        assert_eq!(result, Mask::from_iter([false, false, false]));
        Ok(())
    }

    #[rstest]
    fn test_is_null_projection_over_nullable_struct(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let expr = is_null(root()).bind(reader.dtype())?;
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })?;
        let expected = BoolArray::from_iter([true, false, false]);
        assert_arrays_eq!(result, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    fn test_constant_projection_over_nullable_struct(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let expr = pack([("x", lit(1))], false.into()).bind(reader.dtype())?;

        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })?;
        let expected_x_dtype = DType::Primitive(PType::I32, false.into());
        let expected_fields = StructFields::from_iter([("x", expected_x_dtype)]);
        let expected_dtype = DType::Struct(expected_fields, false.into());
        assert_eq!(result.dtype(), &expected_dtype);

        let result = result.execute::<StructArray>(&mut ctx)?;
        let field = result.unmasked_field_by_name("x")?;
        let expected = PrimitiveArray::from_iter([1i32, 1, 1]);
        assert_arrays_eq!(field, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    fn test_root_projection_over_nullable_struct(
        #[from(null_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let expr = root().bind(reader.dtype())?;
        let result = block_on(|_| {
            reader
                .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))
                .unwrap()
        })?;
        assert_eq!(result.dtype(), reader.dtype());

        let mut probe = result.probe();
        let scalar = probe.execute_scalar(0, &mut ctx)?;
        assert!(scalar.is_null());

        let scalar = probe.execute_scalar(1, &mut ctx)?;
        let field = scalar.as_struct().field_by_idx(0).unwrap();
        assert_eq!(field, Scalar::primitive(2, false.into()));
        Ok(())
    }

    #[rstest]
    #[case(
        Validity::Array(BoolArray::from_iter([false, true, true]).into_array()),
        true
    )]
    #[case(Validity::NonNullable, false)]
    fn test_validity_like_field_name(
        #[case] validity: Validity,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let fields = [("__vortex_validity", buffer![10, 20, 30].into_array())];
        let array = StructArray::try_from_iter_with_validity(fields, validity)?;

        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| {
            let session = new_session().with_handle(handle);
            async move {
                let array = array.into_array().to_array_stream().sequenced(ptr);
                strategy
                    .write_stream(ctx.into(), segments2, array, eof, &session)
                    .await
            }
        })?;

        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;
        let expr = get_item("__vortex_validity", root()).bind(reader.dtype())?;
        let future = reader.projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?;
        let result = block_on(move |_| future)?;

        let mut ctx = SESSION.create_execution_ctx();
        if nullable {
            assert_eq!(
                result.dtype(),
                &DType::Primitive(PType::I32, Nullability::Nullable)
            );
            assert!(result.execute_scalar(0, &mut ctx)?.is_null());
            assert_nth_scalar!(result, 1, 20, &mut ctx);
            assert_nth_scalar!(result, 2, 30, &mut ctx);
        } else {
            let expected = PrimitiveArray::from_iter([10i32, 20, 30]);
            assert_arrays_eq!(result, expected, &mut ctx);
        }
        Ok(())
    }

    /// Builds `$.a` of [`nested_validity_array`] with the given validity.
    ///
    /// `$.a.b` is null at row 2 and `$.a.b.c` is null at row 3.
    fn nested_validity_a(validity: [bool; 6]) -> VortexResult<StructArray> {
        let b = StructArray::try_from_iter_with_validity(
            [
                (
                    "c",
                    PrimitiveArray::from_option_iter([
                        Some(1i32),
                        Some(2),
                        Some(3),
                        None,
                        Some(5),
                        Some(6),
                    ])
                    .into_array(),
                ),
                ("d", buffer![10i32, 20, 30, 40, 50, 60].into_array()),
            ],
            Validity::Array(
                BoolArray::from_iter([true, true, false, true, true, true]).into_array(),
            ),
        )?;
        StructArray::try_from_iter_with_validity(
            [
                ("e", buffer![100i32, 101, 102, 103, 104, 105].into_array()),
                ("b", b.into_array()),
            ],
            Validity::Array(BoolArray::from_iter(validity).into_array()),
        )
    }

    /// A struct where each row is null at a different level of nesting:
    ///
    /// | row | `$`               | `$.a` | `$.a.b` | `$.a.b.c` | `$.a.b.d` | `$.a.e` | `$.x` |
    /// |-----|-------------------|-------|---------|-----------|-----------|---------|-------|
    /// | 0   | null if nullable  |       |         | 1         | 10        | 100     | 0     |
    /// | 1   |                   | null  |         | 2         | 20        | 101     | 1     |
    /// | 2   |                   |       | null    | 3         | 30        | 102     | 2     |
    /// | 3   |                   |       |         | null      | 40        | 103     | 3     |
    /// | 4   |                   |       |         | 5         | 50        | 104     | 4     |
    /// | 5   |                   |       |         | 6         | 60        | 105     | 5     |
    fn nested_validity_array(root_nullable: bool) -> VortexResult<ArrayRef> {
        let a = nested_validity_a([true, false, true, true, true, true])?;
        let validity = if root_nullable {
            Validity::Array(
                BoolArray::from_iter([false, true, true, true, true, true]).into_array(),
            )
        } else {
            Validity::NonNullable
        };
        Ok(StructArray::try_from_iter_with_validity(
            [
                ("x", buffer![0i32, 1, 2, 3, 4, 5].into_array()),
                ("a", a.into_array()),
            ],
            validity,
        )?
        .into_array())
    }

    fn nested_validity_reader(root_nullable: bool) -> VortexResult<LayoutReaderRef> {
        let array = nested_validity_array(root_nullable)?;
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| {
            let session = new_session().with_handle(handle);
            async move {
                let stream = array.to_array_stream().sequenced(ptr);
                strategy
                    .write_stream(ctx.into(), segments2, stream, eof, &session)
                    .await
            }
        })?;
        layout.new_reader("".into(), segments, &SESSION, &Default::default())
    }

    fn nested_validity_project(
        root_nullable: bool,
        expr: Expression,
        mask: Mask,
    ) -> VortexResult<ArrayRef> {
        let reader = nested_validity_reader(root_nullable)?;
        let expr = expr.bind(reader.dtype())?;
        let future = reader.projection_evaluation(&(0..6), &expr, MaskFuture::ready(mask))?;
        block_on(move |_| future)
    }

    fn nested_validity_filter(root_nullable: bool, expr: Expression) -> VortexResult<Mask> {
        let reader = nested_validity_reader(root_nullable)?;
        let expr = expr.bind(reader.dtype())?;
        let future = reader.filter_evaluation(&(0..6), &expr, MaskFuture::new_true(6))?;
        block_on(move |_| future)
    }

    fn a_b() -> Expression {
        get_item("b", col("a"))
    }

    #[rstest]
    #[case::leaf_nullable(
        get_item("c", a_b()),
        [Some(1), None, None, None, Some(5), Some(6)]
    )]
    #[case::leaf_non_nullable(
        get_item("d", a_b()),
        [Some(10), None, None, Some(40), Some(50), Some(60)]
    )]
    #[case::middle(
        get_item("e", col("a")),
        [Some(100), None, Some(102), Some(103), Some(104), Some(105)]
    )]
    fn test_nested_validity_projection(
        #[values(false, true)] root_nullable: bool,
        #[case] expr: Expression,
        #[case] mut expected: [Option<i32>; 6],
    ) -> VortexResult<()> {
        if root_nullable {
            expected[0] = None;
        }
        let result = nested_validity_project(root_nullable, expr, Mask::new_true(6))?;
        let expected = PrimitiveArray::from_option_iter(expected);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_row_mask(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let mask = Mask::from_iter([true, false, true, false, true, true]);
        let result = nested_validity_project(root_nullable, get_item("c", a_b()), mask)?;
        let first = (!root_nullable).then_some(1i32);
        let expected = PrimitiveArray::from_option_iter([first, None, Some(5), Some(6)]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_top_level_field(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let result = nested_validity_project(root_nullable, col("x"), Mask::new_true(6))?;
        let expected = if root_nullable {
            PrimitiveArray::from_option_iter([None, Some(1i32), Some(2), Some(3), Some(4), Some(5)])
        } else {
            PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5])
        };
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_root(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let result = nested_validity_project(root_nullable, root(), Mask::new_true(6))?;
        let expected = nested_validity_array(root_nullable)?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_nested_struct(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let result = nested_validity_project(root_nullable, col("a"), Mask::new_true(6))?;
        let expected = nested_validity_a([!root_nullable, false, true, true, true, true])?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_select(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let expr = select(["c"], a_b());
        let result = nested_validity_project(root_nullable, expr, Mask::new_true(6))?;
        let c = PrimitiveArray::from_option_iter([
            Some(1i32),
            Some(2),
            Some(3),
            None,
            Some(5),
            Some(6),
        ]);
        let validity = BoolArray::from_iter([!root_nullable, false, false, true, true, true]);
        let expected = StructArray::try_from_iter_with_validity(
            [("c", c.into_array())],
            Validity::Array(validity.into_array()),
        )?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_pack_across_levels(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let expr = pack(
            [
                ("c", get_item("c", a_b())),
                ("e", get_item("e", col("a"))),
                ("x", col("x")),
            ],
            Nullability::NonNullable,
        );
        let result = nested_validity_project(root_nullable, expr, Mask::new_true(6))?;

        let first = |value: i32| (!root_nullable).then_some(value);
        let c = PrimitiveArray::from_option_iter([first(1), None, None, None, Some(5), Some(6)]);
        let e = PrimitiveArray::from_option_iter([
            first(100),
            None,
            Some(102),
            Some(103),
            Some(104),
            Some(105),
        ]);
        let x = if root_nullable {
            PrimitiveArray::from_option_iter([None, Some(1i32), Some(2), Some(3), Some(4), Some(5)])
        } else {
            PrimitiveArray::from_iter([0i32, 1, 2, 3, 4, 5])
        };
        let expected = StructArray::try_from_iter_with_validity(
            [
                ("c", c.into_array()),
                ("e", e.into_array()),
                ("x", x.into_array()),
            ],
            Validity::NonNullable,
        )?;
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_nested_validity_projection_is_null(
        #[values(false, true)] root_nullable: bool,
    ) -> VortexResult<()> {
        let result = nested_validity_project(root_nullable, is_null(a_b()), Mask::new_true(6))?;
        let expected = BoolArray::from_iter([root_nullable, true, true, false, false, false]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    #[case::leaf_predicate(
        gt(get_item("c", a_b()), lit(0)),
        [true, false, false, false, true, true],
        [false, false, false, false, true, true]
    )]
    #[case::leaf_is_not_null(
        is_not_null(get_item("c", a_b())),
        [true, false, false, false, true, true],
        [false, false, false, false, true, true]
    )]
    #[case::leaf_is_null(
        is_null(get_item("c", a_b())),
        [false, true, true, true, false, false],
        [true, true, true, true, false, false]
    )]
    #[case::inner_is_null(
        is_null(a_b()),
        [false, true, true, false, false, false],
        [true, true, true, false, false, false]
    )]
    #[case::inner_is_not_null(
        is_not_null(a_b()),
        [true, false, false, true, true, true],
        [false, false, false, true, true, true]
    )]
    #[case::outer_is_null(
        is_null(col("a")),
        [false, true, false, false, false, false],
        [true, true, false, false, false, false]
    )]
    #[case::root_is_null(
        is_null(root()),
        [false, false, false, false, false, false],
        [true, false, false, false, false, false]
    )]
    #[case::root_is_not_null(
        is_not_null(root()),
        [true, true, true, true, true, true],
        [false, true, true, true, true, true]
    )]
    #[case::top_level_or_leaf(
        or(eq(col("x"), lit(0)), gt(get_item("c", a_b()), lit(5))),
        [true, false, false, false, false, true],
        [false, false, false, false, false, true]
    )]
    #[case::middle_or_leaf(
        or(gt(get_item("e", col("a")), lit(99)), gt(get_item("d", a_b()), lit(25))),
        [true, false, true, true, true, true],
        [false, false, true, true, true, true]
    )]
    #[case::middle_and_inner_is_null(
        and(gt(get_item("e", col("a")), lit(99)), is_null(a_b())),
        [false, false, true, false, false, false],
        [false, false, true, false, false, false]
    )]
    #[case::leaf_compared_to_top_level(
        gt(get_item("d", a_b()), col("x")),
        [true, false, false, true, true, true],
        [false, false, false, true, true, true]
    )]
    fn test_nested_validity_filter(
        #[values(false, true)] root_nullable: bool,
        #[case] expr: Expression,
        #[case] non_nullable_root: [bool; 6],
        #[case] nullable_root: [bool; 6],
    ) -> VortexResult<()> {
        let expected = if root_nullable {
            nullable_root
        } else {
            non_nullable_root
        };
        let result = nested_validity_filter(root_nullable, expr)?;
        assert_eq!(result, Mask::from_iter(expected));
        Ok(())
    }

    /// A field named [`RESERVED_VALIDITY_FIELD_NAME`] clashes with the validity partition of a
    /// nullable struct. Such names are reserved and not supported.
    #[test]
    #[should_panic]
    fn test_validity_partition_name_clash() {
        validity_partition_name_clash().unwrap();
    }

    fn validity_partition_name_clash() -> VortexResult<()> {
        let validity = Validity::Array(BoolArray::from_iter([false, true]).into_array());
        let array = StructArray::try_from_iter_with_validity(
            [(RESERVED_VALIDITY_FIELD_NAME, buffer![1i32, 2])],
            validity,
        )?;

        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| {
            let session = new_session().with_handle(handle);
            async move {
                let stream = array.into_array().to_array_stream().sequenced(ptr);
                let flat = Arc::new(FlatLayoutStrategy::default());
                TableStrategy::new(Arc::clone(&flat) as _, flat)
                    .write_stream(
                        ArrayContext::empty().into(),
                        segments2,
                        stream,
                        eof,
                        &session,
                    )
                    .await
            }
        })?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &Default::default())?;

        let expr = col(RESERVED_VALIDITY_FIELD_NAME).bind(reader.dtype())?;
        let future = reader.projection_evaluation(&(0..2), &expr, MaskFuture::new_true(2))?;
        let result = block_on(move |_| future)?;

        let expected = PrimitiveArray::from_option_iter([None, Some(2i32)]);
        assert_arrays_eq!(result, expected, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[rstest]
    fn test_struct_layout_nested(
        #[from(nested_struct_layout)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        // Project out the nested struct field.
        // The projection should preserve the nulls of the `b` struct when we select out the
        // child column `c`.
        let expr = select(
            vec![FieldName::from("c")],
            get_item("b", get_item("a", root())),
        );
        let result = block_on(move |handle| {
            let session = new_session().with_handle(handle);
            async move {
                let reader =
                    layout.new_reader("".into(), segments, &session, &Default::default())?;
                let expr = expr.bind(reader.dtype())?;
                reader
                    .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
                    .await
            }
        })
        .unwrap();

        // The result is a nullable struct (because root.a.b is nullable) with a non-nullable
        // field "c" (because the original field was non-nullable).
        assert_eq!(
            result.dtype(),
            &DType::Struct(
                StructFields::from_iter([(
                    "c",
                    DType::Primitive(PType::I32, Nullability::NonNullable)
                )]),
                Nullability::Nullable,
            )
        );

        // Row 0: struct is valid, field "c" is 4.
        assert_eq!(
            result
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap()
                .as_struct()
                .field_by_idx(0)
                .unwrap(),
            Scalar::primitive(4, Nullability::NonNullable)
        );

        // Row 1: struct is null (because root.a.b was null at this row).
        assert!(
            result
                .execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap()
                .as_struct()
                .is_null()
        );

        // Row 2: struct is valid, field "c" is 6.
        assert_eq!(
            result
                .execute_scalar(2, &mut array_session().create_execution_ctx())
                .unwrap()
                .as_struct()
                .field_by_idx(0)
                .unwrap(),
            Scalar::primitive(6, Nullability::NonNullable)
        );
    }

    #[rstest]
    fn test_empty_struct(
        #[from(empty_struct)] (segments, layout): (Arc<dyn SegmentSource>, LayoutRef),
    ) {
        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();
        let expr = pack(Vec::<(String, Expression)>::new(), Nullability::Nullable)
            .bind(reader.dtype())
            .unwrap();

        let project = reader
            .projection_evaluation(&(0..5), &expr, MaskFuture::new_true(5))
            .unwrap();

        let result = block_on(move |_| project).unwrap();
        assert!(result.dtype().is_struct());

        assert_eq!(result.len(), 5);
    }

    /// Regression test for https://github.com/vortex-data/vortex/issues/7808
    ///
    /// A filter expression whose DType is incompatible with the scanned schema
    /// (e.g. comparing a u8 column to an i32 literal) must return an error, not panic.
    #[test]
    fn test_struct_filter_dtype_mismatch_fails_binding() {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );
        let segments2 = Arc::<TestSegments>::clone(&segments);
        let layout = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            strategy
                .write_stream(
                    ctx.into(),
                    segments2,
                    StructArray::from_fields(
                        [
                            ("age", buffer![7u8, 2, 3].into_array()),
                            ("score", buffer![4u8, 5, 6].into_array()),
                        ]
                        .as_slice(),
                    )
                    .unwrap()
                    .into_array()
                    .to_array_stream()
                    .sequenced(ptr),
                    eof,
                    &session,
                )
                .await
        })
        .unwrap();

        let reader = layout
            .new_reader("".into(), segments, &SESSION, &Default::default())
            .unwrap();

        // DType mismatch: "age" is u8 but literal is i32
        let filt = eq(col("age"), lit(67i32));

        let result = filt.bind(reader.dtype());
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("Cannot compare different DTypes"), "{err}");
    }
}
