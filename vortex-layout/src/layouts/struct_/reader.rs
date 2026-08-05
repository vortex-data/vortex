// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::try_join;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::make_bound_free_field_annotator;
use vortex_array::expr::root;
use vortex_array::expr::transform::BoundPartitionedExpr;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::merge::Merge;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::pack::PackOptions;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
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

pub struct StructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,

    /// A `pack` expression that holds each individual field of the root DType. This expansion
    /// ensures we can correctly partition expressions over the fields of the struct.
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
            partitioned_expr_cache: Default::default(),
        })
    }

    /// Return the [`StructFields`] of this layout.
    fn struct_fields(&self) -> &StructFields {
        self.layout.struct_fields()
    }

    /// Return the child reader for the field.
    fn field_reader(&self, name: &FieldName) -> VortexResult<&LayoutReaderRef> {
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
        // First, we expand the root scope into the fields of the struct to ensure
        // that partitioning works correctly.
        let expr =
            expand_struct_root(expr.clone(), &self.expanded_root_expr, self.struct_fields())?;

        // Partition the expression into expressions that can be evaluated over individual fields
        let mut partitioned = partition_bound(
            expr.clone(),
            make_bound_free_field_annotator(
                self.dtype()
                    .as_struct_fields_opt()
                    .vortex_expect("We know it's a struct DType"),
            ),
        )?;

        if partitioned.partitions.len() == 1 {
            // If there's only one partition, we step into the field scope of the original
            // expression by replacing any `$.a` with `$`.
            return Ok(Partitioned::Single(
                partitioned.partition_names[0].clone(),
                step_into_struct_field(
                    expr,
                    &partitioned.partition_names[0],
                    self.field_reader(&partitioned.partition_names[0])?
                        .dtype()
                        .clone(),
                )?,
            ));
        }

        // We now need to process the partitioned expressions to rewrite the root scope
        // to be that of the field, rather than the struct. In other words, "stepping in"
        // to the field scope.
        let partitions = partitioned
            .partitions
            .iter()
            .zip_eq(partitioned.partition_names.iter())
            .map(|(expr, name)| {
                step_into_struct_field(expr.clone(), name, self.field_reader(name)?.dtype().clone())
            })
            .try_collect::<_, Vec<_>, _>()?
            .into_boxed_slice();
        partitioned.replace_partitions(partitions)?;

        Ok(Partitioned::Multi(Arc::new(partitioned)))
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
        .map(|name| BoundExpression::try_new(GetItem.bind(name.clone()), [root.clone()]))
        .try_collect()?;
    bound_pack(fields.names().clone(), children)
}

fn expand_struct_root(
    expr: BoundExpression,
    expanded_root: &BoundExpression,
    fields: &StructFields,
) -> VortexResult<BoundExpression> {
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
                return Ok(Transformed {
                    value: expanded_root.children()[idx].clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                let names = selection.normalize_to_included_fields(fields.names())?;
                let children = names
                    .iter()
                    .map(|name| {
                        let idx = fields.find(name).vortex_expect(
                            "normalized selection fields must exist in the struct root",
                        );
                        expanded_root.children()[idx].clone()
                    })
                    .collect();
                return Ok(Transformed {
                    value: bound_pack(names, children)?,
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }

            Ok(Transformed::no(node))
        })?
        .into_inner())
}

fn step_into_struct_field(
    expr: BoundExpression,
    field_name: &FieldName,
    field_dtype: DType,
) -> VortexResult<BoundExpression> {
    Ok(expr
        .transform_down(|node| {
            let is_field_access = node
                .as_scalar()
                .and_then(|scalar_fn| scalar_fn.as_opt::<GetItem>())
                .is_some_and(|name| name == field_name)
                && node.children()[0].is_root();

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

fn bound_pack(names: FieldNames, children: Vec<BoundExpression>) -> VortexResult<BoundExpression> {
    BoundExpression::try_new(
        Pack.bind(PackOptions {
            names,
            nullability: Nullability::NonNullable,
        }),
        children,
    )
}

fn is_pack_or_merge(expr: &BoundExpression) -> bool {
    expr.as_scalar()
        .is_some_and(|scalar_fn| scalar_fn.is::<Pack>() || scalar_fn.is::<Merge>())
}

/// When partitioning an expression, in the case it only has a single partition we can avoid
/// some cost and just delegate to the child reader directly.
// TODO(joe): this is a duplicate of the Partitioned enum in arrays/expr/vtable/rules
#[derive(Clone)]
enum Partitioned {
    /// An expression which only operates over a single field
    Single(FieldName, BoundExpression),
    /// An expression which operates over multiple fields
    Multi(Arc<BoundPartitionedExpr<FieldName>>),
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
        // In the case of an empty struct, we need to register the end split.
        splits.push(split_range.root_row_range().end);

        // Register splits for the validity child, if there is one
        if let Some(validity_ref) = self.validity()? {
            validity_ref.register_splits(field_mask, split_range, splits)?;
        }

        self.layout.matching_fields(field_mask, |mask, idx| {
            self.field_reader_by_index(idx)?
                .register_splits(&[mask], split_range, splits)
        })
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
            Partitioned::Multi(_) => {
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
        }
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask_fut: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let validity_fut = self
            .validity()?
            .map(|reader| {
                let root = root().bind(reader.dtype())?;
                reader.projection_evaluation(row_range, &root, mask_fut.clone())
            })
            .transpose()?;

        // Partition the expression into expressions that can be evaluated over individual fields
        let (projected, is_pack_merge) = match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) => {
                let reader = self.field_reader(name)?;
                (
                    reader
                        .projection_evaluation(row_range, partition, mask_fut)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {name}"
                            ))
                        })?,
                    is_pack_or_merge(partition),
                )
            }

            Partitioned::Multi(partitioned) => (
                Arc::clone(partitioned).into_array_future(mask_fut, |name, expr, mask| {
                    let reader = self.field_reader(name)?;
                    reader
                        .projection_evaluation(row_range, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {name}"
                            ))
                        })
                })?,
                is_pack_or_merge(&partitioned.root),
            ),
        };

        let session = self.session.clone();
        Ok(Box::pin(async move {
            if let Some(validity_fut) = validity_fut {
                let (array, validity) = try_join!(projected, validity_fut)?;

                // If root expression was a pack, then we apply the validity to each child field
                if is_pack_merge {
                    let mut ctx = session.create_execution_ctx();
                    let struct_array = array.execute::<StructArray>(&mut ctx)?;
                    let masked_fields: Vec<ArrayRef> = struct_array
                        .iter_unmasked_fields()
                        .map(|a| a.clone().mask(validity.clone()))
                        .try_collect()?;

                    Ok(StructArray::try_new(
                        struct_array.names().clone(),
                        masked_fields,
                        struct_array.len(),
                        struct_array.validity()?,
                    )?
                    .into_array())
                } else {
                    // If the root expression was not a pack or merge, e.g. if it's something like
                    // a get_item, then we apply the validity directly to the result
                    array.mask(validity)
                }
            } else {
                projected.await
            }
        }))
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
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::Expression;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::or;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::expr::select;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_mask::Mask;

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
