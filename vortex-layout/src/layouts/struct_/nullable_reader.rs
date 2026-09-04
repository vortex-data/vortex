// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use itertools::Itertools;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::ExactBoundExpr;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::pack;
use vortex_array::expr::make_bound_free_field_annotator;
use vortex_array::expr::root;
use vortex_array::expr::transform::BoundPartitionedExpr;
use vortex_array::expr::transform::partition_bound;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::get_item::GetItem;
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

/// Reader for nullable structs.
///
/// The serialized child layouts retain their declared dtypes. Parent validity is applied at
/// evaluation time so existing files remain readable and older readers can still read newly
/// written files. Strict predicates may continue to use child pruning because parent nulls only
/// remove logical rows. Non-strict predicates must see inherited validity before evaluation.
pub(super) struct NullableStructReader {
    layout: StructLayout,
    name: Arc<str>,
    lazy_children: LazyReaderChildren,
    session: VortexSession,
    expanded_root_expr: BoundExpression,
    field_lookup: Option<HashMap<FieldName, usize>>,
    partitioned_expr_cache: DashMap<ExactBoundExpr, Arc<OnceLock<Partitioned>>>,
}

impl NullableStructReader {
    pub(super) fn try_new(
        layout: StructLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: crate::LayoutReaderContext,
    ) -> VortexResult<Self> {
        let struct_dt = layout.struct_fields();
        let field_lookup = (struct_dt.nfields() > 80).then(|| {
            struct_dt
                .names()
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect()
        });
        let mut dtypes = Vec::with_capacity(struct_dt.nfields() + 1);
        let mut names = Vec::with_capacity(struct_dt.nfields() + 1);

        dtypes.push(DType::Bool(Nullability::NonNullable));
        names.push(Arc::from("validity"));
        dtypes.extend(struct_dt.fields());
        names.extend(struct_dt.names().iter().map(|name| Arc::clone(name.inner())));

        let lazy_children = LazyReaderChildren::new(
            Arc::clone(layout.children()),
            dtypes,
            names,
            Arc::clone(&segment_source),
            session.clone(),
            ctx,
        );
        let expanded_root_expr = expanded_struct_root(layout.dtype(), struct_dt)?;

        Ok(Self {
            layout,
            name,
            lazy_children,
            session,
            expanded_root_expr,
            field_lookup,
            partitioned_expr_cache: Default::default(),
        })
    }

    fn struct_fields(&self) -> &StructFields {
        self.layout.struct_fields()
    }

    fn field_reader(&self, name: &FieldName) -> VortexResult<&LayoutReaderRef> {
        let idx = self
            .field_lookup
            .as_ref()
            .and_then(|lookup| lookup.get(name).copied())
            .or_else(|| self.struct_fields().find(name))
            .ok_or_else(|| vortex_err!("Field {} not found in struct layout", name))?;
        self.field_reader_by_index(idx)
    }

    fn field_reader_by_index(&self, idx: usize) -> VortexResult<&LayoutReaderRef> {
        let child_index = self
            .layout
            .slot_to_child(idx + 1)
            .vortex_expect("struct field slot is always present");
        self.lazy_children.get(child_index)
    }

    fn validity(&self) -> VortexResult<&LayoutReaderRef> {
        let child_index = self
            .layout
            .slot_to_child(0)
            .vortex_expect("nullable struct validity slot is always present");
        self.lazy_children.get(child_index)
    }

    fn logical_field_dtype(&self, name: &FieldName) -> VortexResult<DType> {
        Ok(self.field_reader(name)?.dtype().as_nullable())
    }

    fn partition_expr(&self, expr: &BoundExpression) -> VortexResult<Partitioned> {
        let key = ExactBoundExpr(expr.clone());
        let cell = match self.partitioned_expr_cache.get(&key) {
            Some(entry) => Arc::clone(entry.value()),
            None => Arc::clone(
                self.partitioned_expr_cache
                    .entry(key)
                    .or_insert_with(|| Arc::new(OnceLock::new()))
                    .value(),
            ),
        };

        if let Some(value) = cell.get() {
            return Ok(value.clone());
        }

        let expr = expand_struct_root(
            expr.clone(),
            &self.expanded_root_expr,
            self.struct_fields(),
        )?;
        let mut partitioned = partition_bound(
            expr.clone(),
            make_bound_free_field_annotator(self.struct_fields()),
        )?;

        let result = if partitioned.partitions.len() == 1 {
            let name = partitioned.partition_names[0].clone();
            Partitioned::Single(
                name.clone(),
                step_into_struct_field(expr, &name, self.logical_field_dtype(&name)?)?,
            )
        } else {
            let partitions = partitioned
                .partitions
                .iter()
                .zip_eq(partitioned.partition_names.iter())
                .map(|(expr, name)| {
                    step_into_struct_field(expr.clone(), name, self.logical_field_dtype(name)?)
                })
                .try_collect::<_, Vec<_>, _>()?
                .into_boxed_slice();
            partitioned.replace_partitions(partitions)?;
            Partitioned::Multi(Arc::new(partitioned))
        };

        Ok(cell.get_or_init(|| result).clone())
    }

    fn validity_projection_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let reader = self.validity()?;
        let validity_root = root().bind(reader.dtype())?;
        reader.projection_evaluation(row_range, &validity_root, mask)
    }

    fn validity_filter_evaluation(
        &self,
        row_range: &Range<u64>,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let reader = self.validity()?;
        let validity_root = root().bind(reader.dtype())?;
        reader.filter_evaluation(row_range, &validity_root, mask)
    }

    fn field_projection_evaluation(
        &self,
        row_range: &Range<u64>,
        name: &FieldName,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let reader = self.field_reader(name)?;
        let validity = self.validity_projection_evaluation(row_range, mask.clone())?;

        if expression_is_strict(expr) {
            let physical_expr = retype_roots(expr.clone(), reader.dtype().clone())?;
            let field = reader.projection_evaluation(row_range, &physical_expr, mask)?;

            return Ok(Box::pin(async move {
                let (field, validity) = futures::try_join!(field, validity)?;
                field.mask(validity)
            }));
        }

        let field_root = root().bind(reader.dtype())?;
        let field = reader.projection_evaluation(row_range, &field_root, mask)?;
        let expr = expr.clone();

        Ok(Box::pin(async move {
            let (field, validity) = futures::try_join!(field, validity)?;
            field.mask(validity)?.apply_bound(&expr)
        }))
    }

    fn field_filter_evaluation(
        &self,
        row_range: &Range<u64>,
        name: &FieldName,
        expr: &BoundExpression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        if expression_is_strict(expr) {
            let reader = self.field_reader(name)?;
            let physical_expr = retype_roots(expr.clone(), reader.dtype().clone())?;
            let valid_mask = self.validity_filter_evaluation(row_range, mask)?;
            return reader.filter_evaluation(row_range, &physical_expr, valid_mask);
        }

        let input_mask = mask.clone();
        let result = self.field_projection_evaluation(row_range, name, expr, mask)?;
        let session = self.session.clone();
        let len = input_mask.len();

        Ok(MaskFuture::new(len, async move {
            let (input_mask, result) = futures::try_join!(input_mask, result)?;
            let mut ctx = session.create_execution_ctx();
            let result_mask = result.null_as_false().execute(&mut ctx)?;
            Ok(input_mask.intersect_by_rank(&result_mask))
        }))
    }
}

fn expression_is_strict(expr: &BoundExpression) -> bool {
    match expr.as_scalar() {
        Some(scalar_fn) => {
            scalar_fn.signature().is_strict()
                && expr.children().iter().all(expression_is_strict)
        }
        None => true,
    }
}

fn retype_roots(expr: BoundExpression, dtype: DType) -> VortexResult<BoundExpression> {
    Ok(expr
        .transform_up(|node| {
            if node.is_root() {
                Ok(Transformed::yes(BoundExpression::new_root(dtype.clone())))
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
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
    Ok(pack(
        fields.names().iter().cloned().zip(children),
        Nullability::NonNullable,
    ))
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
                    .collect::<Vec<_>>();
                return Ok(Transformed {
                    value: pack(names.into_iter().zip(children), Nullability::NonNullable),
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

#[derive(Clone)]
enum Partitioned {
    Single(FieldName, BoundExpression),
    Multi(Arc<BoundPartitionedExpr<FieldName>>),
}

impl LayoutReader for NullableStructReader {
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
        self.validity()?
            .register_splits(field_mask, split_range, splits)?;

        self.layout.matching_fields(field_mask, |mask, idx| {
            self.field_reader_by_index(idx)?
                .register_splits(&[mask], split_range, splits)
        })?;

        splits.push(split_range.root_row_range().end);
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &BoundExpression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) if expression_is_strict(partition) => {
                let reader = self.field_reader(name)?;
                let physical_expr = retype_roots(partition.clone(), reader.dtype().clone())?;

                // Parent nulls only remove logical rows. Therefore a falsity proof over the raw
                // child rows remains sound for a strict predicate, including for files written
                // before inherited struct validity was handled correctly during scans.
                reader
                    .pruning_evaluation(row_range, &physical_expr, mask)
                    .map_err(|err| {
                        err.with_context(format!(
                            "While evaluating pruning filter partition {name}"
                        ))
                    })
            }
            Partitioned::Single(_, _) | Partitioned::Multi(_) => {
                // Non-strict predicates such as `is_null` can become true solely because the
                // parent struct is null. Child-only statistics cannot safely falsify them.
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
        match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) => self
                .field_filter_evaluation(row_range, name, partition, mask)
                .map_err(|err| {
                    err.with_context(format!("While evaluating filter partition {name}"))
                }),
            Partitioned::Multi(partitioned) => Arc::clone(partitioned).into_mask_future(
                mask,
                |name, expr, mask| {
                    self.field_filter_evaluation(row_range, name, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!("While evaluating filter partition {name}"))
                        })
                },
                |name, expr, mask| {
                    self.field_projection_evaluation(row_range, name, expr, mask)
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
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        match &self.partition_expr(expr)? {
            Partitioned::Single(name, partition) => self
                .field_projection_evaluation(row_range, name, partition, mask)
                .map_err(|err| {
                    err.with_context(format!("While evaluating projection partition {name}"))
                }),
            Partitioned::Multi(partitioned) => {
                Arc::clone(partitioned).into_array_future(mask, |name, expr, mask| {
                    self.field_projection_evaluation(row_range, name, expr, mask)
                        .map_err(|err| {
                            err.with_context(format!(
                                "While evaluating projection partition {name}"
                            ))
                        })
                })
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
