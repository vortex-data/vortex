// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::BoundExpr;
use crate::expr::analysis::Annotation;
use crate::expr::analysis::AnnotationFn;
use crate::expr::analysis::Annotations;
use crate::expr::analysis::descendent_annotations;
use crate::expr::get_item;
use crate::expr::pack;
use crate::expr::root;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeRewriter;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;

/// Partition an expression into sub-expressions that are uniquely associated with an annotation.
/// A root expression is also returned that can be used to recombine the results of the partitions
/// into the result of the original expression.
///
/// ## Note
///
/// This function currently respects the validity of each field in the scope, but the not validity
/// of the scope itself. The fix would be for the returned `PartitionedExpr` to include a partition
/// expression for computing the validity, or to include that expression as part of the root.
///
/// See <https://github.com/vortex-data/vortex/issues/1907>.
pub fn partition<A: AnnotationFn>(
    expr: BoundExpr,
    _scope: &DType,
    annotate_fn: A,
) -> VortexResult<PartitionedExpr<A::Annotation>>
where
    A::Annotation: Display,
    FieldName: From<A::Annotation>,
{
    // Annotate each expression with the annotations that any of its descendent expressions have.
    let annotations = descendent_annotations(&expr, annotate_fn);

    // Now we split the original expression into sub-expressions based on the annotations, and
    // generate a root expression to re-assemble the results.
    let mut collector = StructFieldExpressionSplitter::<A::Annotation>::new(&annotations, None);
    expr.clone().rewrite(&mut collector)?;

    let mut partitions = Vec::with_capacity(collector.sub_expressions.len());
    let mut partition_annotations = Vec::with_capacity(collector.sub_expressions.len());
    let mut partition_dtypes = Vec::with_capacity(collector.sub_expressions.len());

    for (annotation, exprs) in collector.sub_expressions.into_iter() {
        // We pack all sub-expressions for the same annotation into a single expression.
        let expr = pack(
            exprs.into_iter().enumerate().map(|(idx, expr)| {
                (
                    StructFieldExpressionSplitter::field_name(&annotation, idx),
                    expr,
                )
            }),
            Nullability::NonNullable,
        );

        let expr = expr.optimize_recursive()?;
        let expr_dtype = expr.dtype().clone();

        partitions.push(expr);
        partition_annotations.push(annotation);
        partition_dtypes.push(expr_dtype);
    }

    let partition_names = partition_annotations
        .iter()
        .map(|id| FieldName::from(id.clone()))
        .collect::<FieldNames>();
    let root_scope = DType::Struct(
        StructFields::new(partition_names.clone(), partition_dtypes.clone()),
        Nullability::NonNullable,
    );
    let mut splitter =
        StructFieldExpressionSplitter::<A::Annotation>::new(&annotations, Some(root_scope));
    let root = expr.clone().rewrite(&mut splitter)?.value;

    Ok(PartitionedExpr {
        root: root.optimize_recursive()?,
        partitions: partitions.into_boxed_slice(),
        partition_names,
        partition_dtypes: partition_dtypes.into_boxed_slice(),
        partition_annotations: partition_annotations.into_boxed_slice(),
    })
}

/// The result of partitioning an expression.
#[derive(Debug)]
pub struct PartitionedExpr<A> {
    /// The root expression used to re-assemble the results.
    pub root: BoundExpr,
    /// The partition expressions themselves.
    pub partitions: Box<[BoundExpr]>,
    /// The field name of each partition as referenced in the root expression.
    pub partition_names: FieldNames,
    /// The return dtype of each partition expression.
    pub partition_dtypes: Box<[DType]>,
    /// The annotation associated with each partition.
    pub partition_annotations: Box<[A]>,
}

impl<A: Display> Display for PartitionedExpr<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "root: {} {{{}}}",
            self.root,
            self.partition_names
                .iter()
                .zip(self.partitions.iter())
                .map(|(name, partition)| format!("{name}: {partition}"))
                .join(", ")
        )
    }
}

impl<A: Annotation> PartitionedExpr<A>
where
    FieldName: From<A>,
{
    /// Return the partition for a given field, if it exists.
    // FIXME(ngates): this should return an iterator since an annotation may have multiple partitions.
    pub fn find_partition(&self, id: &A) -> Option<&BoundExpr> {
        let id = FieldName::from(id.clone());
        self.partition_names
            .iter()
            .position(|field| field == id)
            .map(|idx| &self.partitions[idx])
    }
}

#[derive(Debug)]
struct StructFieldExpressionSplitter<'a, A: Annotation> {
    annotations: &'a Annotations<'a, A>,
    sub_expressions: HashMap<A, Vec<BoundExpr>>,
    root_scope: Option<DType>,
}

impl<'a, A: Annotation + Display> StructFieldExpressionSplitter<'a, A> {
    fn new(annotations: &'a Annotations<'a, A>, root_scope: Option<DType>) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            annotations,
            root_scope,
        }
    }

    /// Each annotation may be associated with multiple sub-expressions, so we need to
    /// a unique name for each sub-expression.
    fn field_name(annotation: &A, idx: usize) -> FieldName {
        format!("{annotation}_{idx}").into()
    }
}

impl<A: Annotation + Display> NodeRewriter for StructFieldExpressionSplitter<'_, A>
where
    FieldName: From<A>,
{
    type NodeTy = BoundExpr;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        match self.annotations.get(&node) {
            // If this expression only accesses a single field, then we can skip the children
            Some(annotations) if annotations.len() == 1 => {
                let annotation = annotations
                    .iter()
                    .next()
                    .vortex_expect("expected one field");
                let sub_exprs = self.sub_expressions.entry(annotation.clone()).or_default();
                let idx = sub_exprs.len();
                sub_exprs.push(node.clone());
                let Some(root_scope) = &self.root_scope else {
                    return Ok(Transformed {
                        value: node,
                        changed: false,
                        order: TraversalOrder::Skip,
                    });
                };
                let value = get_item(
                    StructFieldExpressionSplitter::field_name(annotation, idx),
                    get_item(
                        FieldName::from(annotation.clone()),
                        root(root_scope.clone()),
                    ),
                );
                Ok(Transformed {
                    value,
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            }

            // Otherwise, continue traversing.
            _ => Ok(Transformed::no(node)),
        }
    }

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        Ok(Transformed::no(node))
    }
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::analysis::make_free_field_annotator;
    use crate::expr::and;
    use crate::expr::col as expr_col;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::merge;
    use crate::expr::pack;
    use crate::expr::root as expr_root;
    use crate::expr::transform::replace::replace_root_fields;

    #[fixture]
    fn dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Struct(
                        StructFields::from_iter([
                            ("x", I32.into()),
                            ("y", DType::Bool(NonNullable)),
                        ]),
                        NonNullable,
                    ),
                ),
                ("b", DType::Bool(NonNullable)),
                ("c", I32.into()),
            ]),
            NonNullable,
        )
    }

    fn root(dtype: &DType) -> BoundExpr {
        expr_root(dtype.clone())
    }

    fn col(field: impl Into<FieldName>, dtype: &DType) -> BoundExpr {
        expr_col(field, dtype)
    }

    fn partition_scope<A>(partitioned: &PartitionedExpr<A>) -> DType {
        DType::Struct(
            StructFields::new(
                partitioned.partition_names.clone(),
                partitioned.partition_dtypes.to_vec(),
            ),
            NonNullable,
        )
    }

    #[rstest]
    fn test_expr_top_level_ref(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = root(&dtype);
        let partitioned =
            partition(expr.clone(), &dtype, make_free_field_annotator(fields)).unwrap();

        // An un-expanded root expression is annotated by all fields, but since it is a single node
        assert_eq!(partitioned.partitions.len(), 0);
        assert_eq!(&partitioned.root, &root(&dtype));

        // Instead, callers must expand the root expression themselves.
        let expr = replace_root_fields(expr, fields).unwrap();
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        assert_eq!(partitioned.partitions.len(), fields.names().len());
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_and_split(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = get_item("y", get_item("a", root(&dtype)));

        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();
        let partition_dtype = partition_scope(&partitioned);
        assert_eq!(
            &partitioned.root,
            &get_item("a_0", get_item("a", root(&partition_dtype)))
        );
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_and_split_pack(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = pack(
            [
                ("x", get_item("x", get_item("a", root(&dtype)))),
                ("y", get_item("y", get_item("a", root(&dtype)))),
                ("c", get_item("c", root(&dtype))),
            ],
            NonNullable,
        );
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        let split_a = partitioned.find_partition(&"a".into()).unwrap();
        assert_eq!(
            &split_a.optimize_recursive().unwrap(),
            &pack(
                [
                    ("a_0", get_item("x", get_item("a", root(&dtype)))),
                    ("a_1", get_item("y", get_item("a", root(&dtype))))
                ],
                NonNullable
            )
        );
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_add(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = and(get_item("y", get_item("a", root(&dtype))), lit(true));
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        // Whole expr is a single split
        assert_eq!(partitioned.partitions.len(), 1);
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_add_cannot_split(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = and(
            get_item("y", get_item("a", root(&dtype))),
            get_item("b", root(&dtype)),
        );
        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);
    }

    #[rstest]
    fn test_expr_merge(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = merge([
            col("a", &dtype),
            pack([("b", col("b", &dtype))], NonNullable),
        ]);

        let partitioned = partition(expr, &dtype, make_free_field_annotator(fields)).unwrap();
        let partition_dtype = partition_scope(&partitioned);
        let expected = pack(
            [
                (
                    "x",
                    get_item("x", get_item("a_0", col("a", &partition_dtype))),
                ),
                (
                    "y",
                    get_item("y", get_item("a_0", col("a", &partition_dtype))),
                ),
                (
                    "b",
                    get_item("b", get_item("b_0", col("b", &partition_dtype))),
                ),
            ],
            NonNullable,
        );
        assert_eq!(
            &partitioned.root, &expected,
            "{} {}",
            partitioned.root, expected
        );

        assert_eq!(partitioned.partitions.len(), 2);

        let part_a = partitioned.find_partition(&"a".into()).unwrap();
        let expected_a = pack([("a_0", col("a", &dtype))], NonNullable);
        assert_eq!(part_a, &expected_a, "{part_a} {expected_a}");

        let part_b = partitioned.find_partition(&"b".into()).unwrap();
        let expected_b = pack(
            [("b_0", pack([("b", col("b", &dtype))], NonNullable))],
            NonNullable,
        );
        assert_eq!(part_b, &expected_b, "{part_b} {expected_b}");
    }
}
