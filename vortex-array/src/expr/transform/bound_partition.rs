// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_utils::aliases::hash_map::HashMap;

use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::dtype::FieldNames;
use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::BoundExpression;
use crate::expr::ExactBoundExpr;
use crate::expr::analysis::Annotation;
use crate::expr::analysis::AnnotationFn;
use crate::expr::analysis::BoundAnnotations;
use crate::expr::analysis::descendent_bound_annotations;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeRewriter;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::pack::PackOptions;

/// Partition an expression into sub-expressions that are uniquely associated with an annotation.
/// A root expression is also returned that can be used to recombine the results of the partitions
/// into the result of the original expression.
///
/// ## Note
///
/// This function currently respects the validity of each field in the scope, but the not validity
/// of the scope itself. The fix would be for the returned `BoundPartitionedExpr` to include a
/// partition expression for computing the validity, or to include that expression as part of the
/// root.
///
/// See <https://github.com/vortex-data/vortex/issues/1907>.
pub fn partition_bound<A: AnnotationFn<BoundExpression>>(
    expr: BoundExpression,
    annotate_fn: A,
) -> VortexResult<BoundPartitionedExpr<A::Annotation>>
where
    A::Annotation: Display,
    FieldName: From<A::Annotation>,
{
    // Annotate each expression with the annotations that any of its descendent expressions have.
    let annotations = descendent_bound_annotations(&expr, annotate_fn);
    partition_bound_annotations(expr, annotations)
}

/// Partition an already-annotated bound expression tree.
///
/// Prefer [`partition_bound`] when annotations can be derived by an [`AnnotationFn`].
pub fn partition_bound_annotations<A>(
    expr: BoundExpression,
    annotations: BoundAnnotations<A>,
) -> VortexResult<BoundPartitionedExpr<A>>
where
    A: Display + Clone + Eq + Hash,
    FieldName: From<A>,
{
    let mut collector = PartitionCollector::<A>::new(&annotations);
    expr.clone().rewrite(&mut collector)?;

    let mut partitions = Vec::with_capacity(collector.sub_expressions.len());
    let mut partition_annotations = Vec::with_capacity(collector.sub_expressions.len());

    for (annotation, exprs) in collector.sub_expressions {
        // We pack all sub-expressions for the same annotation into a single expression.
        let names = exprs
            .iter()
            .enumerate()
            .map(|(idx, _)| PartitionCollector::field_name(&annotation, idx))
            .collect();
        let expr = bound_pack(names, exprs)?;

        partitions.push(expr);
        partition_annotations.push(annotation);
    }

    let partition_names = partition_annotations
        .iter()
        .map(|id| FieldName::from(id.clone()))
        .collect::<FieldNames>();
    let root_scope = partition_root_dtype(&partition_names, &partitions);
    let mut rewriter = PartitionRootRewriter::new(&annotations, root_scope);
    let root = expr.rewrite(&mut rewriter)?.value;

    Ok(BoundPartitionedExpr {
        root,
        partitions: partitions.into_boxed_slice(),
        partition_names,
        partition_annotations: partition_annotations.into_boxed_slice(),
    })
}

/// The result of partitioning an expression.
#[derive(Debug)]
pub struct BoundPartitionedExpr<A> {
    /// The root expression used to re-assemble the results.
    pub root: BoundExpression,
    /// The partition expressions themselves.
    pub partitions: Box<[BoundExpression]>,
    /// The field name of each partition as referenced in the root expression.
    pub partition_names: FieldNames,
    /// The annotation associated with each partition.
    pub partition_annotations: Box<[A]>,
}

impl<A: Display> Display for BoundPartitionedExpr<A> {
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

impl<A: Annotation> BoundPartitionedExpr<A>
where
    FieldName: From<A>,
{
    /// Return the partition for a given field, if it exists.
    // FIXME(ngates): this should return an iterator since an annotation may have multiple partitions.
    pub fn find_partition(&self, id: &A) -> Option<&BoundExpression> {
        let id = FieldName::from(id.clone());
        self.partition_names
            .iter()
            .position(|field| field == id)
            .map(|idx| &self.partitions[idx])
    }

    /// Replace the partition expressions and update every root dtype in the recombination tree.
    pub fn replace_partitions(&mut self, partitions: Box<[BoundExpression]>) -> VortexResult<()> {
        vortex_ensure!(
            partitions.len() == self.partition_names.len(),
            "Expected {} partitions, got {}",
            self.partition_names.len(),
            partitions.len()
        );

        let root_dtype = partition_root_dtype(&self.partition_names, &partitions);
        let root = replace_root_dtype(self.root.clone(), root_dtype)?;
        self.partitions = partitions;
        self.root = root;
        Ok(())
    }
}

#[derive(Debug)]
struct PartitionCollector<'a, A: Annotation> {
    annotations: &'a BoundAnnotations<A>,
    sub_expressions: HashMap<A, Vec<BoundExpression>>,
}

impl<'a, A: Annotation + Display> PartitionCollector<'a, A> {
    fn new(annotations: &'a BoundAnnotations<A>) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            annotations,
        }
    }

    /// Each annotation may be associated with multiple sub-expressions, so we need to
    /// a unique name for each sub-expression.
    fn field_name(annotation: &A, idx: usize) -> FieldName {
        format!("{annotation}_{idx}").into()
    }
}

impl<A: Annotation + Display> NodeRewriter for PartitionCollector<'_, A>
where
    FieldName: From<A>,
{
    type NodeTy = BoundExpression;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        match self.annotations.get(&ExactBoundExpr(node.clone())) {
            // If this expression only accesses a single field, then we can skip the children
            Some(annotations) if annotations.len() == 1 => {
                let annotation = annotations
                    .iter()
                    .next()
                    .vortex_expect("expected one field");
                let sub_exprs = self.sub_expressions.entry(annotation.clone()).or_default();
                sub_exprs.push(node.clone());
                Ok(Transformed {
                    value: node,
                    changed: false,
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

struct PartitionRootRewriter<'a, A: Annotation> {
    annotations: &'a BoundAnnotations<A>,
    partition_offsets: HashMap<A, usize>,
    root_dtype: DType,
}

impl<'a, A: Annotation> PartitionRootRewriter<'a, A> {
    fn new(annotations: &'a BoundAnnotations<A>, root_dtype: DType) -> Self {
        Self {
            annotations,
            partition_offsets: HashMap::new(),
            root_dtype,
        }
    }
}

impl<A: Annotation + Display> NodeRewriter for PartitionRootRewriter<'_, A>
where
    FieldName: From<A>,
{
    type NodeTy = BoundExpression;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        let Some(annotations) = self.annotations.get(&ExactBoundExpr(node.clone())) else {
            return Ok(Transformed::no(node));
        };
        if annotations.len() != 1 {
            return Ok(Transformed::no(node));
        }

        let annotation = annotations
            .iter()
            .next()
            .vortex_expect("expected one annotation");
        let offset = self
            .partition_offsets
            .entry(annotation.clone())
            .or_default();
        let field_name = PartitionCollector::field_name(annotation, *offset);
        *offset += 1;

        let partition = bound_get_item(
            FieldName::from(annotation.clone()),
            BoundExpression::new_root(self.root_dtype.clone()),
        )?;
        let value = bound_get_item(field_name, partition)?;

        Ok(Transformed {
            value,
            changed: true,
            order: TraversalOrder::Skip,
        })
    }
}

fn bound_get_item(field_name: FieldName, child: BoundExpression) -> VortexResult<BoundExpression> {
    BoundExpression::try_new(GetItem.bind(field_name), [child])
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

fn partition_root_dtype(names: &FieldNames, partitions: &[BoundExpression]) -> DType {
    DType::Struct(
        StructFields::new(
            names.clone(),
            partitions
                .iter()
                .map(|partition| partition.dtype().clone())
                .collect(),
        ),
        Nullability::NonNullable,
    )
}

fn replace_root_dtype(expr: BoundExpression, root_dtype: DType) -> VortexResult<BoundExpression> {
    Ok(expr
        .transform_down(|node| {
            if node.is_root() {
                Ok(Transformed {
                    value: BoundExpression::new_root(root_dtype.clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

#[cfg(test)]
mod tests {
    use rstest::fixture;
    use rstest::rstest;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType::I32;
    use crate::dtype::StructFields;
    use crate::expr::analysis::make_bound_free_field_annotator;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::merge;
    use crate::expr::pack;
    use crate::expr::root;
    use crate::expr::transform::replace::replace_root_fields;

    #[fixture]
    fn dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Struct(
                        StructFields::from_iter([("x", I32.into()), ("y", DType::from(I32))]),
                        NonNullable,
                    ),
                ),
                ("b", I32.into()),
                ("c", I32.into()),
            ]),
            NonNullable,
        )
    }

    fn partition_by_field(
        expr: BoundExpression,
        dtype: &DType,
    ) -> VortexResult<BoundPartitionedExpr<FieldName>> {
        let fields = dtype.as_struct_fields_opt().unwrap();
        partition_bound(expr, make_bound_free_field_annotator(fields))
    }

    #[rstest]
    fn test_expr_top_level_ref(dtype: DType) {
        let fields = dtype.as_struct_fields_opt().unwrap();

        let expr = root();
        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();

        // An un-expanded root expression is annotated by all fields, but since it is a single node
        assert_eq!(partitioned.partitions.len(), 0);
        assert_eq!(partitioned.root.unbind(), root());

        // Instead, callers must expand the root expression themselves.
        let expr = replace_root_fields(expr, fields);
        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();

        assert_eq!(partitioned.partitions.len(), fields.names().len());
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_and_split(dtype: DType) {
        let expr = get_item("y", get_item("a", root()));

        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();
        assert_eq!(
            partitioned.root.unbind(),
            get_item("a_0", get_item("a", root()))
        );
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_and_split_pack(dtype: DType) {
        let expr = pack(
            [
                ("x", get_item("x", get_item("a", root()))),
                ("y", get_item("y", get_item("a", root()))),
                ("c", get_item("c", root())),
            ],
            NonNullable,
        );
        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();

        let split_a = partitioned.find_partition(&"a".into()).unwrap();
        assert_eq!(
            split_a.unbind(),
            pack(
                [
                    ("a_0", get_item("x", get_item("a", root()))),
                    ("a_1", get_item("y", get_item("a", root())))
                ],
                NonNullable
            )
        );
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_add(dtype: DType) {
        let expr = and(get_item("y", get_item("a", root())), lit(1));
        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();

        // Whole expr is a single split
        assert_eq!(partitioned.partitions.len(), 1);
    }

    #[rstest]
    fn test_expr_top_level_ref_get_item_add_cannot_split(dtype: DType) {
        let expr = and(get_item("y", get_item("a", root())), get_item("b", root()));
        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);
    }

    #[rstest]
    fn test_expr_merge(dtype: DType) {
        let expr = merge([col("a"), pack([("b", col("b"))], NonNullable)]);

        let partitioned = partition_by_field(expr.bind(&dtype).unwrap(), &dtype).unwrap();
        let expected = merge([get_item("a_0", col("a")), get_item("b_0", col("b"))]);
        assert_eq!(
            partitioned.root.unbind(),
            expected,
            "{} {}",
            partitioned.root,
            expected
        );

        assert_eq!(partitioned.partitions.len(), 2);

        let part_a = partitioned.find_partition(&"a".into()).unwrap();
        let expected_a = pack([("a_0", col("a"))], NonNullable);
        assert_eq!(part_a.unbind(), expected_a, "{part_a} {expected_a}");

        let part_b = partitioned.find_partition(&"b".into()).unwrap();
        let expected_b = pack([("b_0", pack([("b", col("b"))], NonNullable))], NonNullable);
        assert_eq!(part_b.unbind(), expected_b, "{part_b} {expected_b}");
    }

    #[rstest]
    fn replacing_partitions_refreshes_root_dtype(dtype: DType) -> VortexResult<()> {
        let mut partitioned = partition_by_field(col("b").bind(&dtype)?, &dtype)?;
        let field_dtype = DType::Primitive(I32, Nullable);
        let replacement = pack([("b_0", root())], NonNullable).bind(&field_dtype)?;

        partitioned.replace_partitions(vec![replacement].into_boxed_slice())?;

        assert_eq!(partitioned.root.dtype(), &field_dtype);
        Ok(())
    }
}
