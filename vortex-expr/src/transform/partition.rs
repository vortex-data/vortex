// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};

use itertools::Itertools;
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;

use crate::transform::annotations::{
    Annotation, AnnotationFn, Annotations, descendent_annotations,
};
use crate::transform::simplify_typed::simplify_typed;
use crate::traversal::{FoldDown, FoldUp, FolderMut, MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, GetItemVTable, get_item, pack, root};

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
    expr: ExprRef,
    scope: &DType,
    annotate_fn: A,
) -> VortexResult<PartitionedExpr<A::Annotation>>
where
    A::Annotation: Display,
{
    // Annotate each expression with the annotations that any of its descendent expressions have.
    let annotations = descendent_annotations(&expr, annotate_fn);

    // Now we split the original expression into sub-expressions based on the annotations, and
    // generate a root expression to re-assemble the results.

    let mut splitter = StructFieldExpressionSplitter::<A::Annotation>::new(&annotations);
    let root = expr
        .clone()
        .transform_with_context(&mut splitter, ())?
        .result();

    let mut partitions = Vec::with_capacity(splitter.sub_expressions.len());
    let mut partition_annotations = Vec::with_capacity(splitter.sub_expressions.len());
    let mut partition_dtypes = Vec::with_capacity(splitter.sub_expressions.len());

    for (annotation, exprs) in splitter.sub_expressions.into_iter() {
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

        let expr = simplify_typed(expr.clone(), scope)?;
        let expr_dtype = expr.return_dtype(scope)?;

        partitions.push(expr);
        partition_annotations.push(annotation);
        partition_dtypes.push(expr_dtype);
    }

    let partition_names =
        FieldNames::from_iter(partition_annotations.iter().map(|id| id.to_string()));
    let root_scope = DType::Struct(
        StructFields::new(partition_names.clone(), partition_dtypes.clone()),
        Nullability::NonNullable,
    );

    Ok(PartitionedExpr {
        root: simplify_typed(root, &root_scope)?,
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
    pub root: ExprRef,
    /// The partition expressions themselves.
    pub partitions: Box<[ExprRef]>,
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

impl<A: Annotation + Display> PartitionedExpr<A> {
    /// Return the partition for a given field, if it exists.
    // FIXME(ngates): this should return an iterator since an annotation may have multiple partitions.
    pub fn find_partition(&self, id: &A) -> Option<&ExprRef> {
        let id = FieldName::from(id.to_string());
        self.partition_names
            .iter()
            .position(|field| field == &id)
            .map(|idx| &self.partitions[idx])
    }
}

#[derive(Debug)]
struct StructFieldExpressionSplitter<'a, A: Annotation> {
    annotations: &'a Annotations<'a, A>,
    sub_expressions: HashMap<A, Vec<ExprRef>>,
}

impl<'a, A: Annotation + Display> StructFieldExpressionSplitter<'a, A> {
    fn new(annotations: &'a Annotations<'a, A>) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            annotations,
        }
    }

    /// Each annotation may be associated with multiple sub-expressions, so we need to
    /// a unique name for each sub-expression.
    fn field_name(annotation: &A, idx: usize) -> FieldName {
        format!("{}_{}", annotation, idx).into()
    }
}

// FIXME(ngates): rewrite as MutNodeVisitor that skips down when annotations.len() == 1
impl<A: Annotation + Display> FolderMut for StructFieldExpressionSplitter<'_, A> {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = ();

    fn visit_down(
        &mut self,
        node: &Self::NodeTy,
        _context: Self::Context,
    ) -> VortexResult<FoldDown<ExprRef, Self::Context>> {
        // If this expression only accesses a single field, then we can skip the children
        let annotations = self.annotations.get(node);
        if annotations.as_ref().is_some_and(|a| a.len() == 1) {
            let annotation = annotations
                .vortex_expect("access is non-empty")
                .iter()
                .next()
                .vortex_expect("expected one field");

            let sub_exprs = self.sub_expressions.entry(annotation.clone()).or_default();
            let idx = sub_exprs.len();
            sub_exprs.push(node.clone());

            // In the root, we replace the annotated sub-expression with a `&.<A>.<A_idx>` since
            // we assemble all sub-expressions for the same annotation into a single child.
            let replacement = get_item(
                StructFieldExpressionSplitter::field_name(annotation, idx),
                get_item(FieldName::from(annotation.to_string()), root()),
            );

            return Ok(FoldDown::SkipChildren(replacement));
        };

        // Otherwise, continue traversing.
        Ok(FoldDown::Continue(()))
    }

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        _context: Self::Context,
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>> {
        Ok(FoldUp::Continue(node.with_children(children)?))
    }
}

pub(crate) struct ReplaceAccessesWithChild(Vec<FieldName>);

impl MutNodeVisitor for ReplaceAccessesWithChild {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<ExprRef>> {
        if let Some(item) = node.as_opt::<GetItemVTable>() {
            if self.0.contains(item.field()) {
                return Ok(TransformResult::yes(item.child().clone()));
            }
        }
        Ok(TransformResult::no(node))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructFields};

    use super::*;
    use crate::transform::immediate_access::annotate_scope_access;
    use crate::transform::replace::replace_root_fields;
    use crate::transform::simplify::simplify;
    use crate::transform::simplify_typed::simplify_typed;
    use crate::{and, col, get_item, lit, merge, pack, root, select};

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

    #[test]
    fn test_expr_top_level_ref() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = root();
        let partitioned = partition(expr.clone(), &dtype, annotate_scope_access(fields)).unwrap();

        // An un-expanded root expression is annotated by all fields, but since it is a single node
        assert_eq!(partitioned.partitions.len(), 0);
        assert_eq!(&partitioned.root, &root());

        // Instead, callers must expand the root expression themselves.
        let expr = replace_root_fields(expr.clone(), fields);
        let partitioned = partition(expr.clone(), &dtype, annotate_scope_access(fields)).unwrap();

        assert_eq!(partitioned.partitions.len(), fields.names().len());
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = get_item("y", get_item("a", root()));

        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();
        assert_eq!(&partitioned.root, &get_item("a_0", get_item("a", root())));
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = pack(
            [
                ("x", get_item("x", get_item("a", root()))),
                ("y", get_item("y", get_item("a", root()))),
                ("c", get_item("c", root())),
            ],
            NonNullable,
        );
        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();

        let split_a = partitioned.find_partition(&"a".into()).unwrap();
        assert_eq!(
            &simplify(split_a.clone()).unwrap(),
            &pack(
                [
                    ("a_0", get_item("x", get_item("a", root()))),
                    ("a_1", get_item("y", get_item("a", root())))
                ],
                NonNullable
            )
        );
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = and(get_item("y", get_item("a", root())), lit(1));
        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();

        // Whole expr is a single split
        assert_eq!(partitioned.partitions.len(), 1);
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add_cannot_split() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = and(get_item("y", get_item("a", root())), get_item("b", root()));
        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);
    }

    // Test that typed_simplify removes select and partition precise
    #[test]
    fn test_expr_partition_many_occurrences_of_field() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = and(
            get_item("y", get_item("a", root())),
            select(vec!["a".into(), "b".into()], root()),
        );
        let expr = simplify_typed(expr, &dtype).unwrap();
        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);

        // This fetches [].$c which is unused, however a previous optimisation should replace select
        // with get_item and pack removing this field.
        assert_eq!(
            &partitioned.root,
            &and(
                get_item("a_0", get_item("a", root())),
                pack(
                    [
                        (
                            "a",
                            get_item(
                                StructFieldExpressionSplitter::<FieldName>::field_name(
                                    &"a".into(),
                                    1
                                ),
                                get_item("a", root())
                            )
                        ),
                        ("b", get_item("b_0", get_item("b", root())))
                    ],
                    NonNullable
                )
            )
        )
    }

    #[test]
    fn test_expr_merge() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = merge(
            [col("a"), pack([("b", col("b"))], NonNullable)],
            NonNullable,
        );

        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();
        let expected = pack(
            [
                ("x", get_item("x", get_item("a_0", col("a")))),
                ("y", get_item("y", get_item("a_0", col("a")))),
                ("b", get_item("b", get_item("b_0", col("b")))),
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
        let expected_a = pack([("a_0", col("a"))], NonNullable);
        assert_eq!(part_a, &expected_a, "{} {}", part_a, expected_a);

        let part_b = partitioned.find_partition(&"b".into()).unwrap();
        let expected_b = pack([("b_0", pack([("b", col("b"))], NonNullable))], NonNullable);
        assert_eq!(part_b, &expected_b, "{} {}", part_b, expected_b);
    }
}
