// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::LazyLock;

use itertools::Itertools;
use vortex_dtype::{DType, FieldName, FieldNames, Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_utils::aliases::hash_map::{DefaultHashBuilder, HashMap};

use crate::transform::annotations::{
    Annotation, AnnotationFn, Annotations, descendent_annotations,
};
use crate::transform::simplify_typed::simplify_typed;
use crate::traversal::{FoldDown, FoldUp, FolderMut, MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, GetItemVTable, ScopeDType, get_item, is_root, pack, root};

static SPLITTER_RANDOM_STATE: LazyLock<DefaultHashBuilder> =
    LazyLock::new(DefaultHashBuilder::default);

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

    let scope_dtype = ScopeDType::new(scope.clone());

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

        let expr = simplify_typed(expr.clone(), &scope_dtype)?;
        let expr_dtype = expr.return_dtype(&scope_dtype)?;

        partitions.push(expr);
        partition_annotations.push(annotation);
        partition_dtypes.push(expr_dtype);
    }

    let num_annotations = annotations.get(&expr).map(|ac| ac.len()).unwrap_or(0);
    // Ensure that there are not more accesses than partitions, we missed something
    assert!(num_annotations <= partitions.len());
    // Ensure that there are as many partitions as there are accesses/fields in the scope,
    // this will affect performance, not correctness.
    debug_assert_eq!(num_annotations, partitions.len());

    let partition_names =
        FieldNames::from_iter(partition_annotations.iter().map(|id| id.to_string()));
    let ctx = ScopeDType::new(DType::Struct(
        StructFields::new(partition_names.clone(), partition_dtypes.clone()),
        Nullability::NonNullable,
    ));

    Ok(PartitionedExpr {
        root: simplify_typed(root, &ctx)?,
        partitions: partitions.into_boxed_slice(),
        partition_names,
        partition_dtypes: partition_dtypes.into_boxed_slice(),
        partition_annotations: partition_annotations.into_boxed_slice(),
    })
}

pub fn partition_by_scope_field(
    _expr: ExprRef,
    dtype: &DType,
) -> VortexResult<PartitionedExpr<FieldName>> {
    let DType::Struct(_fields, _) = dtype else {
        vortex_bail!("Expected a struct dtype, got {:?}", dtype);
    };
    todo!()
    // StructFieldExpressionSplitter::<FieldName>::split(expr, dtype, annotate_scope_access(fields))
}

// TODO(joe): replace with let expressions.
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
        let mut hasher = SPLITTER_RANDOM_STATE.build_hasher();
        annotation.hash(&mut hasher);
        idx.hash(&mut hasher);
        format!("{}_{}", annotation, hasher.finish()).into()
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
        println!(
            "Expr {} has annotations: {:?}",
            node,
            annotations
                .map(|a| a.iter().map(|a| a.to_string()).collect_vec())
                .unwrap_or_default()
        );
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
                FieldName::from(annotation.to_string()),
                get_item(
                    StructFieldExpressionSplitter::field_name(annotation, idx),
                    root(),
                ),
            );

            return Ok(FoldDown::SkipChildren(replacement));
        };

        // If the expression is an identity, then we need to partition it into the fields of the scope.
        // FIXME(ngates): I don't think we need this...
        // if is_root(node) {
        //     let field_names = self.scope_dtype.names();
        //
        //     let mut elements = Vec::with_capacity(field_names.len());
        //
        //     for field_name in field_names.iter() {
        //         let sub_exprs = self
        //             .sub_expressions
        //             .entry(field_name.clone())
        //             .or_insert_with(Vec::new);
        //
        //         let idx = sub_exprs.len();
        //
        //         sub_exprs.push(root());
        //
        //         elements.push((
        //             field_name.clone(),
        //             // Partitions are packed into a struct of field name -> occurrence idx -> array
        //             get_item(
        //                 Self::field_name(field_name, idx),
        //                 get_item(field_name.clone(), root()),
        //             ),
        //         ));
        //     }
        //
        //     return Ok(FoldDown::SkipChildren(pack(
        //         elements,
        //         Nullability::NonNullable,
        //     )));
        // }

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

struct ScopeStepIntoFieldExpr(FieldName);

impl MutNodeVisitor for ScopeStepIntoFieldExpr {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<ExprRef>> {
        if is_root(&node) {
            Ok(TransformResult::yes(pack(
                [(self.0.clone(), root())],
                Nullability::NonNullable,
            )))
        } else {
            Ok(TransformResult::no(node))
        }
    }
}

pub(crate) struct ReplaceAccessesWithChild(Vec<FieldName>);

impl ReplaceAccessesWithChild {
    pub(crate) fn new(field_names: Vec<FieldName>) -> Self {
        Self(field_names)
    }
}

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
    use vortex_utils::aliases::hash_set::HashSet;

    use super::*;
    use crate::transform::immediate_access::annotate_scope_access;
    use crate::transform::simplify::simplify;
    use crate::transform::simplify_typed::simplify_typed;
    use crate::{PackVTable, and, col, get_item, lit, merge, pack, root, select};

    fn dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([
                (
                    "a",
                    DType::Struct(
                        StructFields::from_iter([("a", I32.into()), ("b", DType::from(I32))]),
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
        let split = partition(expr, &dtype, annotate_scope_access(fields));

        assert!(split.is_ok());

        let partitioned = split.unwrap();

        assert!(partitioned.root.is::<PackVTable>());
        // Have a single top level pack with all fields in dtype
        assert_eq!(partitioned.partitions.len(), fields.names().len())
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = get_item("b", get_item("a", root()));

        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();
        let split_a = partitioned.find_partition(&"a".into());
        assert!(split_a.is_some());
        let split_a = split_a.unwrap();

        assert_eq!(&partitioned.root, &get_item("a", root()));
        assert_eq!(&simplify(split_a.clone()).unwrap(), &get_item("b", root()));
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = pack(
            [
                ("a", get_item("a", get_item("a", root()))),
                ("b", get_item("b", get_item("a", root()))),
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
                    (
                        StructFieldExpressionSplitter::<FieldName>::field_name(&"a".into(), 0),
                        get_item("a", root())
                    ),
                    (
                        StructFieldExpressionSplitter::<FieldName>::field_name(&"a".into(), 1),
                        get_item("b", root())
                    )
                ],
                NonNullable
            )
        );
        let split_c = partitioned.find_partition(&"c".into()).unwrap();
        assert_eq!(&simplify(split_c.clone()).unwrap(), &root())
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = and(get_item("b", get_item("a", root())), lit(1));
        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();

        // Whole expr is a single split
        assert_eq!(partitioned.partitions.len(), 1);
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add_cannot_split() {
        let dtype = dtype();
        let fields = dtype.as_struct().unwrap();

        let expr = and(get_item("b", get_item("a", root())), get_item("b", root()));
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
            get_item("b", get_item("a", root())),
            select(vec!["a".into(), "b".into()], root()),
        );
        let expr = simplify_typed(expr, &ScopeDType::new(dtype.clone())).unwrap();
        let partitioned = partition(expr, &dtype, annotate_scope_access(fields)).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);

        // This fetches [].$c which is unused, however a previous optimisation should replace select
        // with get_item and pack removing this field.
        assert_eq!(
            &partitioned.root,
            &and(
                get_item(
                    StructFieldExpressionSplitter::<FieldName>::field_name(&"a".into(), 0),
                    get_item("a", root())
                ),
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
                        ("b", get_item("b", root()))
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
                ("a", get_item("a", col("a"))),
                ("b", get_item("b", col("b"))),
            ],
            NonNullable,
        );
        assert_eq!(
            &partitioned.root, &expected,
            "{} {}",
            partitioned.root, expected
        );
        let expected = [root(), pack([("b", root())], NonNullable)]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            &partitioned
                .partitions
                .clone()
                .into_iter()
                .collect::<HashSet<_>>(),
            &expected,
            "{} {}",
            partitioned.partitions.iter().join(";"),
            expected.iter().join(";")
        );
    }
}
