use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::{DType, Field, FieldName, StructDType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::transform::simplify_typed::simplify_typed;
use crate::traversal::{
    FoldDown, FoldUp, Folder, FolderMut, MutNodeVisitor, Node, TransformResult,
};
use crate::{get_item, ident, pack, ExprRef, GetItem, Identity, Select, SelectField};

/// Partition an expression over the fields of the scope.
///
/// This returns a partitioned expression that can be push-down over each field of the scope.
/// The results of each partition can then be recombined to reproduce the result of the original
/// expression.
///
/// ## Note
///
/// This function currently respects the validity of each field in the scope, but the not validity
/// of the scope itself. The fix would be for the returned `PartitionedExpr` to include a partition
/// expression for computing the validity, or to include that expression as part of the root.
///
/// See <https://github.com/spiraldb/vortex/issues/1907>.
///
// TODO(ngates): document the behaviour of conflicting `Field::Index` and `Field::Name`.
pub fn partition(expr: ExprRef, dtype: &DType) -> VortexResult<PartitionedExpr> {
    if !matches!(dtype, DType::Struct(..)) {
        vortex_bail!("Expected a struct dtype, got {:?}", dtype);
    }
    StructFieldExpressionSplitter::split(expr, dtype)
}

/// The result of partitioning an expression.
#[derive(Debug)]
pub struct PartitionedExpr {
    /// The root expression used to re-assemble the results.
    pub root: ExprRef,
    /// The partitions of the expression.
    pub partitions: Box<[Partition]>,
}

impl PartitionedExpr {
    /// Return the partition for a given field, if it exists.
    pub fn find_partition(&self, field: &FieldName) -> Option<&Partition> {
        self.partitions.iter().find(|p| &p.name == field)
    }
}

/// A single partition of an expression.
#[derive(Debug)]
pub struct Partition {
    /// The name of the partition, to be used when re-assembling the results.
    // TODO(ngates): we wouldn't need this if we had a MergeExpr.
    pub name: FieldName,
    /// The expression that defines the partition.
    pub expr: ExprRef,
}

type FieldAccesses<'a> = HashMap<&'a ExprRef, HashSet<FieldName>>;

// For all subexpressions in an expression, find the fields that are accessed directly from the scope.
struct ImmediateIdentityAccessesAnalysis<'a> {
    sub_expressions: FieldAccesses<'a>,
    scope_dtype: &'a StructDType,
}

impl<'a> ImmediateIdentityAccessesAnalysis<'a> {
    fn new(scope_dtype: &'a StructDType) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            scope_dtype,
        }
    }
}

// This is a very naive, but simple analysis to find the fields that are accessed directly on an
// identity node. This is combined to provide an over-approximation of the fields that are accessed
// by an expression.
// TODO(ngates): rewrite to use Visitor not Folder
impl<'a> Folder<'a> for ImmediateIdentityAccessesAnalysis<'a> {
    type NodeTy = ExprRef;
    type Out = ();
    type Context = ();

    fn visit_down(
        &mut self,
        node: &'a Self::NodeTy,
        _context: (),
    ) -> VortexResult<FoldDown<Self::Out, Self::Context>> {
        // TODO(joe): Resolve idx -> name for Field, this should be done as a separate,
        // previous (not impl, yet), pass
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if get_item
                .child()
                .as_any()
                .downcast_ref::<Identity>()
                .is_some()
            {
                self.sub_expressions
                    .insert(node, HashSet::from_iter(vec![get_item.field().clone()]));

                return Ok(FoldDown::SkipChildren(()));
            }
        } else if let Some(select) = node.as_any().downcast_ref::<Select>() {
            assert!(matches!(select.fields(), SelectField::Include(_)));
            if select.child().as_any().downcast_ref::<Identity>().is_some() {
                self.sub_expressions.insert(
                    node,
                    HashSet::from_iter(select.fields().fields().iter().cloned()),
                );
            }
            return Ok(FoldDown::SkipChildren(()));
        } else if node.as_any().downcast_ref::<Identity>().is_some() {
            let st_dtype = &self.scope_dtype;
            self.sub_expressions
                .insert(node, st_dtype.names().iter().cloned().collect());
        }

        Ok(FoldDown::Continue(()))
    }

    fn visit_up(
        &mut self,
        node: &'a ExprRef,
        _context: (),
        _children: Vec<()>,
    ) -> VortexResult<FoldUp<()>> {
        let accesses = node
            .children()
            .iter()
            .filter_map(|c| self.sub_expressions.get(c).cloned())
            .collect_vec();

        let node_accesses = self.sub_expressions.entry(node).or_default();
        accesses
            .into_iter()
            .for_each(|fields| node_accesses.extend(fields.iter().cloned()));

        Ok(FoldUp::Continue(()))
    }
}

#[derive(Debug)]
struct StructFieldExpressionSplitter<'a> {
    sub_expressions: HashMap<FieldName, Vec<ExprRef>>,
    accesses: FieldAccesses<'a>,
    scope_dtype: &'a StructDType,
}

impl<'a> StructFieldExpressionSplitter<'a> {
    fn new(accesses: FieldAccesses<'a>, scope_dtype: &'a StructDType) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            accesses,
            scope_dtype,
        }
    }

    pub(crate) fn field_idx_name(field: &FieldName, idx: usize) -> FieldName {
        format!("__e__{}.{}", field, idx).into()
    }

    fn split(expr: ExprRef, dtype: &DType) -> VortexResult<PartitionedExpr> {
        let scope_dtype = match dtype {
            DType::Struct(scope_dtype, _) => scope_dtype,
            _ => vortex_bail!("Expected a struct dtype, got {:?}", dtype),
        };

        let mut expr_top_level_ref = ImmediateIdentityAccessesAnalysis::new(scope_dtype);
        expr.accept_with_context(&mut expr_top_level_ref, ())?;

        let expression_accesses = expr_top_level_ref
            .sub_expressions
            .get(&expr)
            .map(|ac| ac.len());

        let mut splitter =
            StructFieldExpressionSplitter::new(expr_top_level_ref.sub_expressions, scope_dtype);

        let split = expr.clone().transform_with_context(&mut splitter, ())?;

        let mut remove_accesses: Vec<FieldName> = Vec::new();

        // Create partitions which can be passed to layout fields
        let partitions: Vec<Partition> = splitter
            .sub_expressions
            .into_iter()
            .map(|(name, exprs)| {
                let field_dtype = scope_dtype
                    .field_info(&Field::Name(name.clone()))?
                    .dtype
                    .value()?;
                // If there is a single expr then we don't need to `pack` this, and we must update
                // the root expr removing this access.
                let expr = if exprs.len() == 1 {
                    remove_accesses.push(Self::field_idx_name(&name, 0));
                    exprs.first().vortex_expect("exprs is non-empty").clone()
                } else {
                    pack(
                        (0..exprs.len())
                            .map(|idx| FieldName::from(Self::field_idx_name(&name, idx)))
                            .collect_vec(),
                        exprs,
                    )
                };
                VortexResult::Ok(Partition {
                    name,
                    expr: simplify_typed(expr, field_dtype)?,
                })
            })
            .try_collect()?;

        // Ensure that there are not more accesses than partitions, we missed something
        assert!(expression_accesses.unwrap_or(0) <= partitions.len());
        // Ensure that there are as many partitions as there are accesses/fields in the scope,
        // this will affect performance, not correctness.
        debug_assert_eq!(expression_accesses.unwrap_or(0), partitions.len());

        let split = split
            .result()
            .transform(&mut ReplaceAccessesWithChild(remove_accesses))?;

        Ok(PartitionedExpr {
            root: simplify_typed(split.result, dtype.clone())?,
            partitions: partitions.into_boxed_slice(),
        })
    }
}

impl FolderMut for StructFieldExpressionSplitter<'_> {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = ();

    fn visit_down(
        &mut self,
        node: &Self::NodeTy,
        _context: Self::Context,
    ) -> VortexResult<FoldDown<ExprRef, Self::Context>> {
        // If this expression only accesses a single field, then we can skip the children
        let access = self.accesses.get(node);
        if access.as_ref().is_some_and(|a| a.len() == 1) {
            let field_name = access
                .vortex_expect("access is non-empty")
                .iter()
                .next()
                .vortex_expect("expected one field");

            // TODO(joe): dedup the sub_expression, if there are two expressions that are the same
            // only create one entry here and reuse it.
            let sub_exprs = self.sub_expressions.entry(field_name.clone()).or_default();
            let idx = sub_exprs.len();

            // Need to replace get_item(f, ident) with ident, making the expr relative to the child.
            let replaced = node
                .clone()
                .transform_with_context(&mut ScopeStepIntoFieldExpr(field_name.clone()), ())?;
            sub_exprs.push(replaced.result());

            let access = get_item(
                Self::field_idx_name(field_name, idx),
                get_item(field_name.clone(), ident()),
            );

            return Ok(FoldDown::SkipChildren(access));
        };

        // If the expression is an identity, then we need to partition it into the fields of the scope.
        if node.as_any().downcast_ref::<Identity>().is_some() {
            let field_names = self.scope_dtype.names();

            let mut pack_fields = Vec::with_capacity(field_names.len());
            let mut pack_exprs = Vec::with_capacity(field_names.len());

            for field_name in field_names.iter() {
                let sub_exprs = self
                    .sub_expressions
                    .entry(field_name.clone())
                    .or_insert_with(Vec::new);

                let idx = sub_exprs.len();

                sub_exprs.push(ident());

                pack_fields.push(field_name.clone());
                // Partitions are packed into a struct of field name -> occurrence idx -> array
                pack_exprs.push(get_item(
                    Self::field_idx_name(field_name, idx),
                    get_item(field_name.clone(), ident()),
                ));
            }

            return Ok(FoldDown::SkipChildren(pack(pack_fields, pack_exprs)));
        }

        // Otherwise, continue traversing.
        Ok(FoldDown::Continue(()))
    }

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        _context: Self::Context,
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>> {
        Ok(FoldUp::Continue(node.replacing_children(children)))
    }
}

struct ScopeStepIntoFieldExpr(FieldName);

impl FolderMut for ScopeStepIntoFieldExpr {
    type NodeTy = ExprRef;
    type Out = ExprRef;
    type Context = ();

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        _context: (),
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>> {
        if node.as_any().downcast_ref::<Identity>().is_some() {
            Ok(FoldUp::Continue(pack(vec![self.0.clone()], vec![ident()])))
        } else {
            Ok(FoldUp::Continue(node.replacing_children(children)))
        }
    }
}

struct ReplaceAccessesWithChild(Vec<FieldName>);

impl MutNodeVisitor for ReplaceAccessesWithChild {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<ExprRef>> {
        if let Some(item) = node.as_any().downcast_ref::<GetItem>() {
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
    use vortex_dtype::{DType, StructDType};

    use super::*;
    use crate::transform::simplify::simplify;
    use crate::transform::simplify_typed::simplify_typed;
    use crate::{and, get_item, ident, lit, pack, select, Pack};

    fn dtype() -> DType {
        DType::Struct(
            StructDType::new(
                vec!["a".into(), "b".into(), "c".into()].into(),
                vec![
                    DType::Struct(
                        StructDType::new(
                            vec!["a".into(), "b".into()].into(),
                            vec![I32.into(), I32.into()],
                        ),
                        NonNullable,
                    ),
                    I32.into(),
                    I32.into(),
                ],
            ),
            NonNullable,
        )
    }

    #[test]
    fn test_expr_top_level_ref() {
        let dtype = dtype();

        let expr = ident();

        let split = StructFieldExpressionSplitter::split(expr, &dtype);

        assert!(split.is_ok());

        let partitioned = split.unwrap();

        assert!(partitioned.root.as_any().downcast_ref::<Pack>().is_some());
        // Have a single top level pack with all fields in dtype
        assert_eq!(
            partitioned.partitions.len(),
            dtype.as_struct().unwrap().names().len()
        )
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split() {
        let dtype = dtype();

        let expr = get_item("b", get_item("a", ident()));

        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();
        let split_a = partitioned.find_partition(&"a".into());
        assert!(split_a.is_some());
        let split_a = split_a.unwrap();

        assert_eq!(&partitioned.root, &get_item(split_a.name.clone(), ident()));
        assert_eq!(
            &simplify(split_a.expr.clone()).unwrap(),
            &get_item("b", ident())
        );
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let dtype = dtype();

        let expr = pack(
            vec!["a".into(), "b".into(), "c".into()],
            vec![
                get_item("a", get_item("a", ident())),
                get_item("b", get_item("a", ident())),
                get_item("c", ident()),
            ],
        );
        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();

        let split_a = partitioned.find_partition(&"a".into()).unwrap();
        assert_eq!(
            &simplify(split_a.expr.clone()).unwrap(),
            &pack(
                vec![
                    StructFieldExpressionSplitter::field_idx_name(&"a".into(), 0),
                    StructFieldExpressionSplitter::field_idx_name(&"a".into(), 1),
                ],
                vec![get_item("a", ident()), get_item("b", ident())]
            )
        );
        let split_c = partitioned.find_partition(&"c".into()).unwrap();
        assert_eq!(&simplify(split_c.expr.clone()).unwrap(), &ident())
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add() {
        let dtype = dtype();

        let expr = and(get_item("b", get_item("a", ident())), lit(1));
        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();

        // Whole expr is a single split
        assert_eq!(partitioned.partitions.len(), 1);
    }

    #[test]
    fn test_expr_top_level_ref_get_item_add_cannot_split() {
        let dtype = dtype();

        let expr = and(
            get_item("b", get_item("a", ident())),
            get_item("b", ident()),
        );
        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);
    }

    // Test that typed_simplify removes select and partition precise
    #[test]
    fn test_expr_partition_many_occurrences_of_field() {
        let dtype = dtype();

        let expr = and(
            get_item("b", get_item("a", ident())),
            select(vec!["a".into(), "b".into()], ident()),
        );
        let expr = simplify_typed(expr, dtype.clone()).unwrap();
        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);

        // This fetches [].$c which is unused, however a previous optimisation should replace select
        // with get_item and pack removing this field.
        assert_eq!(
            &partitioned.root,
            &and(
                get_item(
                    StructFieldExpressionSplitter::field_idx_name(&"a".into(), 0),
                    get_item("a", ident())
                ),
                pack(
                    vec!["a".into(), "b".into()],
                    vec![
                        get_item(
                            StructFieldExpressionSplitter::field_idx_name(&"a".into(), 1),
                            get_item("a", ident())
                        ),
                        get_item("b", ident()),
                    ]
                )
            )
        )
    }
}
