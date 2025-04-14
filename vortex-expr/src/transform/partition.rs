use std::fmt::{Display, Formatter};
use std::sync::Arc;

use itertools::Itertools;
use uuid::Uuid;
use vortex_array::aliases::hash_map::HashMap;
use vortex_dtype::{DType, FieldName, FieldNames, StructDType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::transform::immediate_access::{FieldAccesses, immediate_scope_accesses};
use crate::transform::simplify_typed::simplify_typed;
use crate::traversal::{FoldDown, FoldUp, FolderMut, MutNodeVisitor, Node, TransformResult};
use crate::{ExprRef, GetItem, Identity, get_item, ident, pack};

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
    pub partitions: Box<[ExprRef]>,
    /// The field names for the partitions
    pub partition_names: FieldNames,
    /// The return DTypes of each partition.
    pub partition_dtypes: Box<[DType]>,
}

impl Display for PartitionedExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "root: {} {{{}}}",
            self.root,
            self.partition_names
                .iter()
                .zip(self.partitions.iter())
                .map(|(name, partition)| format!("{}: {}", name, partition))
                .join(", ")
        )
    }
}

impl PartitionedExpr {
    /// Return the partition for a given field, if it exists.
    pub fn find_partition(&self, field: &FieldName) -> Option<&ExprRef> {
        self.partition_names
            .iter()
            .position(|name| name == field)
            .map(|idx| &self.partitions[idx])
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, PartialOrd)]
struct FieldNameIdx(FieldName, usize);

#[derive(Debug)]
struct StructFieldExpressionSplitter<'a> {
    sub_expressions: HashMap<FieldName, Vec<ExprRef>>,
    remap: HashMap<FieldNameIdx, FieldName>,
    accesses: &'a FieldAccesses<'a>,
    scope_dtype: &'a StructDType,
}

impl<'a> StructFieldExpressionSplitter<'a> {
    fn new(accesses: &'a FieldAccesses<'a>, scope_dtype: &'a StructDType) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            remap: HashMap::new(),
            accesses,
            scope_dtype,
        }
    }

    fn split(expr: ExprRef, dtype: &DType) -> VortexResult<PartitionedExpr> {
        let scope_dtype = match dtype {
            DType::Struct(scope_dtype, _) => scope_dtype,
            _ => vortex_bail!("Expected a struct dtype, got {}", dtype),
        };

        let field_accesses = immediate_scope_accesses(&expr, scope_dtype)?;

        let mut splitter = StructFieldExpressionSplitter::new(&field_accesses, scope_dtype);

        let split = expr.clone().transform_with_context(&mut splitter, ())?;

        let mut remove_accesses: Vec<FieldName> = Vec::new();

        // Create partitions which can be passed to layout fields
        let mut partitions = Vec::with_capacity(splitter.sub_expressions.len());
        let mut partition_names = Vec::with_capacity(splitter.sub_expressions.len());
        let mut partition_dtypes = Vec::with_capacity(splitter.sub_expressions.len());
        for (name, exprs) in splitter.sub_expressions.into_iter() {
            let field_dtype = scope_dtype.field(&name)?;
            // If there is a single expr then we don't need to `pack` this, and we must update
            // the root expr removing this access.
            let expr = if exprs.len() == 1 {
                remove_accesses.push(
                    splitter
                        .remap
                        .get(&FieldNameIdx(name.clone(), 0))
                        .vortex_expect("Expected expression to be remapped")
                        .clone(),
                );
                exprs.first().vortex_expect("exprs is non-empty").clone()
            } else {
                pack(exprs.into_iter().enumerate().map(|(idx, expr)| {
                    (
                        splitter
                            .remap
                            .get(&FieldNameIdx(name.clone(), idx))
                            .vortex_expect("Expected expression to be remapped")
                            .clone(),
                        expr,
                    )
                }))
            };

            let expr = simplify_typed(expr.clone(), &field_dtype)?;
            let expr_dtype = expr.return_dtype(&field_dtype)?;

            partitions.push(expr);
            partition_names.push(name);
            partition_dtypes.push(expr_dtype);
        }

        let expression_access_counts = field_accesses.get(&expr).map(|ac| ac.len());
        // Ensure that there are not more accesses than partitions, we missed something
        assert!(expression_access_counts.unwrap_or(0) <= partitions.len());
        // Ensure that there are as many partitions as there are accesses/fields in the scope,
        // this will affect performance, not correctness.
        debug_assert_eq!(expression_access_counts.unwrap_or(0), partitions.len());

        let split = split
            .result()
            .transform(&mut ReplaceAccessesWithChild(remove_accesses))?;

        Ok(PartitionedExpr {
            root: simplify_typed(split.result, dtype)?,
            partitions: partitions.into_boxed_slice(),
            partition_names: partition_names.into(),
            partition_dtypes: partition_dtypes.into_boxed_slice(),
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
                .transform(&mut ScopeStepIntoFieldExpr(field_name.clone()))?;
            sub_exprs.push(replaced.result);

            let remapped: Arc<str> = Arc::from(Uuid::new_v4().to_string());
            self.remap
                .insert(FieldNameIdx(field_name.clone(), idx), remapped.clone());

            let access = get_item(remapped, get_item(field_name.clone(), ident()));

            return Ok(FoldDown::SkipChildren(access));
        };

        // If the expression is an identity, then we need to partition it into the fields of the scope.
        if node.as_any().is::<Identity>() {
            let field_names = self.scope_dtype.names();

            let mut elements = Vec::with_capacity(field_names.len());

            for field_name in field_names.iter() {
                let sub_exprs = self
                    .sub_expressions
                    .entry(field_name.clone())
                    .or_insert_with(Vec::new);

                let idx = sub_exprs.len();

                sub_exprs.push(ident());

                let remapped: Arc<str> = Arc::from(Uuid::new_v4().to_string());
                self.remap
                    .insert(FieldNameIdx(field_name.clone(), idx), remapped.clone());

                elements.push((
                    field_name.clone(),
                    // Partitions are packed into a struct of field name -> occurrence idx -> array
                    get_item(remapped, get_item(field_name.clone(), ident())),
                ));
            }

            return Ok(FoldDown::SkipChildren(pack(elements)));
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

impl MutNodeVisitor for ScopeStepIntoFieldExpr {
    type NodeTy = ExprRef;

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<ExprRef>> {
        if node.as_any().is::<Identity>() {
            Ok(TransformResult::yes(pack([(self.0.clone(), ident())])))
        } else {
            Ok(TransformResult::no(node))
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
    use std::sync::Arc;

    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::I32;
    use vortex_dtype::{DType, StructDType};

    use super::*;
    use crate::transform::simplify::simplify;
    use crate::transform::simplify_typed::simplify_typed;
    use crate::{BinaryExpr, Pack, and, get_item, ident, lit, pack, select};

    fn dtype() -> DType {
        DType::Struct(
            Arc::new(StructDType::from_iter([
                (
                    "a",
                    DType::Struct(
                        Arc::new(StructDType::from_iter([
                            ("a", I32.into()),
                            ("b", DType::from(I32)),
                        ])),
                        NonNullable,
                    ),
                ),
                ("b", I32.into()),
                ("c", I32.into()),
            ])),
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

        assert!(partitioned.root.as_any().is::<Pack>());
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

        assert_eq!(&partitioned.root, &get_item("a", ident()));
        assert_eq!(&simplify(split_a.clone()).unwrap(), &get_item("b", ident()));
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let dtype = dtype();

        let expr = pack([
            ("a", get_item("a", get_item("a", ident()))),
            ("b", get_item("b", get_item("a", ident()))),
            ("c", get_item("c", ident())),
        ]);
        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();

        let split_a = partitioned.find_partition(&"a".into()).unwrap();
        let simplified = simplify(split_a.clone()).unwrap();
        let pack = simplified.as_any().downcast_ref::<Pack>().unwrap();
        assert_eq!(pack.names().len(), 2);
        assert_eq!(
            pack.values(),
            &[get_item("a", ident()), get_item("b", ident())]
        );
        let split_c = partitioned.find_partition(&"c".into()).unwrap();
        assert_eq!(&simplify(split_c.clone()).unwrap(), &ident())
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
        let expr = simplify_typed(expr, &dtype).unwrap();
        let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();

        // One for id.a and id.b
        assert_eq!(partitioned.partitions.len(), 2);

        // This fetches [].$c which is unused, however a previous optimisation should replace select
        // with get_item and pack removing this field.
        let root_and = partitioned
            .root
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        let and_left = root_and.lhs().as_any().downcast_ref::<GetItem>().unwrap();
        let and_right = root_and.rhs().as_any().downcast_ref::<Pack>().unwrap();
        let right_a_get_item = and_right.values()[0]
            .as_any()
            .downcast_ref::<GetItem>()
            .unwrap();

        assert_eq!(and_left.child(), &get_item("a", ident()));
        assert_eq!(
            and_right.names(),
            &FieldNames::from(["a".into(), "b".into()])
        );
        assert_eq!(right_a_get_item.child(), &get_item("a", ident()));
    }
}
