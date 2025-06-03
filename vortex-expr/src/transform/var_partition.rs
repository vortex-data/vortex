use std::fmt::{Display, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::LazyLock;

use itertools::Itertools;
use vortex_array::aliases::hash_map::{DefaultHashBuilder, HashMap};
use vortex_dtype::{FieldName, FieldNames, Nullability};
use vortex_error::{VortexExpect, VortexResult};

use crate::transform::access_analysis::{Accesses, variable_scope_accesses};
use crate::transform::immediate_access::FieldAccesses;
use crate::transform::partition::ReplaceAccessesWithChild;
use crate::traversal::{FoldDown, FoldUp, FolderMut, Node};
use crate::{ExprRef, Identifier, get_item, pack, var};

static SPLITTER_RANDOM_STATE: LazyLock<DefaultHashBuilder> =
    LazyLock::new(DefaultHashBuilder::default);

/// Partition an expression over the variables.
pub fn partition_var(expr: ExprRef) -> VortexResult<VarPartitionedExpr> {
    VariableExpressionSplitter::split(expr)
}

// TODO(joe): replace with let expressions.
/// The result of partitioning an expression.
#[derive(Debug)]
pub struct VarPartitionedExpr {
    /// The root expression used to re-assemble the results.
    pub root: ExprRef,
    /// The partitions of the expression.
    pub partitions: Box<[ExprRef]>,
    /// The field names for the partitions
    pub partition_names: FieldNames,
}

impl Display for VarPartitionedExpr {
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

impl VarPartitionedExpr {
    /// Return the partition for a given field, if it exists.
    pub fn find_partition(&self, field: &FieldName) -> Option<&ExprRef> {
        self.partition_names
            .iter()
            .position(|name| name == field)
            .map(|idx| &self.partitions[idx])
    }
}

#[derive(Debug)]
struct VariableExpressionSplitter<'a> {
    sub_expressions: HashMap<FieldName, Vec<ExprRef>>,
    accesses: &'a Accesses<'a, Identifier>,
}

impl<'a> VariableExpressionSplitter<'a> {
    fn new(accesses: &'a FieldAccesses<'a>) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            accesses,
        }
    }

    pub(crate) fn field_idx_name(field: &FieldName, idx: usize) -> FieldName {
        let mut hasher = SPLITTER_RANDOM_STATE.build_hasher();
        field.hash(&mut hasher);
        idx.hash(&mut hasher);
        hasher.finish().to_string().into()
    }

    fn split(expr: ExprRef) -> VortexResult<VarPartitionedExpr> {
        let field_accesses = variable_scope_accesses(&expr)?;

        let mut splitter = VariableExpressionSplitter::new(&field_accesses);
        let split = expr.clone().transform_with_context(&mut splitter, ())?;
        let mut remove_accesses: Vec<FieldName> = Vec::new();

        let mut partitions = Vec::with_capacity(splitter.sub_expressions.len());
        let mut partition_names = Vec::with_capacity(splitter.sub_expressions.len());
        for (name, exprs) in splitter.sub_expressions.into_iter() {
            // If there is a single expr then we don't need to `pack` this, and we must update
            // the root expr removing this access.
            let expr = if exprs.len() == 1 {
                remove_accesses.push(Self::field_idx_name(&name, 0));
                exprs.first().vortex_expect("exprs is non-empty").clone()
            } else {
                pack(
                    exprs
                        .into_iter()
                        .enumerate()
                        .map(|(idx, expr)| (Self::field_idx_name(&name, idx), expr)),
                    Nullability::NonNullable,
                )
            };

            partitions.push(expr);
            partition_names.push(name);
        }

        let expression_access_counts = field_accesses.get(&expr).map(|ac| ac.len());
        // Ensure that there are not more accesses than partitions, we missed something
        assert!(expression_access_counts.unwrap_or(0) <= partitions.len());
        // Ensure that there are as many partitions as there are accesses/fields in the scope,
        // this will affect performance, not correctness.
        debug_assert_eq!(expression_access_counts.unwrap_or(0), partitions.len());

        let split = split
            .result()
            .transform(&mut ReplaceAccessesWithChild::new(remove_accesses))?;

        Ok(VarPartitionedExpr {
            root: split.result,
            partitions: partitions.into_boxed_slice(),
            partition_names: partition_names.into(),
        })
    }
}

impl FolderMut for VariableExpressionSplitter<'_> {
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

            let sub_exprs = self.sub_expressions.entry(field_name.clone()).or_default();
            let idx = sub_exprs.len();

            sub_exprs.push(node.clone());

            let access = get_item(
                Self::field_idx_name(field_name, idx),
                var(field_name.clone()),
            );

            return Ok(FoldDown::SkipChildren(access));
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
        Ok(FoldUp::Continue(node.replacing_children(children)))
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::PType::{I32, U64};

    use super::*;
    use crate::{Pack, ScopeDType, and, root, var};

    fn dtype() -> ScopeDType {
        ScopeDType::new(DType::Primitive(I32, NonNullable))
            .with_value("x".into(), DType::Primitive(U64, Nullability::Nullable))
    }

    #[test]
    fn test_expr_top_level_ref() {
        let dtype = dtype();

        let expr = root();

        let split = VariableExpressionSplitter::split(expr);

        assert!(split.is_ok());

        let partitioned = split.unwrap();
        println!("{}", partitioned);
        println!("{:?}", partitioned);

        assert!(partitioned.root.as_any().is::<Pack>());
        // Have a single top level pack with all fields in dtype
        assert_eq!(partitioned.partitions.len(), dtype.value_size())
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split() {
        let dtype = dtype();

        let expr = pack([("root", root()), ("x", var("x"))], NonNullable);

        let partitioned = VariableExpressionSplitter::split(expr).unwrap();
        println!("{}", partitioned);
        println!("{:?}", partitioned);

        assert_eq!(partitioned.partitions.len(), dtype.value_size());

        let split_a = partitioned.find_partition(&"a".into());
        assert!(split_a.is_some());

        assert_eq!(&partitioned.root, &get_item("a", root()));

        // let split_a = split_a.unwrap();
        // assert_eq!(&simplify(split_a.clone()).unwrap(), &get_item("b", root()));
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let expr = and(and(var("x"), root()), var("x"));
        let partitioned = VariableExpressionSplitter::split(expr).unwrap();
        println!("{}", partitioned);
        println!("{:?}", partitioned);

        // let split_a = partitioned.find_partition(&"a".into()).unwrap();
        // assert_eq!(
        //     &simplify(split_a.clone()).unwrap(),
        //     &pack(
        //         [
        //             (
        //                 StructFieldExpressionSplitter::field_idx_name(&"a".into(), 0),
        //                 get_item("a", root())
        //             ),
        //             (
        //                 StructFieldExpressionSplitter::field_idx_name(&"a".into(), 1),
        //                 get_item("b", root())
        //             )
        //         ],
        //         NonNullable
        //     )
        // );
        // let split_c = partitioned.find_partition(&"c".into()).unwrap();
        // assert_eq!(&simplify(split_c.clone()).unwrap(), &root())
    }
    //
    // #[test]
    // fn test_expr_top_level_ref_get_item_add() {
    //     let dtype = dtype();
    //
    //     let expr = and(get_item("b", get_item("a", root())), lit(1));
    //     let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();
    //
    //     // Whole expr is a single split
    //     assert_eq!(partitioned.partitions.len(), 1);
    // }
    //
    // #[test]
    // fn test_expr_top_level_ref_get_item_add_cannot_split() {
    //     let dtype = dtype();
    //
    //     let expr = and(get_item("b", get_item("a", root())), get_item("b", root()));
    //     let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();
    //
    //     // One for id.a and id.b
    //     assert_eq!(partitioned.partitions.len(), 2);
    // }
    //
    // // Test that typed_simplify removes select and partition precise
    // #[test]
    // fn test_expr_partition_many_occurrences_of_field() {
    //     let dtype = dtype();
    //
    //     let expr = and(
    //         get_item("b", get_item("a", root())),
    //         select(vec!["a".into(), "b".into()], root()),
    //     );
    //     let expr = simplify_typed(expr, &ScopeDType::new(dtype.clone())).unwrap();
    //     let partitioned = StructFieldExpressionSplitter::split(expr, &dtype).unwrap();
    //
    //     // One for id.a and id.b
    //     assert_eq!(partitioned.partitions.len(), 2);
    //
    //     // This fetches [].$c which is unused, however a previous optimisation should replace select
    //     // with get_item and pack removing this field.
    //     assert_eq!(
    //         &partitioned.root,
    //         &and(
    //             get_item(
    //                 StructFieldExpressionSplitter::field_idx_name(&"a".into(), 0),
    //                 get_item("a", root())
    //             ),
    //             pack(
    //                 [
    //                     (
    //                         "a",
    //                         get_item(
    //                             StructFieldExpressionSplitter::field_idx_name(&"a".into(), 1),
    //                             get_item("a", root())
    //                         )
    //                     ),
    //                     ("b", get_item("b", root()))
    //                 ],
    //                 NonNullable
    //             )
    //         )
    //     )
    // }
}
