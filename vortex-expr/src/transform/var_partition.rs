// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Display, Formatter};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::LazyLock;

use itertools::Itertools;
use vortex_dtype::{FieldName, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::{DefaultHashBuilder, HashMap};

use crate::transform::access_analysis::{Accesses, variable_scope_accesses};
use crate::transform::partition::ReplaceAccessesWithChild;
use crate::traversal::{FoldDown, FoldUp, FolderMut, Node};
use crate::{ExprRef, Identifier, get_item, pack, var};

static SPLITTER_RANDOM_STATE: LazyLock<DefaultHashBuilder> =
    LazyLock::new(DefaultHashBuilder::default);

/// Partition an expression by the variable identifiers.
pub fn var_partitions(expr: &ExprRef) -> VortexResult<VarPartitionedExpr> {
    VariableExpressionSplitter::split_all(expr)
}

/// Partition an expression using the partition function `f`
/// e.g. var(x) + var(y) + var(z), where f(x) = {x} and f(y | z) = {y}
/// the partitioned expr will be
/// root: var(x) + var(y).0 + var(y).1, { x: var(x), y: pack(0: var(y), 1: var(z) }
pub fn var_partitions_with_map(
    expr: &ExprRef,
    f: impl Fn(&Identifier) -> Identifier,
) -> VortexResult<VarPartitionedExpr> {
    VariableExpressionSplitter::split(expr, f)
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
    pub partition_names: Box<[Identifier]>,
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
    pub fn find_partition(&self, field: &Identifier) -> Option<&ExprRef> {
        self.partition_names
            .iter()
            .position(|name| name == field)
            .map(|idx| &self.partitions[idx])
    }
}

#[derive(Debug)]
struct VariableExpressionSplitter<'a> {
    sub_expressions: HashMap<Identifier, Vec<ExprRef>>,
    accesses: &'a Accesses<'a, Identifier>,
}

impl<'a> VariableExpressionSplitter<'a> {
    fn new(accesses: &'a Accesses<'a, Identifier>) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            accesses,
        }
    }

    pub(crate) fn field_idx_name(field: &Identifier, idx: usize) -> FieldName {
        let mut hasher = SPLITTER_RANDOM_STATE.build_hasher();
        field.hash(&mut hasher);
        idx.hash(&mut hasher);
        hasher.finish().to_string().into()
    }

    fn split_all(expr: &ExprRef) -> VortexResult<VarPartitionedExpr> {
        Self::split(expr, Clone::clone)
    }

    fn split(
        expr: &ExprRef,
        f: impl Fn(&Identifier) -> Identifier,
    ) -> VortexResult<VarPartitionedExpr> {
        let field_accesses = variable_scope_accesses(expr, f)?;

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
            root: split.into_inner(),
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
    use vortex_dtype::Nullability::NonNullable;

    use super::*;
    use crate::{Pack, Var, and, root, var};

    #[test]
    fn test_expr_top_level_ref() {
        let expr = root();

        let split = VariableExpressionSplitter::split_all(&expr);

        assert!(split.is_ok());

        let partitioned = split.unwrap();

        assert!(partitioned.root.as_any().is::<Var>());
        // Have a single top level pack with all fields in dtype
        assert_eq!(partitioned.partitions.len(), 1)
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split() {
        let expr = pack([("root", root()), ("x", var("x"))], NonNullable);

        let partitioned = VariableExpressionSplitter::split_all(&expr).unwrap();

        assert_eq!(partitioned.partitions.len(), 2);
        assert_eq!(partitioned.find_partition(&"".into()), Some(&root()));
        assert_eq!(partitioned.find_partition(&"x".into()), Some(&var("x")));
    }

    #[test]
    fn test_partition_var_split_with() {
        let expr = pack(
            [("root", root()), ("x", var("x")), ("y", var("y"))],
            NonNullable,
        );

        let partitioned = VariableExpressionSplitter::split(&expr, |id| {
            if id == "x" { id.clone() } else { "".into() }
        })
        .unwrap();

        assert_eq!(partitioned.partitions.len(), 2);
        assert!(
            partitioned
                .find_partition(&"".into())
                .unwrap()
                .as_any()
                .is::<Pack>()
        );
        assert_eq!(partitioned.find_partition(&"x".into()), Some(&var("x")));
    }

    #[test]
    fn test_expr_top_level_ref_get_item_and_split_pack() {
        let expr = and(and(var("x"), root()), var("x"));
        let partitioned = VariableExpressionSplitter::split_all(&expr).unwrap();
        assert_eq!(partitioned.partitions.len(), 2);
    }
}
