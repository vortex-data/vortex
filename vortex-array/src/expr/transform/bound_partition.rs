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
use crate::expr::bound::get_item;
use crate::expr::bound::pack;
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
        let names: FieldNames = exprs
            .iter()
            .enumerate()
            .map(|(idx, _)| PartitionCollector::field_name(&annotation, idx))
            .collect();
        let expr = pack(names.into_iter().zip(exprs), Nullability::NonNullable);

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

/// The result of partitioning a bound expression.
///
/// The root and partitions remain bound so callers can cache and reuse their shared tree identity
/// without an unbind/rebind round trip.
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

        let partition = get_item(
            FieldName::from(annotation.clone()),
            BoundExpression::new_root(self.root_dtype.clone()),
        );
        let value = get_item(field_name, partition);

        Ok(Transformed {
            value,
            changed: true,
            order: TraversalOrder::Skip,
        })
    }
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
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::analysis::make_bound_free_field_annotator;
    use crate::expr::bound;

    fn scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        )
    }

    #[test]
    fn root_has_no_independent_partition() -> VortexResult<()> {
        let dtype = scope();
        let fields = dtype.as_struct_fields_opt().unwrap();
        let partitioned = partition_bound(
            bound::root(dtype.clone()),
            make_bound_free_field_annotator(fields),
        )?;
        assert!(partitioned.partitions.is_empty());
        assert_eq!(partitioned.root, bound::root(dtype));
        Ok(())
    }

    #[test]
    fn partitions_independent_fields() -> VortexResult<()> {
        let dtype = scope();
        let fields = dtype.as_struct_fields_opt().unwrap();
        let expression = bound::and(
            bound::eq(bound::col("a", dtype.clone()), bound::lit(1_i32)),
            bound::eq(bound::col("b", dtype.clone()), bound::lit(2_i32)),
        );
        let partitioned = partition_bound(expression, make_bound_free_field_annotator(fields))?;
        assert_eq!(partitioned.partitions.len(), 2);
        assert!(partitioned.find_partition(&"a".into()).is_some());
        assert!(partitioned.find_partition(&"b".into()).is_some());
        Ok(())
    }
}
