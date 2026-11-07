// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A greedy, statistics-driven reorder of inner equi-join trees.
//!
//! DataFusion keeps the join order the SQL was written in: `JoinSelection` only swaps the two
//! inputs of each join. This rule flattens each tree of inner hash joins into its base relations
//! and the equality classes of their join keys, then rebuilds it bottom-up, always joining the
//! connected pair whose estimated output is smallest (greedy operator ordering). The estimates
//! use the relations' column statistics, so the distinct counts Vortex files record decide the
//! order. `JoinSelection`, which runs after this rule, still picks each join's build side and
//! partition mode.

use std::sync::Arc;

use datafusion_common::JoinType;
use datafusion_common::NullEquality;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::internal_err;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::replace_children_if_necessary;
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::joins::PartitionMode;
use datafusion_physical_plan::projection::ProjectionExec;

use super::cardinality::RelationEstimate;
use super::cardinality::estimate_relation;

/// A column of a base relation: `(relation, column index in its output)`.
type ColRef = (usize, usize);

/// Reorders inner equi-join trees by estimated cardinality. See the module documentation.
#[derive(Debug, Default)]
pub struct JoinReorder;

impl JoinReorder {
    /// Create the rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for JoinReorder {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        rewrite(plan)
    }

    fn name(&self) -> &str {
        "vortex_join_reorder"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn rewrite(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    if is_reorderable_join(&plan) {
        let mut graph = JoinGraph::default();
        let output = graph.flatten(&plan)?;
        if graph.relations.len() >= 3 {
            // Reorder inside each base relation too, e.g. joins beneath an aggregate.
            for relation in &mut graph.relations {
                *relation = rewrite(Arc::clone(relation))?;
            }
            if let Some(reordered) = graph.reorder(&output, &plan)? {
                return Ok(reordered);
            }
        }
    }
    let children = plan
        .children()
        .into_iter()
        .map(|child| rewrite(Arc::clone(child)))
        .collect::<Result<Vec<_>>>()?;
    if children.is_empty() {
        return Ok(plan);
    }
    replace_children_if_necessary(plan, children)
}

/// An inner equi-join this rule may reorder: no join filter, no null-equal keys, and keys that
/// are plain columns.
fn is_reorderable_join(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
        return false;
    };
    *join.join_type() == JoinType::Inner
        && *join.partition_mode() == PartitionMode::Auto
        && join.filter().is_none()
        && !join.null_aware
        && join.null_equality() == NullEquality::NullEqualsNothing
        && join.on().iter().all(|(l, r)| {
            l.downcast_ref::<Column>().is_some() && r.downcast_ref::<Column>().is_some()
        })
}

/// A column-only projection over a reorderable join, which the flattening looks through.
fn transparent_projection(plan: &Arc<dyn ExecutionPlan>) -> Option<(&ProjectionExec, Vec<usize>)> {
    let projection = plan.downcast_ref::<ProjectionExec>()?;
    if !is_reorderable_join(projection.input()) {
        return None;
    }
    let columns = projection
        .expr()
        .iter()
        .map(|e| e.expr.downcast_ref::<Column>().map(Column::index))
        .collect::<Option<Vec<_>>>()?;
    Some((projection, columns))
}

#[derive(Default)]
struct JoinGraph {
    relations: Vec<Arc<dyn ExecutionPlan>>,
    /// Equalities between base-relation columns from the original join keys.
    edges: Vec<(ColRef, ColRef)>,
}

impl JoinGraph {
    /// Flatten `plan` into base relations and key equalities, returning the base-relation column
    /// of each of its output columns.
    fn flatten(&mut self, plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<ColRef>> {
        if is_reorderable_join(plan) {
            let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
                return internal_err!("expected a hash join");
            };
            let left = self.flatten(join.left())?;
            let right = self.flatten(join.right())?;
            for (l, r) in join.on() {
                let (Some(l), Some(r)) = (l.downcast_ref::<Column>(), r.downcast_ref::<Column>())
                else {
                    return internal_err!("expected column join keys");
                };
                self.edges.push((left[l.index()], right[r.index()]));
            }
            let mut output = left;
            output.extend(right);
            if let Some(projection) = &join.projection {
                output = projection.iter().map(|&i| output[i]).collect();
            }
            return Ok(output);
        }
        if let Some((projection, columns)) = transparent_projection(plan) {
            let input = self.flatten(projection.input())?;
            return Ok(columns.into_iter().map(|i| input[i]).collect());
        }
        let relation = self.relations.len();
        self.relations.push(Arc::clone(plan));
        Ok((0..plan.schema().fields().len())
            .map(|col| (relation, col))
            .collect())
    }

    /// Rebuild the join tree greedily, returning a plan with `original`'s schema, or `None` if
    /// the relations are not all connected by join keys.
    fn reorder(
        &self,
        output: &[ColRef],
        original: &Arc<dyn ExecutionPlan>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let classes = EquivalenceClasses::new(&self.edges);
        let estimates: Vec<RelationEstimate> =
            self.relations.iter().map(estimate_relation).collect();

        let mut components: Vec<Component> = self
            .relations
            .iter()
            .enumerate()
            .map(|(rel, plan)| Component::leaf(rel, Arc::clone(plan), &estimates[rel], &classes))
            .collect();

        while components.len() > 1 {
            let mut best: Option<(usize, usize, f64)> = None;
            for first in 0..components.len() {
                for second in first + 1..components.len() {
                    let Some(rows) = components[first].join_rows(&components[second]) else {
                        continue;
                    };
                    if best.is_none_or(|(_, _, best_rows)| rows < best_rows) {
                        best = Some((first, second, rows));
                    }
                }
            }
            // A cross product would be needed: keep the original order.
            let Some((first, second, rows)) = best else {
                return Ok(None);
            };
            // Remove the later index first so the earlier one stays valid.
            let second = components.swap_remove(second);
            let first = components.swap_remove(first);
            components.push(Component::join(first, second, rows, &classes)?);
        }

        let Some(root) = components.pop() else {
            return Ok(None);
        };
        let schema = original.schema();
        let mut exprs = Vec::with_capacity(output.len());
        for (col, field) in output.iter().zip(schema.fields()) {
            let Some(index) = root.layout.iter().position(|c| c == col) else {
                return internal_err!("join reorder lost an output column");
            };
            exprs.push((
                Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            ));
        }
        Ok(Some(Arc::new(ProjectionExec::try_new(exprs, root.plan)?)))
    }
}

/// The equivalence classes of base-relation columns joined by equality.
struct EquivalenceClasses {
    members: Vec<Vec<ColRef>>,
}

impl EquivalenceClasses {
    fn new(edges: &[(ColRef, ColRef)]) -> Self {
        let mut members: Vec<Vec<ColRef>> = Vec::new();
        for &(a, b) in edges {
            let ia = members.iter().position(|m| m.contains(&a));
            let ib = members.iter().position(|m| m.contains(&b));
            match (ia, ib) {
                (Some(ia), Some(ib)) if ia == ib => {}
                (Some(ia), Some(ib)) => {
                    let merged = members.swap_remove(ia.max(ib));
                    members[ia.min(ib)].extend(merged);
                }
                (Some(i), None) => members[i].push(b),
                (None, Some(i)) => members[i].push(a),
                (None, None) => members.push(vec![a, b]),
            }
        }
        Self { members }
    }

    fn class_of(&self, col: ColRef) -> Option<usize> {
        self.members.iter().position(|m| m.contains(&col))
    }
}

/// A joined set of base relations.
struct Component {
    plan: Arc<dyn ExecutionPlan>,
    /// The base-relation column of each output column.
    layout: Vec<ColRef>,
    rows: f64,
    /// The estimated distinct count of each key class with a column in this component.
    class_ndv: Vec<Option<f64>>,
}

impl Component {
    fn leaf(
        relation: usize,
        plan: Arc<dyn ExecutionPlan>,
        estimate: &RelationEstimate,
        classes: &EquivalenceClasses,
    ) -> Self {
        let layout: Vec<ColRef> = (0..estimate.ndv.len()).map(|c| (relation, c)).collect();
        let mut class_ndv = vec![None; classes.members.len()];
        for &col in &layout {
            if let Some(class) = classes.class_of(col) {
                let ndv = estimate.ndv[col.1];
                class_ndv[class] = Some(class_ndv[class].map_or(ndv, |n: f64| n.min(ndv)));
            }
        }
        Self {
            plan,
            layout,
            rows: estimate.rows,
            class_ndv,
        }
    }

    /// The estimated rows of joining `self` with `other`, or `None` if no key connects them.
    ///
    /// Like DataFusion's estimate, this takes the most selective shared key class:
    /// `rows_a * rows_b / max(ndv_a, ndv_b)`.
    fn join_rows(&self, other: &Self) -> Option<f64> {
        let denominator = self
            .class_ndv
            .iter()
            .zip(&other.class_ndv)
            .filter_map(|(a, b)| Some(a.as_ref()?.max(*b.as_ref()?)))
            .reduce(f64::max)?;
        Some((self.rows * other.rows / denominator.max(1.0)).max(1.0))
    }

    /// Join two components on one key per shared class, putting the smaller one on the left.
    fn join(first: Self, second: Self, rows: f64, classes: &EquivalenceClasses) -> Result<Self> {
        let (left, right) = if first.rows <= second.rows {
            (first, second)
        } else {
            (second, first)
        };
        let mut on = Vec::new();
        for (class, members) in classes.members.iter().enumerate() {
            if left.class_ndv[class].is_none() || right.class_ndv[class].is_none() {
                continue;
            }
            let position = |layout: &[ColRef]| layout.iter().position(|c| members.contains(c));
            let (Some(left_col), Some(right_col)) =
                (position(&left.layout), position(&right.layout))
            else {
                continue;
            };
            on.push((
                column(&left.plan, left_col) as Arc<dyn PhysicalExpr>,
                column(&right.plan, right_col) as Arc<dyn PhysicalExpr>,
            ));
        }
        let plan = Arc::new(HashJoinExec::try_new(
            Arc::clone(&left.plan),
            Arc::clone(&right.plan),
            on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::Auto,
            NullEquality::NullEqualsNothing,
            false,
        )?);
        let class_ndv = left
            .class_ndv
            .iter()
            .zip(&right.class_ndv)
            .map(|(left_ndv, right_ndv)| match (left_ndv, right_ndv) {
                (Some(left_ndv), Some(right_ndv)) => Some(left_ndv.min(*right_ndv).min(rows)),
                (Some(ndv), None) | (None, Some(ndv)) => Some(ndv.min(rows)),
                (None, None) => None,
            })
            .collect();
        let mut layout = left.layout;
        layout.extend(right.layout);
        Ok(Self {
            plan,
            layout,
            rows,
            class_ndv,
        })
    }
}

fn column(plan: &Arc<dyn ExecutionPlan>, index: usize) -> Arc<Column> {
    Arc::new(Column::new(plan.schema().field(index).name(), index))
}
