// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::io::IoPriority;
use crate::node::ChildPoll;
use crate::node::ExecCx;
use crate::node::ExecNode;
use crate::node::ExecPoll;
use crate::node::NodeId;
use crate::node::PlanCx;
use crate::node::PlanItem;
use crate::node::PlanPoll;
use crate::node::RetireCx;
use crate::node::Value;
use crate::node::ValueBatch;

/// The root of a morsel: refine the demand with the filter, then project under it.
pub struct FilterExec {
    predicate: Option<NodeId>,
    projection: NodeId,
    projection_expr: BoundExpression,
    output_dtype: DType,

    // Per-morsel state.
    range: Range<u64>,
    plan_stage: u8,
    plan_started: bool,
    mask: Option<Mask>,
    done: bool,
    children: Vec<NodeId>,
}

impl FilterExec {
    /// Build a filter node.
    pub fn new(
        predicate: Option<NodeId>,
        projection: NodeId,
        projection_expr: BoundExpression,
        output_dtype: DType,
    ) -> Self {
        let children = predicate.into_iter().chain([projection]).collect();
        Self {
            predicate,
            projection,
            projection_expr,
            output_dtype,
            range: 0..0,
            plan_stage: 0,
            plan_started: false,
            mask: None,
            done: false,
            children,
        }
    }
}

impl ExecNode for FilterExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_stage = 0;
        self.plan_started = false;
        self.mask = None;
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        loop {
            let child = match (self.plan_stage, self.predicate) {
                (0, Some(predicate)) => predicate,
                (0, None) | (1, _) => self.projection,
                _ => return Ok(PlanPoll::Complete),
            };
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            let priority = if self.predicate.is_some() && self.plan_stage > 0 {
                IoPriority::Speculative
            } else {
                IoPriority::Required
            };
            if cx.plan_child_with_priority(child, self.range.clone(), fresh, priority)? {
                self.plan_stage += if self.plan_stage == 0 && self.predicate.is_none() {
                    2
                } else {
                    1
                };
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }
        if self.mask.is_none() {
            let demand = cx.demand().clone();
            let mask = match self.predicate {
                Some(predicate) => match cx.child_mask(predicate, demand)? {
                    ChildPoll::Value(mask) => mask,
                    ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                    ChildPoll::Done => {
                        return Err(vortex_err!("filter predicate produced no value"));
                    }
                },
                None => demand,
            };

            if mask.all_false() {
                self.done = true;
                cx.stats().morsels_empty += 1;
                return Ok(ExecPoll::Value(ValueBatch {
                    coverage: self.range.clone(),
                    value: Value::Array(Canonical::empty(&self.output_dtype).into_array()),
                }));
            }
            self.mask = Some(mask);
        }

        // The projection subtree executes only for surviving rows. A sealed-empty chunk avoids
        // cloning and decoding its projection tickets, although planning may have prefetched them.
        let mask = self
            .mask
            .as_ref()
            .vortex_expect("non-empty predicate mask is retained")
            .clone();
        let array = match cx.child_array(self.projection, mask)? {
            ChildPoll::Value(array) => array,
            ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
            ChildPoll::Done => return Err(vortex_err!("filter projection produced no value")),
        };
        let array = array.apply_bound(&self.projection_expr)?;
        self.mask = None;
        self.done = true;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Array(array),
        }))
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        self.mask = None;
        for &child in &self.children {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}
