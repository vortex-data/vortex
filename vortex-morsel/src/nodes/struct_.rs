// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldNames;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

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

/// Struct is almost nothing: identity edges to each field, then a zip.
///
/// Every field is planned and executed under the *same* demand — the identity map means sharing
/// the demand handle rather than transforming it.
pub struct StructExec {
    names: FieldNames,
    children: Arc<[NodeId]>,

    // Per-morsel state.
    range: Range<u64>,
    plan_cursor: usize,
    plan_started: bool,
    exec_cursor: usize,
    fields: Vec<ArrayRef>,
    done: bool,
}

impl StructExec {
    /// Build a struct node over one child per projected field.
    pub fn new(names: FieldNames, children: Arc<[NodeId]>) -> Self {
        debug_assert_eq!(names.len(), children.len());
        Self {
            names,
            children,
            range: 0..0,
            plan_cursor: 0,
            plan_started: false,
            exec_cursor: 0,
            fields: Vec::new(),
            done: false,
        }
    }
}

impl ExecNode for StructExec {
    fn reset(&mut self, range: Range<u64>) {
        self.range = range;
        self.plan_cursor = 0;
        self.plan_started = false;
        self.exec_cursor = 0;
        self.fields.clear();
        self.done = false;
    }

    fn next_plan(&mut self, cx: &mut PlanCx<'_>) -> VortexResult<PlanPoll> {
        while self.plan_cursor < self.children.len() {
            if cx.out_of_budget() {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
            let fresh = !self.plan_started;
            self.plan_started = true;
            if cx.plan_child(self.children[self.plan_cursor], self.range.clone(), fresh)? {
                self.plan_cursor += 1;
                self.plan_started = false;
            } else {
                return Ok(PlanPoll::Item(PlanItem::Plan));
            }
        }
        Ok(PlanPoll::Complete)
    }

    fn execute(&mut self, cx: &mut ExecCx<'_>) -> VortexResult<ExecPoll> {
        if self.done {
            return Ok(ExecPoll::Done);
        }

        let demand = cx.demand().clone();
        let len = demand.true_count();
        if self.fields.capacity() < self.children.len() {
            self.fields
                .reserve(self.children.len().saturating_sub(self.fields.len()));
        }
        while self.exec_cursor < self.children.len() {
            let child = self.children[self.exec_cursor];
            match cx.child_array(child, demand.clone())? {
                ChildPoll::Value(array) => {
                    self.fields.push(array);
                    self.exec_cursor += 1;
                }
                ChildPoll::Blocked(waits) => return Ok(ExecPoll::Blocked(waits)),
                ChildPoll::Done => {
                    return Err(vortex_err!("struct child {child} produced no value"));
                }
            }
        }

        let fields = std::mem::take(&mut self.fields);
        let array = StructArray::try_new(self.names.clone(), fields, len, Validity::NonNullable)?
            .into_array();
        self.done = true;

        Ok(ExecPoll::Value(ValueBatch {
            coverage: self.range.clone(),
            value: Value::Array(array),
        }))
    }

    fn retire(&mut self, cx: &mut RetireCx<'_>) {
        for &child in self.children.iter() {
            cx.retire_child(child);
        }
    }

    fn children(&self) -> &[NodeId] {
        &self.children
    }
}
