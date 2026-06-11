// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;

use crate::cpp;
use crate::duckdb::Expression;
use crate::duckdb::ExpressionRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(
    /// Aggregates we want to push at Vortex at once
    AggregatePushdownInput,
    cpp::duckdb_vx_agg_input, |_| {});

pub struct AggregateExpression<'a> {
    pub expr: &'a ExpressionRef,
    /// Output column projection id after the pass has expanded columns with
    /// multiple aggregations per column
    pub projection_id: u64,
}

impl AggregatePushdownInputRef {
    pub fn len(&self) -> usize {
        unsafe { cpp::duckdb_vx_aggregate_len(self.as_ptr()) }.as_()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&'_ self, index: usize) -> AggregateExpression<'_> {
        let mut projection_id = 0u64;
        let expr = unsafe {
            cpp::duckdb_vx_aggregate_at(self.as_ptr(), index as u64, &raw mut projection_id)
        };
        let expr = unsafe { Expression::borrow(expr) };
        AggregateExpression {
            expr,
            projection_id,
        }
    }
}
