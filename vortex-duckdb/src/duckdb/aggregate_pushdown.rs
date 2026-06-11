// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cpp;
use crate::duckdb::Expression;
use crate::duckdb::ExpressionRef;
use crate::lifetime_wrapper;

lifetime_wrapper!(AggInput, cpp::duckdb_vx_agg_input, |_| {});

pub struct AggregateExpression<'a> {
    pub expr: &'a ExpressionRef,
    pub projection_id: usize,
}

impl AggInputRef {
    pub fn get_size(&self) -> usize {
        let size = unsafe { cpp::duckdb_vx_aggregate_len(self.as_ptr()) };
        size as usize
    }

    pub fn get_i(&'_ self, index: usize) -> AggregateExpression<'_> {
        let mut projection_id = 0u64;
        let expr = unsafe {
            cpp::duckdb_vx_aggregate_i(self.as_ptr(), index as u64, &mut projection_id as *mut u64)
        };
        let expr = unsafe { Expression::borrow(expr) };
        AggregateExpression {
            expr,
            projection_id: projection_id as usize,
        }
    }
}
