// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::compute::BooleanOperator;
use crate::compute::arrow_boolean;

/// Execute a boolean operation between two arrays.
///
/// This is the entry point for boolean operations from the binary expression.
/// Falls back to Arrow for the actual computation.
pub(crate) fn execute_boolean(
    lhs: &dyn Array,
    rhs: &dyn Array,
    op: BooleanOperator,
) -> VortexResult<ArrayRef> {
    arrow_boolean(lhs.to_array(), rhs.to_array(), op)
}
