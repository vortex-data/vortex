// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::operators::Operator;

/// Point-wise Kleene logical _and_ between two Boolean arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn and_kleene(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.to_array().binary(rhs.to_array(), Operator::And)
}

/// Point-wise Kleene logical _or_ between two Boolean arrays.
#[deprecated(note = "Use `ArrayBuiltins::binary` instead")]
pub fn or_kleene(lhs: &ArrayRef, rhs: &ArrayRef) -> VortexResult<ArrayRef> {
    lhs.to_array().binary(rhs.to_array(), Operator::Or)
}
