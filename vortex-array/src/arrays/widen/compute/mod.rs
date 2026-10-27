// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operations that run on the values of a [`Widen`](crate::arrays::Widen).
//!
//! The aggregate kernels follow one rule. An aggregate on a `Widen` runs in the accumulator of the
//! logical dtype, and a kernel returns exactly the partial that this accumulator builds from the
//! decoded values. No accumulator receives a batch of another dtype, and no partial is cast. A
//! query on the values may run an accumulator of the values dtype, but only values or a verdict
//! leave it. So a kernel either asks the values for extrema or a verdict that decoding does not
//! change and decodes any value it returns, or it sums the values into the sum state of the logical
//! dtype. A kernel that cannot prove the partial declines.
//!
//! [`AggregateFnSession::default`] registers the aggregate kernels for their exact aggregate ids.
//! [`initialize`](super::initialize) cannot register them, because the kernel session runs it on a
//! temporary session whose aggregate registry is dropped.
//!
//! [`AggregateFnSession::default`]: crate::aggregate_fn::session::AggregateFnSession::default

mod cast;
mod compare;
mod fill_null;
mod filter;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;
mod mask;
pub(crate) mod min_max;
pub(crate) mod rules;
mod slice;
pub(crate) mod sum;
mod take;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::WidenArray;
use crate::dtype::DType;

/// Returns `values` when it has `dtype`, and a `Widen` of `dtype` over `values` otherwise.
///
/// `dtype` **must** have the nullability of `values`, because the values hold the validity of the
/// `Widen`. A mask, a fill null or a take with nullable indices can change the nullability of the
/// values, so those callers take the nullability of the result.
pub(super) fn wrap_values(values: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
    if values.dtype() == &dtype {
        return Ok(values);
    }

    Ok(WidenArray::try_new(values, dtype)?.into_array())
}
