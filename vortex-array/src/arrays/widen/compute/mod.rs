// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operations that run on the values of a [`Widen`](crate::arrays::Widen) instead of on a
//! widened copy.
//!
//! An operation runs on the values only where it gives the canonical result. Decoding a value is a
//! strictly increasing, one-to-one map that keeps validity, so comparisons, sortedness, constancy
//! and extrema carry over from the values, and every value in a result widens exactly to the
//! logical dtype.
//!
//! [`AggregateFnSession::default`] registers the aggregate kernel, not
//! [`initialize`](super::initialize), because the kernel session runs `initialize` on a temporary
//! session and drops its aggregate registry.
//!
//! [`AggregateFnSession::default`]: crate::aggregate_fn::session::AggregateFnSession::default

pub(crate) mod aggregate;
mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
pub(crate) mod numeric;
pub(crate) mod rules;
mod slice;
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
