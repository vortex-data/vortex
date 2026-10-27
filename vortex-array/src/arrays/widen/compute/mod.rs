// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operations that run on the values of a [`Widen`](crate::arrays::Widen).
//!
//! An aggregate of a `Widen` runs on its values, so that it reads the narrow values instead of a
//! widened copy, and the values keep their own cached statistics and aggregate kernels. Decoding a
//! value is a strictly increasing, one-to-one map that keeps validity. So min and max, constancy,
//! and sortedness (strict or not) are the same on the values and on the `Widen`, and every value
//! that a partial holds widens exactly to the logical dtype. Sum needs two more conditions, which
//! [`WidenArrayAggregate`](aggregate::WidenArrayAggregate) explains.
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
