// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Limit serialized extrema without changing the live result cache.
//!
//! Truncation preserves lower bounds for minima and upper bounds for maxima. A maximum without a
//! representable upper bound is omitted from the snapshot.

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::lower_bound;
use crate::scalar::upper_bound;
use crate::stats::AggregateResults;

/// Copy finalized results with string and binary extrema limited to `max_length` bytes.
///
/// Existing inexact results remain inexact. Exact nulls and non-extremum results are preserved.
/// Callers serialize the returned snapshot while the live cache retains its original values.
pub fn truncate_results(
    results: &AggregateResults,
    max_length: usize,
) -> VortexResult<AggregateResults> {
    let entries = results
        .iter()
        .map(|(aggregate, precision)| {
            let Some(value) = precision.as_ref().into_inner() else {
                return Ok((aggregate.clone(), Precision::Absent));
            };
            if value.is_null() || !(aggregate.is::<Min>() || aggregate.is::<Max>()) {
                return Ok((aggregate.clone(), precision.clone()));
            }

            let is_max = aggregate.is::<Max>();
            let truncated = match value.dtype() {
                DType::Utf8(_) => truncate::<BufferString>(value.clone(), is_max, max_length)?,
                DType::Binary(_) => truncate::<ByteBuffer>(value.clone(), is_max, max_length)?,
                _ => return Ok((aggregate.clone(), precision.clone())),
            };
            let truncated = if precision.is_exact() {
                truncated
            } else {
                truncated.into_inexact()
            };

            Ok((aggregate.clone(), truncated))
        })
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(AggregateResults::from_validated(entries))
}

fn truncate<T: ScalarTruncation>(
    value: Scalar,
    is_max: bool,
    max_length: usize,
) -> VortexResult<Precision<Scalar>> {
    let nullability = value.dtype().nullability();
    let value = T::from_scalar(value)?;
    let bound = if is_max {
        upper_bound(value, max_length, nullability)
    } else {
        lower_bound(value, max_length, nullability)
    };

    Ok(match bound {
        Some((value, true)) => Precision::Inexact(value),
        Some((value, false)) => Precision::Exact(value),
        None => Precision::Absent,
    })
}
