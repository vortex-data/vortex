// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Truncate variable length bounds in a detached node summary.
//!
//! Serialization can shorten stored extrema while leaving exact results in the live array cache.

use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;
use crate::scalar::ScalarTruncation;
use crate::scalar::lower_bound;
use crate::scalar::upper_bound;
use crate::stats::AggregateResults;

/// Truncate variable length extrema in a detached snapshot, preserving existing bounds.
pub(crate) fn truncate_summary(
    results: &AggregateResults,
    max_length: usize,
) -> VortexResult<AggregateResults> {
    let mut entries = results
        .iter()
        .map(|(aggregate, value)| (aggregate.clone(), value.clone()))
        .collect::<Vec<_>>();
    for (aggregate, value) in &mut entries {
        let stat = if aggregate == Stat::Min.finalized_aggregate_fn() {
            Stat::Min
        } else if aggregate == Stat::Max.finalized_aggregate_fn() {
            Stat::Max
        } else {
            continue;
        };
        let Some(scalar) = value.as_ref().into_inner() else {
            continue;
        };
        if scalar.is_null() {
            continue;
        }
        *value = match scalar.dtype() {
            DType::Utf8(_) => truncate_value::<BufferString>(value, stat, max_length)?,
            DType::Binary(_) => truncate_value::<ByteBuffer>(value, stat, max_length)?,
            _ => continue,
        };
    }
    Ok(AggregateResults::from_validated(entries))
}

fn truncate_value<T: ScalarTruncation>(
    value: &Precision<Scalar>,
    stat: Stat,
    max_length: usize,
) -> VortexResult<Precision<Scalar>> {
    let Some(scalar) = value.as_ref().into_inner() else {
        return Ok(Precision::Absent);
    };
    let nullability = scalar.dtype().nullability();
    let scalar = T::from_scalar(scalar.clone())?;
    let truncated = match stat {
        Stat::Min => lower_bound(scalar, max_length, nullability),
        _ => upper_bound(scalar, max_length, nullability),
    };
    Ok(match truncated {
        Some((scalar, true)) => Precision::Inexact(scalar),
        Some((_, false)) => value.clone(),
        None => Precision::Absent,
    })
}
