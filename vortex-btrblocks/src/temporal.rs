// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Specialized compressor for DateTimeParts metadata.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::TemporalArray;
use vortex_datetime_parts::DateTimePartsArray;
use vortex_datetime_parts::TemporalParts;
use vortex_datetime_parts::split_temporal;
use vortex_error::VortexResult;

use crate::Compressor;
use crate::MAX_CASCADE;
use crate::integer::IntCompressor;

/// Compress a temporal array into a `DateTimePartsArray`.
pub fn compress_temporal(array: TemporalArray) -> VortexResult<ArrayRef> {
    let dtype = array.dtype().clone();
    let TemporalParts {
        days,
        seconds,
        subseconds,
    } = split_temporal(array)?;

    let days =
        IntCompressor::compress(&days.to_primitive().narrow()?, false, MAX_CASCADE - 1, &[])?;
    let seconds = IntCompressor::compress(
        &seconds.to_primitive().narrow()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;
    let subseconds = IntCompressor::compress(
        &subseconds.to_primitive().narrow()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;

    Ok(DateTimePartsArray::try_new(dtype, days, seconds, subseconds)?.into_array())
}
