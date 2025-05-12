//! Specialized compressor for DateTimeParts metadata.

use vortex_array::arrays::TemporalArray;
use vortex_array::compress::downscale_integer_array;
use vortex_array::{ArrayRef, IntoArray, ToCanonical};
use vortex_datetime_parts::{DateTimePartsArray, TemporalParts, split_temporal};
use vortex_error::VortexResult;

use crate::integer::IntCompressor;
use crate::{Compressor, MAX_CASCADE};

/// Compress a temporal array into a `DateTimePartsArray`.
pub fn compress_temporal(array: TemporalArray) -> VortexResult<ArrayRef> {
    let dtype = array.dtype().clone();
    let TemporalParts {
        days,
        seconds,
        subseconds,
    } = split_temporal(array)?;

    let days = IntCompressor::compress(
        &downscale_integer_array(days)?.to_primitive()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;
    let seconds = IntCompressor::compress(
        &downscale_integer_array(seconds)?.to_primitive()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;
    let subseconds = IntCompressor::compress(
        &downscale_integer_array(subseconds)?.to_primitive()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;

    Ok(DateTimePartsArray::try_new(dtype, days, seconds, subseconds)?.into_array())
}
