//! Specialized compressor for DateTimeParts metadata.

use vortex_array::arrays::TemporalArray;
use vortex_array::{Array, IntoArray, IntoArrayVariant};
use vortex_datetime_parts::{split_temporal, DateTimePartsArray, TemporalParts};
use vortex_error::VortexResult;

use crate::downscale::downscale_integer_array;
use crate::integer::IntCompressor;
use crate::{Compressor, MAX_CASCADE};

/// Compress a temporal array into a `DateTimePartsArray`.
pub fn compress_temporal(array: TemporalArray) -> VortexResult<Array> {
    let dtype = array.dtype().clone();
    let TemporalParts {
        days,
        seconds,
        subseconds,
    } = split_temporal(array)?;

    let days = IntCompressor::compress(
        &downscale_integer_array(days)?.into_primitive()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;
    let seconds = IntCompressor::compress(
        &downscale_integer_array(seconds)?.into_primitive()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;
    let subseconds = IntCompressor::compress(
        &downscale_integer_array(subseconds)?.into_primitive()?,
        false,
        MAX_CASCADE - 1,
        &[],
    )?;

    Ok(DateTimePartsArray::try_new(dtype, days, seconds, subseconds)?.into_array())
}
