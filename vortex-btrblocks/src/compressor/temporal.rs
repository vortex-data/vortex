// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Specialized compressor for DateTimeParts metadata.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::datetime::TemporalArray;
use vortex_datetime_parts::DateTimePartsArray;
use vortex_datetime_parts::TemporalParts;
use vortex_datetime_parts::split_temporal;
use vortex_error::VortexResult;

use crate::BtrBlocksCompressor;
use crate::CanonicalCompressor;
use crate::CompressorContext;
use crate::Excludes;

/// Compress a temporal array into a `DateTimePartsArray`.
pub fn compress_temporal(
    compressor: &BtrBlocksCompressor,
    array: TemporalArray,
) -> VortexResult<ArrayRef> {
    let dtype = array.dtype().clone();
    let TemporalParts {
        days,
        seconds,
        subseconds,
    } = split_temporal(array)?;

    let ctx = CompressorContext::default().descend();

    let days = compressor.compress_canonical(
        Canonical::Primitive(days.to_primitive().narrow()?),
        ctx,
        Excludes::none(),
    )?;
    let seconds = compressor.compress_canonical(
        Canonical::Primitive(seconds.to_primitive().narrow()?),
        ctx,
        Excludes::none(),
    )?;
    let subseconds = compressor.compress_canonical(
        Canonical::Primitive(subseconds.to_primitive().narrow()?),
        ctx,
        Excludes::none(),
    )?;

    Ok(DateTimePartsArray::try_new(dtype, days, seconds, subseconds)?.into_array())
}
