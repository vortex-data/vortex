// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::EntropyBins;
use crate::EntropyBinsArrayExt;

/// Nullability changes and widening casts to an integer type of the same signedness keep the
/// array compressed: they change only its dtype (see [`EntropyBinsData::widened`]).
///
/// [`EntropyBinsData::widened`]: crate::array::EntropyBinsData::widened
impl CastReduce for EntropyBins {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Primitive(target, nullability) = dtype else {
            return Ok(None);
        };
        let data = array.data();
        let source = data.ptype();
        if !target.is_int()
            || target.is_signed_int() != source.is_signed_int()
            || target.byte_width() < source.byte_width()
        {
            return Ok(None);
        }
        let Some(validity) = array
            .unsliced_validity()
            .trivially_cast_nullability(*nullability, data.unsliced_rows())?
        else {
            return Ok(None);
        };
        Ok(Some(
            EntropyBins::try_new(dtype.clone(), data.widened(*target), validity)?.into_array(),
        ))
    }
}
