// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::EntropyBins;
use crate::EntropyBinsArrayExt;

/// Masking only narrows the validity, which is stored for the unsliced rows: a sliced array's
/// mask is padded with nulls for the rows outside the slice, which are never read.
impl MaskReduce for EntropyBins {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let data = array.data();
        let (start, stop) = data.slice_range();
        let n_rows = data.unsliced_rows();
        let mask = if start == 0 && stop == n_rows {
            mask.clone()
        } else {
            let bool_dtype = DType::Bool(Nullability::NonNullable);
            let chunks = [
                (start > 0).then(|| ConstantArray::new(false, start).into_array()),
                Some(mask.clone()),
                (stop < n_rows).then(|| ConstantArray::new(false, n_rows - stop).into_array()),
            ];
            ChunkedArray::try_new(chunks.into_iter().flatten(), bool_dtype)?.into_array()
        };
        let validity = array.unsliced_validity().and(Validity::Array(mask))?;
        Ok(Some(
            EntropyBins::try_new(array.dtype().as_nullable(), data.clone(), validity)?.into_array(),
        ))
    }
}
