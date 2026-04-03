// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::TurboQuantData;

impl SliceReduce for TurboQuant {
    fn slice(
        array: ArrayView<'_, TurboQuant>,
        range: Range<usize>,
    ) -> VortexResult<Option<ArrayRef>> {
        let sliced_codes = array.codes().slice(range.clone())?;
        let sliced_norms = array.norms().slice(range)?;

        let result = TurboQuantData::try_new(
            array.dtype.clone(),
            sliced_codes,
            sliced_norms,
            array.centroids().clone(),
            array.rotation_signs().clone(),
        )?;

        Ok(Some(result.into_array()))
    }
}
