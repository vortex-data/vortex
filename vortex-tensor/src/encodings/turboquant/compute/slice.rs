// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::encodings::turboquant::array::QjlCorrection;
use crate::encodings::turboquant::array::TurboQuant;
use crate::encodings::turboquant::array::TurboQuantData;

impl SliceReduce for TurboQuant {
    fn slice(
        array: ArrayView<'_, TurboQuant>,
        range: Range<usize>,
    ) -> VortexResult<Option<ArrayRef>> {
        let sliced_codes = array.codes().slice(range.clone())?;
        let sliced_norms = array.norms().slice(range.clone())?;

        let sliced_qjl = array
            .qjl()
            .map(|qjl| -> VortexResult<QjlCorrection> {
                Ok(QjlCorrection {
                    signs: qjl.signs.slice(range.clone())?,
                    residual_norms: qjl.residual_norms.slice(range.clone())?,
                    rotation_signs: qjl.rotation_signs.clone(),
                })
            })
            .transpose()?;

        let mut result = TurboQuantData::try_new_mse(
            array.dtype.clone(),
            sliced_codes,
            sliced_norms,
            array.centroids().clone(),
            array.rotation_signs().clone(),
            array.dimension,
            array.bit_width,
        )?;
        if let Some(qjl) = sliced_qjl {
            result.set_qjl(qjl);
        }

        Ok(Some(result.into_array()))
    }
}
