// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::array::QjlCorrection;
use crate::array::TurboQuant;
use crate::array::TurboQuantArray;

impl SliceReduce for TurboQuant {
    fn slice(array: &TurboQuantArray, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
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

        let mut result = TurboQuantArray::try_new_mse(
            array.dtype.clone(),
            sliced_codes,
            sliced_norms,
            array.centroids().clone(),
            array.rotation_signs().clone(),
            array.dimension,
            array.bit_width,
        )?;
        if let Some(qjl) = sliced_qjl {
            result.slots[crate::array::Slot::QjlSigns as usize] = Some(qjl.signs);
            result.slots[crate::array::Slot::QjlResidualNorms as usize] = Some(qjl.residual_norms);
            result.slots[crate::array::Slot::QjlRotationSigns as usize] = Some(qjl.rotation_signs);
        }

        Ok(Some(result.into_array()))
    }
}
