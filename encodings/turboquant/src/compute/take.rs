// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::array::QjlCorrection;
use crate::array::TurboQuant;
use crate::array::TurboQuantArray;

impl TakeExecute for TurboQuant {
    fn take(
        array: &TurboQuantArray,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // FSL children handle per-row take natively.
        let taken_codes = array.codes().take(indices.clone())?;
        let taken_norms = array.norms().take(indices.clone())?;

        let taken_qjl = array
            .qjl()
            .map(|qjl| -> VortexResult<QjlCorrection> {
                Ok(QjlCorrection {
                    signs: qjl.signs.take(indices.clone())?,
                    residual_norms: qjl.residual_norms.take(indices.clone())?,
                    rotation_signs: qjl.rotation_signs.clone(),
                })
            })
            .transpose()?;

        let mut result = TurboQuantArray::try_new_mse(
            array.dtype.clone(),
            taken_codes,
            taken_norms,
            array.centroids().clone(),
            array.rotation_signs().clone(),
            array.dimension,
            array.bit_width,
        )?;
        if let Some(qjl) = taken_qjl {
            result.slots[crate::array::Slot::QjlSigns as usize] = Some(qjl.signs);
            result.slots[crate::array::Slot::QjlResidualNorms as usize] = Some(qjl.residual_norms);
            result.slots[crate::array::Slot::QjlRotationSigns as usize] = Some(qjl.rotation_signs);
        }

        Ok(Some(result.into_array()))
    }
}
