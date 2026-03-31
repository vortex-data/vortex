// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::encodings::turboquant::array::QjlCorrection;
use crate::encodings::turboquant::array::Slot;
use crate::encodings::turboquant::array::TurboQuant;
use crate::encodings::turboquant::array::TurboQuantArray;

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
            result.set_qjl(qjl);
        }
        // Permutation is shared (not per-row), clone unchanged.
        result.slots[Slot::Permutation as usize] = array.permutation().cloned();

        Ok(Some(result.into_array()))
    }
}
