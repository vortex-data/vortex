// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::match_each_integer_ptype;

impl TakeExecute for Null {
    #[expect(clippy::cast_possible_truncation)]
    fn take(
        array: ArrayView<'_, Null>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let indices = indices.clone().execute::<PrimitiveArray>(ctx)?;

        // Enforce all indices are valid
        match_each_integer_ptype!(indices.ptype(), |T| {
            for index in indices.as_slice::<T>() {
                if (*index as usize) >= array.len() {
                    vortex_bail!(OutOfBounds: *index as usize, 0, array.len());
                }
            }
        });

        Ok(Some(NullArray::new(indices.len()).into_array()))
    }
}
