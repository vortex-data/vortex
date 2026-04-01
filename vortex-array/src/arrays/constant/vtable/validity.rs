// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::Constant;
use crate::validity::Validity;

impl ValidityVTable<Constant> for Constant {
    fn validity(array: ArrayView<'_, Constant>) -> VortexResult<Validity> {
        debug_assert!(array.dtype().is_nullable());
        Ok(if array.scalar().is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        })
    }
}
