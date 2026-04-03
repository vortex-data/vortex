// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::vtable::ParquetVariant;

impl ValidityVTable<ParquetVariant> for ParquetVariant {
    fn validity(array: ArrayView<'_, ParquetVariant>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
