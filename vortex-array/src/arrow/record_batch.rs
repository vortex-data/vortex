// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use vortex_error::{VortexError, VortexResult, vortex_bail, vortex_ensure};

use crate::arrow::compute::to_arrow_preferred;
use crate::{Array, Canonical};

impl TryFrom<&dyn Array> for RecordBatch {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> VortexResult<Self> {
        let Canonical::Struct(struct_array) = value.to_canonical() else {
            vortex_bail!("RecordBatch can only be constructed from ")
        };

        vortex_ensure!(
            struct_array.all_valid(),
            "RecordBatch can only be constructed from StructArray with no nulls"
        );

        let array_ref = to_arrow_preferred(struct_array.as_ref())?;
        Ok(RecordBatch::from(array_ref.as_struct()))
    }
}
