// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::vtable::OperationsVTable;
use vortex::error::VortexResult;
use vortex::scalar::Scalar;

use crate::encodings::norm::array::NormVectorArray;
use crate::encodings::norm::vtable::NormVector;

impl OperationsVTable<NormVector> for NormVector {
    fn scalar_at(array: &NormVectorArray, index: usize) -> VortexResult<Scalar> {
        todo!()
    }
}
