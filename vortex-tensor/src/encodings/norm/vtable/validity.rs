// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::vtable::ValidityChild;

use crate::encodings::norm::array::NormVectorArray;
use crate::encodings::norm::vtable::NormVector;

impl ValidityChild<NormVector> for NormVector {
    fn validity_child(array: &NormVectorArray) -> &ArrayRef {
        array.vector_array()
    }
}
