// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::VectorOps;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::CanonicalOutput;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ScalarFnVTable> for ScalarFnVTable {
    fn validity(array: &ScalarFnArray) -> VortexResult<Validity> {
        // TODO(ngates): we should make this lazy by adding a ScalarFnValidityArray
        //  and referencing evaluate_validity
        Ok(Validity::from_mask(
            Self::validity_mask(array),
            array.dtype().nullability(),
        ))
    }

    fn validity_mask(array: &ScalarFnArray) -> Mask {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let output = array
            .to_array()
            .execute::<CanonicalOutput>(&mut ctx)
            .vortex_expect("Validity mask computation should be fallible");
        match output {
            CanonicalOutput::Constant(c) => Mask::new(array.len, c.scalar().is_valid()),
            CanonicalOutput::Array(a) => a
                .into_array()
                .execute::<Vector>(&mut ctx)
                .vortex_expect("Failed to convert canonical to vector")
                .validity()
                .clone(),
        }
    }
}
