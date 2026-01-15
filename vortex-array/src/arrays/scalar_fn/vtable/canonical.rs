// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_vector::Datum;
use vortex_vector::Vector;

use crate::Array;
use crate::Canonical;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::executor::CanonicalOutput;
use crate::expr::ExecutionArgs;
use crate::vectors::VectorIntoArray;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ScalarFnVTable> for ScalarFnVTable {
    // TODO(joe): fixme move to execute
    fn canonicalize(array: &ScalarFnArray) -> Canonical {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        array
            .to_array()
            .execute::<Canonical>(&mut ctx)
            .vortex_expect("should handle panic")
    }
}
