// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::scalar_fn::array::ScalarFnArray;
use crate::arrays::scalar_fn::vtable::ScalarFnVTable;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ScalarFnVTable> for ScalarFnVTable {
    // TODO(joe): fixme move to execute
    fn canonicalize(array: &ScalarFnArray) -> VortexResult<Canonical> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        array.to_array().execute::<Canonical>(&mut ctx)
    }
}
