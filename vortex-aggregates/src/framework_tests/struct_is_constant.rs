// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `is_constant` over struct arrays.

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

use crate::is_constant::is_constant;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[test]
fn test_empty_struct_is_constant() {
    let array = StructArray::new_fieldless_with_len(2);
    let mut ctx = SESSION.create_execution_ctx();
    let result = is_constant(&array.into_array(), &mut ctx)
        .vortex_expect("operation should succeed in test");
    assert!(result);
}
