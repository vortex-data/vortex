// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::SharedArray;
use crate::arrays::shared::SharedArrayExt;
use crate::hash::ArrayEq;
use crate::hash::EqMode;
use crate::validity::Validity;

#[test]
fn shared_array_caches_on_canonicalize() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
    let shared = SharedArray::new(array);

    let session = crate::array_session();
    let mut ctx = ExecutionCtx::new(session);

    let first = shared.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

    // Second call should return cached without invoking the closure.
    let second = shared.get_or_compute(|_| panic!("should not execute twice"))?;

    assert!(first.array_eq(&second, EqMode::Value));

    Ok(())
}
