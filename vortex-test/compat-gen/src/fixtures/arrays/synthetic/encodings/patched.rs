// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PatchedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::patches::Patches;
use vortex_array::vtable::ArrayVTable;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::fixtures::FlatLayoutFixture;

pub struct PatchedFixture;

impl FlatLayoutFixture for PatchedFixture {
    fn name(&self) -> &str {
        "patched.vortex"
    }

    fn description(&self) -> &str {
        "A set of patches to apply on top of an inner array"
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let mut ctx = ExecutionCtx::new(VortexSession::empty());

        let array = PrimitiveArray::from_option_iter((0u64..2048).map(Some)).into_array();
        let patches = Patches::new(
            2048,
            0,
            PrimitiveArray::from_iter([0u32, 1024, 1025, 1026]).into_array(),
            PrimitiveArray::from_option_iter([Some(1u64), Some(2), Some(3), Some(4)]).into_array(),
            None,
        )?;

        Ok(PatchedArray::from_array_and_patches(array, &patches, &mut ctx)?.into_array())
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Patched.id()]
    }
}
