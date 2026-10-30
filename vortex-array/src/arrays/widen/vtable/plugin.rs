// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization of `Widen` arrays through existing wire forms.
//!
//! `Widen` has no wire form of its own, so its plugin registers only a writer.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::arrays::Primitive;
use crate::arrays::Widen;
use crate::session::ArraySessionExt;

/// Writes [`Widen`] arrays using an existing encoding's wire form.
#[derive(Clone, Debug)]
pub struct WidenPlugin;

impl ArrayPlugin for WidenPlugin {
    fn id(&self) -> ArrayId {
        ArrayVTable::id(&Widen)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![ArrayVTable::id(&Primitive)]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        // TODO(connor)[Widen]: Add `vortex.widen` to both ID lists when it gets a wire form
        // (#10733), and write it only where the writer permits it. This method cannot see the
        // permitted IDs today.

        // The child validity can be lazy, so the writer needs recursive canonicalization to
        // produce nodes that it can store.
        let canonical = array
            .clone()
            .execute::<RecursiveCanonical>(&mut session.create_execution_ctx())?
            .0
            .into_array();

        session.array_serialize(&canonical)
    }

    fn deserialize(
        &self,
        _parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("Widen has no wire form yet (#10733)")
    }
}
