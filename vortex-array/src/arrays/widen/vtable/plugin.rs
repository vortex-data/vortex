// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization of decimal `Widen` arrays as narrow `vortex.decimal` nodes.
//!
//! The node stores the child's buffer width, while its containing format supplies the logical dtype
//! of the `Widen`. No `Widen` node or child decimal dtype is written.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::ArrayVTable;
use crate::arrays::Decimal;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::session::ArraySessionExt;

/// Writes a decimal [`Widen`] as one `vortex.decimal` node with its child's values buffer.
///
/// The old decimal wire format allows a buffer narrower than the logical dtype's canonical width.
/// For example, a `Widen` of `Decimal(38, 2)` over a `Decimal(9, 2)` child writes the child's `i32`
/// buffer while keeping the logical dtype `Decimal(38, 2)`.
///
/// The child **must** be a [`Decimal`] leaf. Its serializer supplies width metadata, buffers and
/// validity children through [`ArraySerialization`], which contains no dtype. The containing format
/// supplies the original `Widen` dtype and length, and array serialization keeps its statistics.
///
/// This plugin registers only a writer. The decimal reader handles the resulting node. Serialization
/// executes neither values nor validity, so integer arrays, non-decimal children and validity that
/// cannot be serialized require preparation by the caller.
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
        vec![ArrayVTable::id(&Decimal)]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let widen = array.as_::<Widen>();
        if !array.dtype().is_decimal() {
            vortex_bail!(
                "Widen serialization requires a decimal dtype, got {}",
                array.dtype()
            );
        }

        let values = widen.values();
        if !values.is::<Decimal>() {
            vortex_bail!(
                "Decimal Widen serialization requires a Decimal leaf, got {}",
                values.encoding_id()
            );
        }

        // Only the child's wire components are returned, not its dtype. The enclosing format and
        // serialization tree still use the original Widen's logical dtype and statistics.
        session.array_serialize(values)
    }

    fn deserialize(
        &self,
        _parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_bail!("Widen has no wire form yet (#10733)")
    }
}
