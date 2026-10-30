// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `vortex.decimal`, with reads at every stored width of a decimal node.
//!
//! Older writers stored a decimal dtype at any width that held its values, and [`WidenPlugin`]
//! writes a decimal `Widen` as the narrow decimal node of its child. The wire ID is the same for
//! every width. [`DecimalPlugin`] reads a non-canonical node into the canonical array or a decimal
//! [`Widen`], while [`Decimal`] itself rejects such a node.
//!
//! IPC and writes without a compressor write a narrow node for every decimal `Widen`. Each read
//! of such a node scans its valid values once, because the deserializer attaches stored statistics
//! only after the plugin returns. The default file writer executes to canonical before compressing,
//! so its decimal compression path is unchanged.
//!
//! [`WidenPlugin`]: crate::arrays::widen::WidenPlugin

use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::Decimal;
use super::StoredDecimal;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayDeserialization;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::array::ArraySerialization;
use crate::array::VTable;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalDataParts;
use crate::expr::stats::Precision;
use crate::validity::Validity;

/// Reads every `vortex.decimal` node into the canonical decimal array or a decimal [`Widen`].
///
/// A node at the canonical width of its dtype reads as a [`Decimal`] leaf with the same buffer,
/// with no copy and no scan. Its buffer can stay on a device, and its valid values are trusted to
/// fit the precision. [`WidenArray::from_stored_decimal`] reads any other node on the host into
/// the canonical array or a `Widen`.
///
/// A non-canonical node is copied to the host on CUDA read paths, because CUDA has no `Widen`
/// kernel. If the copy to the host fails, the read returns an error.
///
/// The default [`ArraySession`] registers this plugin in place of [`Decimal`]. The plugin writes
/// arrays as [`Decimal`] does.
///
/// [`ArraySession`]: crate::session::ArraySession
#[derive(Clone, Debug)]
pub struct DecimalPlugin;

impl ArrayPlugin for DecimalPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&Decimal)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![self.id()]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![self.id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        ArrayPlugin::serialize(&Decimal, array, session)
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            self.id(),
            "DecimalPlugin reads only its own serialized ID",
        );

        let stored = StoredDecimal::deserialize(
            parts.dtype,
            parts.len,
            parts.metadata,
            parts.buffers,
            parts.children,
        )?;

        if stored.values_type == stored.decimal_dtype.canonical_values_type() {
            return Ok(Array::<Decimal>::try_from_parts(stored.into_array_parts()?)?.into_array());
        }

        // `from_stored_decimal` copies the values to the host itself, but not the validity.
        let parts = DecimalDataParts {
            decimal_dtype: stored.decimal_dtype,
            values: stored.values,
            values_type: stored.values_type,
            validity: validity_on_host(stored.validity)?,
        };

        WidenArray::from_stored_decimal(
            parts,
            Precision::Absent,
            &mut session.create_execution_ctx(),
        )
    }

    fn is_supported_encoding(&self, id: &ArrayId) -> bool {
        *id == VTable::id(&Decimal) || *id == VTable::id(&Widen)
    }
}

/// Returns `validity` with every buffer of its array on the host.
///
/// [`WidenArray::from_stored_decimal`] executes the validity on the host, and a host read of a
/// device buffer panics, so a non-canonical node read from device segments needs this copy first.
fn validity_on_host(validity: Validity) -> VortexResult<Validity> {
    match validity {
        Validity::Array(array) => Ok(Validity::Array(array.try_into_host_sync()?)),
        Validity::NonNullable | Validity::AllValid | Validity::AllInvalid => Ok(validity),
    }
}

#[cfg(test)]
mod tests;
