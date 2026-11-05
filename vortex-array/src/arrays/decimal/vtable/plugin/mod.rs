// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for `vortex.decimal` nodes, including legacy nodes.
//!
//! [`StoredDecimal::is_legacy`] defines a legacy node. [`Decimal`] rejects legacy nodes, and
//! [`DecimalPlugin`] reads them as the canonical array or a [`Widen`].
//!
//! Each legacy read scans the valid values once, because the deserializer attaches stored
//! statistics only after the plugin returns. This cost is not limited to old files, because IPC
//! and writes without a compressor still write a decimal `Widen` as a legacy node.

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

/// Reads every `vortex.decimal` node as the canonical decimal array or a decimal [`Widen`].
///
/// A canonical node keeps its buffer without a copy or a scan, so the buffer can stay on a device,
/// and its valid values are trusted to fit the precision. A legacy node goes through
/// [`WidenArray::from_stored_decimal`], which copies it to the host to scan its valid values.
///
/// The default [`ArraySession`] registers this plugin in place of [`Decimal`], and the plugin
/// writes arrays as [`Decimal`] does.
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

        if !stored.is_legacy() {
            return Ok(Array::<Decimal>::try_from_parts(stored.into_array_parts()?)?.into_array());
        }

        // This is a legacy node, whose width a `Decimal` leaf no longer allows, so the valid values
        // pick the canonical array or a `Widen`.
        let parts = DecimalDataParts {
            decimal_dtype: stored.decimal_dtype,
            values: stored.values,
            values_type: stored.values_type,
            validity: stored.validity,
        };

        // Stored statistics are attached only after this returns, so no range is known yet.
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

#[cfg(test)]
mod tests;
