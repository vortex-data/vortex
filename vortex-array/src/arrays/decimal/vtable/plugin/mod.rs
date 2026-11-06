// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`Decimal`], including legacy nodes.
//!
//! [`Decimal`] writes a [`DecimalArray`] as a `vortex.decimal` node, and reads every such node as
//! the canonical decimal array or a decimal [`Widen`]. [`StoredDecimal::is_legacy`] defines a
//! legacy node.
//!
//! A canonical node keeps its buffer without a copy or a scan, so the buffer can stay on a device,
//! and its valid values are trusted to fit the precision. A legacy node goes through
//! [`WidenArray::from_stored_decimal`], which scans its valid values on the host, so a legacy node
//! in device memory fails to read.
//!
//! Each legacy read scans the valid values once, because the deserializer attaches stored
//! statistics only after the plugin returns. This cost is not limited to old files, because IPC
//! and writes without a compressor still write a decimal `Widen` as a legacy node.
//!
//! [`DecimalArray`]: super::DecimalArray

mod stored;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use self::stored::StoredDecimal;
use super::Decimal;
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
use crate::dtype::DecimalType;
use crate::expr::stats::Precision;

/// The metadata of a `vortex.decimal` node.
#[derive(prost::Message)]
struct DecimalMetadata {
    /// The [`DecimalType`] discriminant of the stored width.
    ///
    /// The width is recorded apart from the dtype, because a legacy node can store a width other
    /// than the canonical width of its precision.
    #[prost(int32, tag = "1")]
    values_type: i32,
}

impl DecimalMetadata {
    fn get_values_type(&self) -> VortexResult<DecimalType> {
        DecimalType::try_from(self.values_type).map_err(|_| {
            vortex_err!(
                "DecimalMetadata values_type must be a known DecimalType, got {}",
                self.values_type
            )
        })
    }
}

impl ArrayPlugin for Decimal {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let Some(decimal) = array.as_opt::<Decimal>() else {
            vortex_bail!(
                "Decimal plugin serializes only Decimal arrays, got {}",
                array.encoding_id()
            );
        };
        let metadata = DecimalMetadata {
            values_type: decimal.values_type() as i32,
        };

        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            metadata.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "Decimal plugin reads only its own serialized ID",
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
