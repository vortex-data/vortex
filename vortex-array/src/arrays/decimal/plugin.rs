// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Preserve the historical decimal wire representation.
//!
//! In-memory decimals have an integer child. This plugin serializes its materialized stored width
//! so existing readers retain the same buffer and validity format.

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use super::Decimal;
use super::DecimalArray;
use super::DecimalArrayExt;
use super::vtable::DecimalMetadata;
use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::array::validity_to_child;

/// Reads and writes the historical decimal buffer representation.
///
/// The in-memory decimal has one integer child. Serialization materializes only that child's
/// stored width, so existing readers can decode it without expanding values to the precision.
#[derive(Clone, Debug)]
pub struct DecimalPlugin;

impl ArrayPlugin for DecimalPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&Decimal)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure!(
            array.is::<Decimal>(),
            "Decimal plugin cannot serialize {}",
            array.encoding_id()
        );
        let mut ctx = session.create_execution_ctx();
        let array = array.as_::<Decimal>().materialize_values(&mut ctx)?;
        let metadata = DecimalMetadata {
            values_type: array.values_type() as i32,
        }
        .encode_to_vec();
        let validity = validity_to_child(&array.validity()?, array.len());
        Ok(Some(ArraySerialization::new(
            self.id(),
            metadata,
            vec![array.buffer_handle().to_host_sync()],
            validity.into_iter().collect(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            parts.serialized_id == self.id(),
            "Unknown decimal ID {}",
            parts.serialized_id
        );
        Ok(DecimalArray::try_from_parts(VTable::deserialize(
            &Decimal,
            parts.dtype,
            parts.len,
            parts.metadata,
            parts.buffers,
            parts.children,
            session,
        )?)?
        .into_array())
    }
}
