// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] serialization and deserialization for `vortex.rans`.
//!
//! Wire slot `0` always corresponds to [`RansSlots::VALIDITY_CHILD`] (`Validity::DTYPE`), using a
//! zero-buffer `ConstantArray(true, len)` when non-nullable or all-valid so any future slots keep
//! stable 1-to-1 wire child indices. Readers ignore trailing children and trailing node buffers
//! beyond the declared ones for forward compatibility.

use std::sync::Arc;

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Rans;
use crate::RansArrayExt;
use crate::RansArraySlotsExt;
use crate::RansData;
use crate::RansMetadata;
use crate::codec::BLOCK_VALUES;
use crate::codec::PROB_SCALE;
use crate::codec::RansDecodeTable;
use crate::codec::pack_sym_freqs;
use crate::codec::unpack_sym_freqs;

/// The serialized and in-memory [`ArrayId`] of [`Rans`] (`"vortex.rans"`).
pub fn rans_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.rans");
    *ID
}

/// [`ArrayPlugin`] implementation for [`Rans`].
#[derive(Clone, Debug)]
pub struct RansPlugin;

impl ArrayPlugin for RansPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&Rans)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<Rans>()
            .ok_or_else(|| vortex_err!("RansPlugin cannot serialize {}", array.encoding_id()))?;

        let metadata = RansMetadata {
            offset: u32::from(view.offset()),
            has_extra_bits: view.has_extra_bits(),
            sym_freqs: pack_sym_freqs(view.sym_freqs()),
            payload_base_offset: view.payload_base_offset(),
        };

        let validity_wire_child = view
            .validity_child()
            .cloned()
            .unwrap_or_else(|| ConstantArray::new(true, view.len()).into_array());

        Ok(Some(ArraySerialization::new(
            rans_id(),
            metadata.encode_to_vec(),
            array.buffers(),
            vec![validity_wire_child],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            rans_id(),
            "RansPlugin does not recognize serialized ID {}",
            parts.serialized_id
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;

        vortex_ensure!(
            dtype.is_int(),
            "RansArray requires an integer dtype, got {dtype}"
        );
        vortex_ensure!(
            buffers.len() >= 2,
            "RansArray expects at least 2 buffers (block_headers, payload), got {}",
            buffers.len()
        );
        vortex_ensure!(
            !children.is_empty(),
            "RansArray expects at least 1 child (slot 0 validity), got 0"
        );

        let metadata = RansMetadata::decode(metadata)?;
        vortex_ensure!(
            usize::try_from(metadata.offset).is_ok_and(|off| off < BLOCK_VALUES),
            "RansMetadata offset {} must be < {BLOCK_VALUES}",
            metadata.offset
        );
        let offset = u16::try_from(metadata.offset)?;
        if len == 0 {
            vortex_ensure_eq!(
                offset,
                0,
                "Empty RansArray must have offset 0, got {offset}"
            );
        }

        let sym_freqs = if len == 0 && metadata.sym_freqs.is_empty() {
            vec![(0u8, u16::try_from(PROB_SCALE)?)]
        } else {
            unpack_sym_freqs(&metadata.sym_freqs)?
        };
        let (decode_table, computed_extra_bits) = RansDecodeTable::try_from_sym_freqs(&sym_freqs)?;
        vortex_ensure_eq!(
            metadata.has_extra_bits,
            computed_extra_bits,
            "RansMetadata has_extra_bits disagrees with active symbol table"
        );

        let validity_arr = children.get(0, &Validity::DTYPE, len)?;
        let validity = match dtype.nullability() {
            Nullability::NonNullable => {
                if let Some(c) = validity_arr.as_opt::<Constant>()
                    && c.scalar().value() == Some(&ScalarValue::Bool(true))
                {
                    Validity::NonNullable
                } else {
                    vortex_bail!("Non-nullable RansArray requires Constant(true) in slot 0");
                }
            }
            Nullability::Nullable => child_to_validity(Some(&validity_arr), Nullability::Nullable),
        };

        let data = RansData::try_new(
            offset,
            metadata.payload_base_offset,
            metadata.has_extra_bits,
            Arc::from(sym_freqs),
            Arc::new(decode_table),
            buffers[0].clone(),
            buffers[1].clone(),
        )?;
        Ok(Rans::try_new(dtype.clone(), len, data, validity)?.into_array())
    }
}
