// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::Symbol;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::FSST;
use super::FSSTArraySlotsExt;
use super::FSSTData;
use super::FSSTMetadata;
use super::FSSTSlots;

impl ArrayPlugin for FSST {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<FSST>()
            .ok_or_else(|| vortex_err!("FSST plugin cannot serialize {}", array.encoding_id()))?;
        let codes_offsets = view.codes_offsets();
        let metadata = FSSTMetadata {
            uncompressed_lengths_ptype: view.uncompressed_lengths().dtype().as_ptype().into(),
            codes_offsets_ptype: codes_offsets.dtype().as_ptype().into(),
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata,
        )))
    }

    /// Deserializes an FSST array from its serialized components.
    ///
    /// Supports two serialization formats:
    ///
    /// ## Legacy format (2 buffers, 2 children)
    ///
    /// The original FSST layout stored the compressed codes as a full `VarBinArray` child.
    /// - **Buffers**: `[symbols, symbol_lengths]`
    /// - **Children**: `[codes (VarBinArray), uncompressed_lengths (Primitive)]`
    ///
    /// The codes VarBinArray child is decomposed: its bytes become the `codes_bytes` buffer,
    /// and its offsets/validity are extracted into slots.
    /// See `FSST::deserialize_legacy`.
    ///
    /// ## Current format (3 buffers, 2-3 children)
    ///
    /// The current layout stores the compressed bytes as a raw buffer alongside the symbol
    /// table, with offsets and validity as separate children.
    /// - **Buffers**: `[symbols, symbol_lengths, compressed_codes_bytes]`
    /// - **Children**: `[uncompressed_lengths, codes_offsets, (optional) codes_validity]`
    ///
    /// The `codes_bytes` buffer is stored directly in `FSSTData`. A `VarBinArray` for the
    /// codes can be reconstructed on demand via [`FSSTArrayExt::codes()`] using the bytes
    /// from `FSSTData` combined with offsets and validity from the array's slots.
    ///
    /// [`FSSTArrayExt::codes()`]: super::FSSTArrayExt::codes
    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let ArrayDeserialization {
            serialized_id,
            dtype,
            len,
            metadata,
            buffers,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "FSST plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = FSSTMetadata::decode(metadata)?;
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[0].clone().try_to_host_sync()?);
        let symbol_lengths = Buffer::<u8>::from_byte_buffer(buffers[1].clone().try_to_host_sync()?);

        let mut ctx = session.create_execution_ctx();
        if buffers.len() == 2 {
            return Ok(Array::try_from_parts(Self::deserialize_legacy(
                self,
                dtype,
                len,
                &metadata,
                &symbols,
                &symbol_lengths,
                children,
                &mut ctx,
            )?)?
            .into_array());
        }

        if buffers.len() == 3 {
            let uncompressed_lengths = children.get(
                0,
                &DType::Primitive(
                    metadata.get_uncompressed_lengths_ptype()?,
                    Nullability::NonNullable,
                ),
                len,
            )?;

            let codes_bytes = buffers[2].clone();
            let codes_offsets = children.get(
                1,
                &DType::Primitive(
                    PType::try_from(metadata.codes_offsets_ptype)?,
                    Nullability::NonNullable,
                ),
                // VarBin offsets are len + 1
                len + 1,
            )?;

            let codes_validity = if children.len() == 2 {
                Validity::from(dtype.nullability())
            } else if children.len() == 3 {
                let validity = children.get(2, &Validity::DTYPE, len)?;
                Validity::Array(validity)
            } else {
                vortex_bail!("Expected 2 or 3 children, got {}", children.len());
            };

            FSSTData::validate_parts(
                symbols.as_slice(),
                symbol_lengths.as_slice(),
                &codes_bytes,
                &codes_offsets,
                dtype.nullability(),
                &uncompressed_lengths,
                dtype,
                len,
                &mut ctx,
            )?;
            let slots = FSSTSlots {
                uncompressed_lengths,
                codes_offsets,
                codes_validity: validity_to_child(&codes_validity, len),
            }
            .into_slots();
            let data = FSSTData::try_new(symbols, symbol_lengths, codes_bytes, len)?;
            return Ok(Array::try_from_parts(
                ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
            )?
            .into_array());
        }

        vortex_bail!(
            "InvalidArgument: Expected 2 or 3 buffers, got {}",
            buffers.len()
        );
    }
}
