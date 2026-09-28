// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::validity_to_child;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::OnPair;
use super::OnPairArraySlotsExt;
use super::OnPairData;
use super::OnPairMetadata;
use super::OnPairSlots;

impl ArrayPlugin for OnPair {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<OnPair>()
            .ok_or_else(|| vortex_err!("OnPair plugin cannot serialize {}", array.encoding_id()))?;
        let dict_size = u32::try_from(view.dict_offsets().len().saturating_sub(1))
            .map_err(|_| vortex_err!("OnPair dict_size exceeds u32"))?;
        let codes_len = view.codes().len() as u64;
        let metadata = OnPairMetadata {
            uncompressed_lengths_ptype: view.uncompressed_lengths().dtype().as_ptype().into(),
            dict_size,
            codes_len,
            dict_offsets_ptype: view.dict_offsets().dtype().as_ptype().into(),
            codes_ptype: view.codes().dtype().as_ptype().into(),
            codes_offsets_ptype: view.codes_offsets().dtype().as_ptype().into(),
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
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
            "OnPair plugin does not recognize serialized ID {serialized_id}"
        );
        if buffers.len() != 1 {
            vortex_bail!(InvalidArgument: "Expected 1 buffer, got {}", buffers.len());
        }
        let metadata = OnPairMetadata::decode(metadata)?;
        let uncompressed_ptype = metadata.get_uncompressed_lengths_ptype()?;

        // Slot children do not persist their own lengths, so metadata records
        // the dictionary and code-stream sizes needed to deserialize them.
        let dict_offsets_len = metadata.dict_size as usize + 1;
        let codes_len = usize::try_from(metadata.codes_len)
            .map_err(|_| vortex_err!("codes_len {} overflows usize", metadata.codes_len))?;
        // The cascading compressor may have narrowed any of these integer
        // children to a tighter ptype; the recorded ptype tells the framework
        // exactly which dtype to materialise as.
        let dict_offsets_ptype = PType::try_from(metadata.dict_offsets_ptype).map_err(|_| {
            vortex_err!("invalid dict_offsets_ptype {}", metadata.dict_offsets_ptype)
        })?;
        let codes_ptype = PType::try_from(metadata.codes_ptype)
            .map_err(|_| vortex_err!("invalid codes_ptype {}", metadata.codes_ptype))?;
        let codes_offsets_ptype = PType::try_from(metadata.codes_offsets_ptype).map_err(|_| {
            vortex_err!(
                "invalid codes_offsets_ptype {}",
                metadata.codes_offsets_ptype
            )
        })?;
        let dict_offsets = children.get(
            0,
            &DType::Primitive(dict_offsets_ptype, Nullability::NonNullable),
            dict_offsets_len,
        )?;
        let codes = children.get(
            1,
            &DType::Primitive(codes_ptype, Nullability::NonNullable),
            codes_len,
        )?;
        let codes_offsets = children.get(
            2,
            &DType::Primitive(codes_offsets_ptype, Nullability::NonNullable),
            len + 1,
        )?;
        let uncompressed_lengths = children.get(
            3,
            &DType::Primitive(uncompressed_ptype, Nullability::NonNullable),
            len,
        )?;
        let validity = match children.len() {
            4 => Validity::from(dtype.nullability()),
            5 => Validity::Array(children.get(4, &Validity::DTYPE, len)?),
            other => vortex_bail!(InvalidArgument: "Expected 4 or 5 children, got {other}"),
        };

        let data = OnPairData::new(buffers[0].clone());
        let slots = OnPairSlots {
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity: validity_to_child(&validity, len),
        }
        .into_slots();
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
