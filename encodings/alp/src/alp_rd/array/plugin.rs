// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use prost::Message;
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
use vortex_array::patches::Patches;
use vortex_array::vtable::VTable;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::ALPRD;
use super::ALPRDArrayExt;
use super::ALPRDArraySlotsExt;
use super::ALPRDData;
use super::ALPRDMetadata;

impl ArrayPlugin for ALPRD {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<ALPRD>()
            .ok_or_else(|| vortex_err!("ALPRD plugin cannot serialize {}", array.encoding_id()))?;
        let dict = view
            .left_parts_dictionary()
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>();

        let metadata = ALPRDMetadata {
            right_bit_width: view.right_bit_width() as u32,
            dict_len: view.left_parts_dictionary().len() as u32,
            dict,
            left_parts_ptype: view.left_parts().dtype().as_ptype() as i32,
            patches: view
                .left_parts_patches()
                .map(|p| p.to_metadata(view.len(), p.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata,
        )))
    }

    #[allow(clippy::disallowed_methods)]
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
            buffers: _,
            children,
        } = parts;
        vortex_ensure!(
            serialized_id == VTable::id(self),
            "ALPRD plugin does not recognize serialized ID {serialized_id}"
        );
        let metadata = ALPRDMetadata::decode(metadata)?;
        if children.len() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                children.len()
            );
        }

        let left_parts_dtype = DType::Primitive(metadata.left_parts_ptype(), dtype.nullability());
        let left_parts = children.get(0, &left_parts_dtype, len)?;
        let left_parts_dictionary: Buffer<u16> = metadata.dict.as_slice()
            [0..metadata.dict_len as usize]
            .iter()
            .map(|&i| {
                u16::try_from(i)
                    .map_err(|_| vortex_err!("left_parts_dictionary code {i} does not fit in u16"))
            })
            .try_collect()?;

        let right_parts_dtype = match &dtype {
            DType::Primitive(PType::F32, _) => {
                DType::Primitive(PType::U32, Nullability::NonNullable)
            }
            DType::Primitive(PType::F64, _) => {
                DType::Primitive(PType::U64, Nullability::NonNullable)
            }
            _ => vortex_bail!("Expected f32 or f64 dtype, got {:?}", dtype),
        };
        let right_parts = children.get(1, &right_parts_dtype, len)?;

        let left_parts_patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(2, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(3, &left_parts_dtype.as_nonnullable(), p.len()?)?;

                Patches::new(
                    len,
                    p.offset()?,
                    indices,
                    values,
                    // TODO(0ax1): handle chunk offsets
                    None,
                )
            })
            .transpose()?;
        let slots = ALPRDData::make_slots(&left_parts, &right_parts, left_parts_patches.as_ref());
        let data = ALPRDData::new(
            left_parts_dictionary,
            u8::try_from(metadata.right_bit_width).map_err(|_| {
                vortex_err!(
                    "right_bit_width {} out of u8 range",
                    metadata.right_bit_width
                )
            })?,
            left_parts_patches,
        );
        Ok(Array::try_from_parts(
            ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots),
        )?
        .into_array())
    }
}
