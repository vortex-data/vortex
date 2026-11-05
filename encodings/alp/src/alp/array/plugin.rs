// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`ALP`].

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use super::ALP;
use super::ALPArrayExt;
use super::ALPData;
use super::ALPMetadata;
use crate::alp::Exponents;

impl ArrayPlugin for ALP {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn read_ids(&self) -> Vec<ArrayId> {
        vec![VTable::id(self)]
    }

    fn write_ids(&self) -> Vec<ArrayId> {
        vec![VTable::id(self)]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            VTable::id(self),
            "ALP plugin cannot serialize an array of another encoding",
        );
        let array = array.as_::<ALP>();
        let exponents = array.exponents();
        let metadata = ALPMetadata {
            exp_e: exponents.e as u32,
            exp_f: exponents.f as u32,
            patches: array
                .patches()
                .map(|p| p.to_metadata(array.len(), array.dtype()))
                .transpose()?,
        }
        .encode_to_vec();
        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array.array(),
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            VTable::id(self),
            "ALP plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            children,
            ..
        } = parts;
        let metadata = ALPMetadata::decode(metadata)?;
        let encoded_ptype = match &dtype {
            DType::Primitive(PType::F32, n) => DType::Primitive(PType::I32, *n),
            DType::Primitive(PType::F64, n) => DType::Primitive(PType::I64, *n),
            d => vortex_bail!("expected type: f32 or f64 but instead got {}", d),
        };
        let encoded = children.get(0, &encoded_ptype, len)?;

        let patches = metadata
            .patches
            .map(|p| {
                let indices = children.get(1, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(2, dtype, p.len()?)?;
                let chunk_offsets = p
                    .chunk_offsets_dtype()?
                    .map(|dtype| children.get(3, &dtype, usize::try_from(p.chunk_offsets_len())?))
                    .transpose()?;

                Patches::from_metadata(
                    len,
                    &p,
                    indices,
                    values,
                    chunk_offsets,
                    &mut session.create_execution_ctx(),
                )
            })
            .transpose()?;

        let slots = ALPData::make_slots(&encoded, patches.as_ref());
        let data = ALPData::new(
            Exponents {
                e: u8::try_from(metadata.exp_e)?,
                f: u8::try_from(metadata.exp_f)?,
            },
            patches,
        );
        Ok(Array::try_from_parts(ArrayParts::new(
            self.clone(),
            dtype.clone(),
            len,
            data,
            slots,
        ))?
        .into_array())
    }
}
