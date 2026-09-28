// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::VTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::Sequence;
use super::SequenceData;
use super::SequenceMetadata;

impl ArrayPlugin for Sequence {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<Sequence>().ok_or_else(|| {
            vortex_err!("Sequence plugin cannot serialize {}", array.encoding_id())
        })?;
        let metadata = SequenceMetadata {
            base: Some((&view.base()).into()),
            multiplier: Some((&view.multiplier()).into()),
        };

        Ok(Some(ArraySerialization::from_array(
            VTable::id(self),
            array,
            metadata.encode_to_vec(),
        )))
    }

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
            "Sequence plugin does not recognize serialized ID {serialized_id}"
        );
        vortex_ensure!(
            buffers.is_empty(),
            "SequenceArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.is_empty(),
            "SequenceArray expects 0 children, got {}",
            children.len()
        );
        let DType::Primitive(output_ptype, _) = dtype else {
            vortex_bail!(
                "only primitive dtypes are supported in SequenceArray currently, got {dtype}"
            );
        };
        let metadata = SequenceMetadata::decode(metadata)?;

        let base_metadata = metadata
            .base
            .as_ref()
            .ok_or_else(|| vortex_err!("base required"))?;

        let multiplier_metadata = metadata
            .multiplier
            .as_ref()
            .ok_or_else(|| vortex_err!("multiplier required"))?;

        // We go via Scalar to validate that the value is valid for the ptype.
        let base = Scalar::from_proto_value(
            base_metadata,
            &DType::Primitive(*output_ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array base should be a non-nullable primitive");

        // The serialized step preserves signedness independently of the output ptype.
        let multiplier_ptype = SequenceData::multiplier_ptype_from_proto(multiplier_metadata)?;
        let multiplier = Scalar::from_proto_value(
            multiplier_metadata,
            &DType::Primitive(multiplier_ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array multiplier should be a non-nullable primitive");

        let data =
            SequenceData::try_new(base, multiplier, *output_ptype, dtype.nullability(), len)?;
        Ok(
            Array::try_from_parts(ArrayParts::new(self.clone(), dtype.clone(), len, data))?
                .into_array(),
        )
    }
}
