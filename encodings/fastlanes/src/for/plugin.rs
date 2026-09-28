// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ArrayPlugin implementation for FoR.

use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::smallvec::smallvec;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::FoR;
use crate::FoRData;
use crate::r#for::array::FoRArrayExt;

/// Serde for the [`FoR`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable serde. Direct registration of
/// [`FoR`] does not support serde.
#[derive(Clone, Debug)]
pub struct FoRPlugin;

impl ArrayPlugin for FoRPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&FoR)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array
            .as_opt::<FoR>()
            .ok_or_else(|| vortex_err!("FoR plugin cannot serialize {}", array.encoding_id()))?;
        // Note that we **only** serialize the optional scalar value (not including the dtype).
        let metadata = ScalarValue::to_proto_bytes(view.reference_scalar().value());
        Ok(Some(ArraySerialization::from_array(
            self.id(),
            array,
            metadata,
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            parts.serialized_id == self.id(),
            "FoR plugin does not recognize serialized ID {}",
            parts.serialized_id
        );
        vortex_ensure!(
            parts.buffers.is_empty(),
            "FoRArray expects 0 buffers, got {}",
            parts.buffers.len()
        );
        if parts.children.len() != 1 {
            vortex_bail!(
                "Expected 1 child for FoR encoding, found {}",
                parts.children.len()
            )
        }

        let scalar_value = ScalarValue::from_proto_bytes(parts.metadata, parts.dtype, session)?;
        let reference = Scalar::try_new(parts.dtype.clone(), scalar_value)?;
        let encoded = parts.children.get(0, parts.dtype, parts.len)?;
        let slots = smallvec![Some(encoded)];

        let data = FoRData::try_new(reference)?;
        Ok(Array::try_from_parts(
            ArrayParts::new(FoR, parts.dtype.clone(), parts.len, data).with_slots(slots),
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::ByteBufferMut;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::FoRArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    fn roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
        let array_ctx = ArrayContext::empty();
        let buffers = array.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
        let mut bytes = ByteBufferMut::empty();
        for buffer in buffers {
            bytes.extend_from_slice(&buffer);
        }
        SerializedArray::try_from(bytes.freeze())?.decode(
            array.dtype(),
            array.len(),
            &ReadContext::new(array_ctx.to_ids()),
            &SESSION,
        )
    }

    #[rstest]
    #[case::signed(FoR::encode(PrimitiveArray::from_iter([-5i32, 0, 7, 100]), &mut SESSION.create_execution_ctx()))]
    #[case::unsigned(FoR::encode(PrimitiveArray::from_iter(1_000u64..3_000), &mut SESSION.create_execution_ctx()))]
    #[case::nullable(FoR::encode(
        PrimitiveArray::from_option_iter([Some(10i16), None, Some(12)]),
        &mut SESSION.create_execution_ctx(),
    ))]
    fn serde_roundtrip(#[case] array: VortexResult<FoRArray>) -> VortexResult<()> {
        let array = array?;
        let serialization = SESSION
            .array_serialize(array.as_array())?
            .ok_or_else(|| vortex_err!("FoR must serialize"))?;
        assert_eq!(serialization.serialized_id, VTable::id(&FoR));
        assert_eq!(
            serialization.metadata,
            ScalarValue::to_proto_bytes::<Vec<u8>>(array.reference_scalar().value())
        );

        let read = roundtrip(array.as_array())?;
        assert_eq!(read.encoding_id(), VTable::id(&FoR));
        assert_arrays_eq!(read, array, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn vtable_serde_requires_plugin() -> VortexResult<()> {
        let array = FoR::encode(
            PrimitiveArray::from_iter([1u8, 2, 3]),
            &mut SESSION.create_execution_ctx(),
        )?;
        let session = vortex_array::array_session();
        session.arrays().register(FoR);
        assert!(session.array_serialize(&array.into_array()).is_err());
        Ok(())
    }
}
