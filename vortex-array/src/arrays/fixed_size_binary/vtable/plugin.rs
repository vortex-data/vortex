// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ArrayPlugin`] implementation for [`FixedSizeBinary`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;

use crate::ArrayDeserialization;
use crate::ArrayId;
use crate::ArrayPlugin;
use crate::ArrayRef;
use crate::ArraySerialization;
use crate::IntoArray;
use crate::VTable;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::fixed_width::vtable as fixed_width;
use crate::dtype::DType;

impl ArrayPlugin for FixedSizeBinary {
    fn id(&self) -> ArrayId {
        VTable::id(self)
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        vortex_ensure_eq!(
            array.encoding_id(),
            ArrayPlugin::id(self),
            "FixedSizeBinary plugin cannot serialize an array of another encoding",
        );
        Ok(Some(ArraySerialization::from_array(
            ArrayPlugin::id(self),
            array,
            vec![],
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            parts.serialized_id,
            ArrayPlugin::id(self),
            "FixedSizeBinary plugin does not recognize the serialized ID",
        );
        let ArrayDeserialization {
            dtype,
            len,
            metadata,
            buffers,
            children,
            ..
        } = parts;
        if !metadata.is_empty() {
            vortex_bail!(
                "FixedSizeBinaryArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let DType::FixedSizeBinary(byte_width, _) = dtype else {
            vortex_bail!("FixedSizeBinaryArray requires a fixed-size binary dtype, got {dtype}");
        };
        let values = fixed_width::single_buffer(buffers)?;
        let validity = fixed_width::deserialize_validity(dtype.nullability(), len, children)?;

        Ok(FixedSizeBinaryArray::try_new_handle(values, *byte_width, len, validity)?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeBinaryArray;
    use crate::assert_arrays_eq;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::non_nullable(FixedSizeBinaryArray::from_iter([b"ab", b"cd", b"ef"], 2))]
    #[case::nullable(FixedSizeBinaryArray::from_option_iter(
        [Some(b"abc"), None, Some(b"xyz")],
        3,
    ))]
    #[case::zero_width(FixedSizeBinaryArray::from_option_iter(
        [Some(b""), None, Some(b"")],
        0,
    ))]
    fn serde_round_trip(#[case] array: FixedSizeBinaryArray) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = array.dtype().clone();
        let len = array.len();

        let array_ctx = ArrayContext::empty();
        let serialized = array.clone().into_array().serialize(
            &array_ctx,
            &SESSION,
            &SerializeOptions::default(),
        )?;

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &SESSION)?;

        assert_arrays_eq!(decoded, array, &mut ctx);
        Ok(())
    }
}
