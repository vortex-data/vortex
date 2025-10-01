// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::arrays::{MaskedArray, MaskedEncoding, MaskedVTable};
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{SerdeVTable, VisitorVTable};
use crate::{ArrayBufferVisitor, ArrayChildVisitor, EmptyMetadata};

impl SerdeVTable<MaskedVTable> for MaskedVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_array: &MaskedArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _encoding: &MaskedEncoding,
        dtype: &DType,
        len: usize,
        _metadata: &Self::Metadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<MaskedArray> {
        if !buffers.is_empty() {
            vortex_bail!("Expected 0 buffer, got {}", buffers.len());
        }

        let child = children.get(0, &dtype.as_nonnullable(), len)?;

        let validity = if children.len() == 1 {
            Validity::from(dtype.nullability())
        } else if children.len() == 2 {
            let validity = children.get(1, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!(
                "`MaskedArray::build` expects 1 or 2 children, got {}",
                children.len()
            );
        };

        MaskedArray::try_new(child, validity)
    }
}

impl VisitorVTable<MaskedVTable> for MaskedVTable {
    fn visit_buffers(_array: &MaskedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &MaskedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", array.child.as_ref());
        visitor.visit_validity(&array.validity, array.child.len());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;

    use super::*;
    use crate::arrays::{MaskedArray, PrimitiveArray};
    use crate::serde::{ArrayParts, SerializeOptions};
    use crate::{ArrayContext, EncodingRef, IntoArray};

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_serde_roundtrip(#[case] array: MaskedArray) {
        let dtype = array.dtype().clone();
        let len = array.len();
        let ctx = ArrayContext::empty().with(EncodingRef::new_ref(MaskedEncoding.as_ref()));

        let serialized = array
            .to_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();
        let decoded = parts.decode(&ctx, &dtype, len).unwrap();

        assert_eq!(decoded.encoding_id(), MaskedEncoding.id());
        assert_eq!(
            array.as_ref().display_values().to_string(),
            decoded.display_values().to_string()
        );
    }
}
