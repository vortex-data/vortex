// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::DType;
#[cfg(test)]
use vortex_dtype::DecimalDType;
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};
use vortex_scalar::{DecimalValueType, NativeDecimalType, match_each_decimal_value_type};

use super::{DecimalArray, DecimalEncoding};
use crate::ProstMetadata;
use crate::arrays::DecimalVTable;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;

// The type of the values can be determined by looking at the type info...right?
#[derive(prost::Message)]
pub struct DecimalMetadata {
    #[prost(enumeration = "DecimalValueType", tag = "1")]
    pub(super) values_type: i32,
}

impl SerdeVTable<DecimalVTable> for DecimalVTable {
    type Metadata = ProstMetadata<DecimalMetadata>;

    fn metadata(array: &DecimalArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(ProstMetadata(DecimalMetadata {
            values_type: array.values_type() as i32,
        })))
    }

    fn build(
        _encoding: &DecimalEncoding,
        dtype: &DType,
        len: usize,
        metadata: &DecimalMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DecimalArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let Some(decimal_dtype) = dtype.as_decimal_opt() else {
            vortex_bail!("Expected Decimal dtype, got {:?}", dtype)
        };

        match_each_decimal_value_type!(metadata.values_type(), |D| {
            // Check and reinterpret-cast the buffer
            vortex_ensure!(
                buffer.is_aligned(Alignment::of::<D>()),
                "DecimalArray buffer not aligned for values type {:?}",
                D::VALUES_TYPE
            );
            let buffer = Buffer::<D>::from_byte_buffer(buffer);
            DecimalArray::try_new::<D>(buffer, *decimal_dtype, validity)
        })
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{ByteBufferMut, buffer};

    use super::*;
    use crate::serde::{ArrayParts, SerializeOptions};
    use crate::{ArrayContext, EncodingRef, IntoArray};

    #[test]
    fn test_array_serde() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 400i128, 500i128],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let dtype = array.dtype().clone();
        let ctx = ArrayContext::empty().with(EncodingRef::new_ref(DecimalEncoding.as_ref()));
        let out = array
            .into_array()
            .serialize(&ctx, &SerializeOptions::default())
            .unwrap();
        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend_from_slice(buf.as_ref());
        }

        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();

        let decoded = parts.decode(&ctx, &dtype, 5).unwrap();
        assert_eq!(decoded.encoding_id(), DecimalEncoding.id());
    }
}
