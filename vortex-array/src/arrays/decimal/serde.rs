use serde::{Deserialize, Serialize};
use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::i256;

use super::{DecimalArray, DecimalEncoding};
use crate::arrays::NativeDecimalType;
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::EncodingVTable;
use crate::{
    Array, ArrayContext, ArrayRef, Canonical, DeserializeMetadata, EncodingId, SerdeMetadata,
};

/// Type of the decimal values.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecimalValueType {
    // TODO(aduffy): add I32, I64 once arrow-rs adds support for Decimal32/Decimal64.
    // I32 = 0,
    // I64 = 1,
    I128 = 2,
    I256 = 3,
}

// The type of the values can be determined by looking at the type info...right?
#[derive(Debug, Serialize, Deserialize)]
pub struct DecimalMetadata {
    pub(super) values_type: DecimalValueType,
}

impl EncodingVTable for DecimalEncoding {
    fn id(&self) -> EncodingId {
        EncodingId::new_ref("vortex.decimal")
    }

    fn decode(
        &self,
        parts: &ArrayParts,
        ctx: &ArrayContext,
        dtype: DType,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        if parts.nbuffers() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", parts.nbuffers());
        }
        let buffer = parts.buffer(0)?;

        let validity = if parts.nchildren() == 0 {
            Validity::from(dtype.nullability())
        } else if parts.nchildren() == 1 {
            let validity = parts.child(0).decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", parts.nchildren());
        };

        let decimal_dtype = match &dtype {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => vortex_bail!("Expected Decimal dtype, got {:?}", dtype),
        };

        let metadata = SerdeMetadata::<DecimalMetadata>::deserialize(parts.metadata())?;
        match metadata.values_type {
            DecimalValueType::I128 => {
                check_and_build_decimal::<i128>(len, buffer, decimal_dtype, validity)
            }
            DecimalValueType::I256 => {
                check_and_build_decimal::<i256>(len, buffer, decimal_dtype, validity)
            }
        }
    }

    fn encode(
        &self,
        input: &Canonical,
        _like: Option<&dyn Array>,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(input.clone().into_decimal()?.into_array()))
    }
}

fn check_and_build_decimal<T: NativeDecimalType>(
    array_len: usize,
    buffer: ByteBuffer,
    decimal_dtype: DecimalDType,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    // Assuming 16-byte alignment for decimal values
    if !buffer.is_aligned(Alignment::of::<T>()) {
        vortex_bail!("Buffer is not aligned to 16-byte boundary");
    }

    let buffer = Buffer::<T>::from_byte_buffer(buffer);
    if buffer.len() != array_len {
        vortex_bail!(
            "Buffer length {} does not match expected length {} for decimal values",
            buffer.len(),
            array_len,
        );
    }

    Ok(DecimalArray::new(buffer, decimal_dtype, validity).into_array())
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{ByteBufferMut, buffer};

    use super::*;
    use crate::Encoding;
    use crate::serde::SerializeOptions;

    #[test]
    fn test_array_serde() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 400i128, 500i128],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let dtype = array.dtype().clone();
        let ctx = ArrayContext::empty().with(DecimalEncoding.vtable());
        let out = array
            .into_array()
            .serialize(&ctx, &SerializeOptions::default());
        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend(buf.as_ref());
        }

        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();

        let decoded = parts.decode(&ctx, dtype, 5).unwrap();
        assert_eq!(decoded.encoding(), DecimalEncoding.id());
    }
}
