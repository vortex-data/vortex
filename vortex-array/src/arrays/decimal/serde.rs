use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{DType, DecimalDType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_scalar::i256;

use super::{DecimalArray, DecimalVTable};
use crate::arrays::{DecimalEncoding, NativeDecimalType};
use crate::serde::ArrayParts;
use crate::validity::Validity;
use crate::vtable::SerdeVTable;
use crate::{ArrayContext, ProstMetadata};

/// Type of the decimal values.
#[derive(Clone, Copy, Debug, prost::Enumeration, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum DecimalValueType {
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
    I128 = 4,
    I256 = 5,
}

// The type of the values can be determined by looking at the type info...right?
#[derive(prost::Message)]
pub struct DecimalMetadata {
    #[prost(enumeration = "DecimalValueType", tag = "1")]
    pub(super) values_type: i32,
}

impl SerdeVTable<DecimalVTable> for DecimalVTable {
    type Metadata = ProstMetadata<DecimalMetadata>;

    fn metadata(array: &DecimalArray) -> Option<Self::Metadata> {
        Some(ProstMetadata(DecimalMetadata {
            values_type: array.values_type() as i32,
        }))
    }

    fn decode(
        _encoding: &DecimalEncoding,
        dtype: DType,
        len: usize,
        metadata: &DecimalMetadata,
        buffers: &[ByteBuffer],
        children: &[ArrayParts],
        ctx: &ArrayContext,
    ) -> VortexResult<DecimalArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone();

        let validity = if children.len() == 0 {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children[0].decode(ctx, Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Expected 0 or 1 child, got {}", children.len());
        };

        let decimal_dtype = match &dtype {
            DType::Decimal(decimal_dtype, _) => *decimal_dtype,
            _ => vortex_bail!("Expected Decimal dtype, got {:?}", dtype),
        };

        match metadata.values_type() {
            DecimalValueType::I8 => {
                check_and_build_decimal::<i8>(len, buffer, decimal_dtype, validity)
            }
            DecimalValueType::I16 => {
                check_and_build_decimal::<i16>(len, buffer, decimal_dtype, validity)
            }
            DecimalValueType::I32 => {
                check_and_build_decimal::<i32>(len, buffer, decimal_dtype, validity)
            }
            DecimalValueType::I64 => {
                check_and_build_decimal::<i64>(len, buffer, decimal_dtype, validity)
            }
            DecimalValueType::I128 => {
                check_and_build_decimal::<i128>(len, buffer, decimal_dtype, validity)
            }
            DecimalValueType::I256 => {
                check_and_build_decimal::<i256>(len, buffer, decimal_dtype, validity)
            }
        }
    }
}

fn check_and_build_decimal<T: NativeDecimalType>(
    array_len: usize,
    buffer: ByteBuffer,
    decimal_dtype: DecimalDType,
    validity: Validity,
) -> VortexResult<DecimalArray> {
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

    Ok(DecimalArray::new(buffer, decimal_dtype, validity))
}

#[macro_export]
macro_rules! match_each_decimal_value {
    ($self:expr, | $_:tt $value:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $value:ident ) => ( $($body)* )}
        macro_rules! __with__ {( $_ $value:ident ) => ( $($body)* )}
        match $self {
            DecimalValue::I8(v) => __with__! { v },
            DecimalValue::I16(v) => __with__! { v },
            DecimalValue::I32(v) => __with__! { v },
            DecimalValue::I64(v) => __with__! { v },
            DecimalValue::I128(v) => __with__! { v },
            DecimalValue::I256(v) => __with__! { v },
        }
    });
}

/// Macro to match over each decimal value type, binding the corresponding native type (from `DecimalValueType`)
#[macro_export]
macro_rules! match_each_decimal_value_type {
    ($self:expr, | $_:tt $enc:ident | $($body:tt)*) => ({
        macro_rules! __with__ {( $_ $enc:ident ) => ( $($body)* )}
        use $crate::arrays::DecimalValueType;
        use vortex_scalar::i256;
        match $self {
            DecimalValueType::I8 => __with__! { i8 },
            DecimalValueType::I16 => __with__! { i16 },
            DecimalValueType::I32 => __with__! { i32 },
            DecimalValueType::I64 => __with__! { i64 },
            DecimalValueType::I128 => __with__! { i128 },
            DecimalValueType::I256 => __with__! { i256 },
        }
    });
    ($self:expr, | ($_0:tt $enc:ident, $_1:tt $dv_path:ident) | $($body:tt)*) => ({
        macro_rules! __with2__ { ( $_0 $enc:ident, $_1 $dv_path:ident ) => ( $($body)* ) }
        use $crate::arrays::DecimalValueType;
        use vortex_scalar::i256;
        use vortex_scalar::DecimalValue::*;

        match $self {
            DecimalValueType::I8 => __with2__! { i8, I8 },
            DecimalValueType::I16 => __with2__! { i16, I16 },
            DecimalValueType::I32 => __with2__! { i32, I32 },
            DecimalValueType::I64 => __with2__! { i64, I64 },
            DecimalValueType::I128 => __with2__! { i128, I128 },
            DecimalValueType::I256 => __with2__! { i256, I256 },
        }
    });
}

#[cfg(test)]
mod tests {
    use vortex_buffer::{ByteBufferMut, buffer};

    use super::*;
    use crate::serde::SerializeOptions;
    use crate::{EncodingRef, IntoArray};

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
            .serialize(&ctx, &SerializeOptions::default());
        // Concat into a single buffer
        let mut concat = ByteBufferMut::empty();
        for buf in out {
            concat.extend(buf.as_ref());
        }

        let concat = concat.freeze();

        let parts = ArrayParts::try_from(concat).unwrap();

        let decoded = parts.decode(&ctx, dtype, 5).unwrap();
        assert_eq!(decoded.encoding_id(), DecimalEncoding.id());
    }
}
