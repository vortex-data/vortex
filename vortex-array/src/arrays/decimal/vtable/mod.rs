// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::{Alignment, Buffer, ByteBuffer};
use vortex_dtype::{DType, NativeDecimalType, PrecisionScale, match_each_decimal_value_type};
use vortex_error::{VortexResult, vortex_bail, vortex_ensure};
use vortex_scalar::DecimalType;
use vortex_vector::Vector;
use vortex_vector::decimal::DVector;

use crate::arrays::DecimalArray;
use crate::execution::ExecutionCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::{ArrayVTableExt, NotSupported, VTable, ValidityVTableFromValidityHelper};
use crate::{DeserializeMetadata, ProstMetadata, SerializeMetadata, vtable};

mod array;
mod canonical;
mod operations;
pub mod operator;
mod validity;
mod visitor;

pub use operator::DecimalMaskedValidityRule;

use crate::vtable::{ArrayId, ArrayVTable};

vtable!(Decimal);

// The type of the values can be determined by looking at the type info...right?
#[derive(prost::Message)]
pub struct DecimalMetadata {
    #[prost(enumeration = "DecimalType", tag = "1")]
    pub(super) values_type: i32,
}

impl VTable for DecimalVTable {
    type Array = DecimalArray;

    type Metadata = ProstMetadata<DecimalMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type OperatorVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.decimal")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        DecimalVTable.as_vtable()
    }

    fn metadata(array: &DecimalArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(DecimalMetadata {
            values_type: array.values_type() as i32,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(bytes: &[u8]) -> VortexResult<Self::Metadata> {
        let metadata = ProstMetadata::<DecimalMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(metadata))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
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
                D::DECIMAL_TYPE
            );
            let buffer = Buffer::<D>::from_byte_buffer(buffer);
            DecimalArray::try_new::<D>(buffer, *decimal_dtype, validity)
        })
    }

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        match_each_decimal_value_type!(array.values_type(), |D| {
            Ok(unsafe {
                DVector::<D>::new_unchecked(
                    PrecisionScale::new_unchecked(array.precision(), array.scale()),
                    array.buffer::<D>(),
                    array.validity_mask(),
                )
            }
            .into())
        })
    }
}

#[derive(Clone, Debug)]
pub struct DecimalVTable;

#[cfg(test)]
mod tests {
    use vortex_buffer::{ByteBufferMut, buffer};
    use vortex_dtype::DecimalDType;

    use crate::arrays::{DecimalArray, DecimalVTable};
    use crate::serde::{ArrayParts, SerializeOptions};
    use crate::validity::Validity;
    use crate::vtable::ArrayVTableExt;
    use crate::{ArrayContext, IntoArray};

    #[test]
    fn test_array_serde() {
        let array = DecimalArray::new(
            buffer![100i128, 200i128, 300i128, 400i128, 500i128],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let dtype = array.dtype().clone();
        let ctx = ArrayContext::empty().with(DecimalVTable.as_vtable());
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
        assert!(decoded.is::<DecimalVTable>());
    }
}
