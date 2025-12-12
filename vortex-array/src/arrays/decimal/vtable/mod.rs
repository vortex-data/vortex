// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferHandle;
use vortex_dtype::DType;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_scalar::DecimalType;
use vortex_vector::decimal::DVector;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::ProstMetadata;
use crate::SerializeMetadata;
use crate::arrays::DecimalArray;
use crate::kernel::BindCtx;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable;
use crate::vtable::ArrayVTableExt;
use crate::vtable::NotSupported;
use crate::vtable::VTable;
use crate::vtable::ValidityVTableFromValidityHelper;

mod array;
mod canonical;
mod operations;
pub mod operator;
mod validity;
mod visitor;

pub use operator::DecimalMaskedValidityRule;

use crate::kernel::KernelRef;
use crate::kernel::kernel;
use crate::vtable::ArrayId;
use crate::vtable::ArrayVTable;

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
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<DecimalArray> {
        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let buffer = buffers[0].clone().try_to_bytes()?;

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

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() <= 1,
            "DecimalArray expects 0 or 1 child (validity), got {}",
            children.len()
        );

        if children.is_empty() {
            array.validity = Validity::from(array.dtype.nullability());
        } else {
            array.validity = Validity::Array(
                children
                    .into_iter()
                    .next()
                    .vortex_expect("children length already validated"),
            );
        }
        Ok(())
    }

    fn bind_kernel(array: &Self::Array, _ctx: &mut BindCtx) -> VortexResult<KernelRef> {
        use vortex_dtype::BigCast;

        match_each_decimal_value_type!(array.values_type(), |D| {
            // TODO(ngates): we probably shouldn't convert here... Do we allow larger P/S for a
            //  given physical type, because we know that our values actually fit?
            let min_value_type = DecimalType::smallest_decimal_value_type(&array.decimal_dtype());
            match_each_decimal_value_type!(min_value_type, |E| {
                let decimal_dtype = array.decimal_dtype();
                let buffer = array.buffer::<D>();
                let validity_mask = array.validity_mask();

                Ok(kernel(move || {
                    // Copy from D to E, possibly widening, possibly narrowing
                    let values =
                        Buffer::<E>::from_trusted_len_iter(buffer.iter().map(|d| {
                            <E as BigCast>::from(*d).vortex_expect("Decimal cast failed")
                        }));

                    Ok(unsafe {
                        DVector::<E>::new_unchecked(
                            // TODO(ngates): this is too small?
                            PrecisionScale::new_unchecked(
                                decimal_dtype.precision(),
                                decimal_dtype.scale(),
                            ),
                            values,
                            validity_mask,
                        )
                    }
                    .into())
                }))
            })
        })
    }
}

#[derive(Debug)]
pub struct DecimalVTable;

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_dtype::DecimalDType;

    use crate::ArrayContext;
    use crate::IntoArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::DecimalVTable;
    use crate::serde::ArrayParts;
    use crate::serde::SerializeOptions;
    use crate::validity::Validity;
    use crate::vtable::ArrayVTableExt;

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
