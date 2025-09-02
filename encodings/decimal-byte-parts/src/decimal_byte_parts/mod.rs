// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod compute;
mod serde;

use std::ops::Range;

use vortex_array::arrays::DecimalArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityChild,
    ValidityHelper, ValidityVTableFromChild,
};
use vortex_array::{
    Array, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable,
};
use vortex_dtype::{DType, DecimalDType, match_each_signed_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::{DecimalValue, Scalar};

vtable!(DecimalByteParts);

impl VTable for DecimalBytePartsVTable {
    type Array = DecimalBytePartsArray;
    type Encoding = DecimalBytePartsEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type SerdeVTable = Self;
    type PipelineVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.decimal_byte_parts")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(DecimalBytePartsEncoding.as_ref())
    }
}

/// This array encodes decimals as between 1-4 columns of primitive typed children.
/// The most significant part (msp) sorting the most significant decimal bits.
/// This array must be signed and is nullable iff the decimal is nullable.
///
/// e.g. for a decimal i128 \[ 127..64 | 64..0 \] msp = 127..64 and lower_part\[0\] = 64..0
#[derive(Clone, Debug)]
pub struct DecimalBytePartsArray {
    msp: ArrayRef,
    // NOTE: the lower_parts is currently unused, we reserve this field so that it is properly
    //  read/written during serde, but provide no constructor to initialize this to anything
    //  other than the empty Vec.
    _lower_parts: Vec<ArrayRef>,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalBytePartsArray {
    pub fn try_new(msp: ArrayRef, decimal_dtype: DecimalDType) -> VortexResult<Self> {
        if !msp.dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        let nullable = msp.dtype().nullability();
        Ok(Self {
            msp,
            _lower_parts: Vec::new(),
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        })
    }

    pub(crate) unsafe fn new_unchecked(msp: ArrayRef, decimal_dtype: DecimalDType) -> Self {
        let nullable = msp.dtype().nullability();
        Self {
            msp,
            _lower_parts: Vec::new(),
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        }
    }

    pub fn decimal_dtype(&self) -> &DecimalDType {
        self.dtype
            .as_decimal_opt()
            .vortex_expect("must be a decimal dtype")
    }

    pub(crate) fn msp(&self) -> &ArrayRef {
        &self.msp
    }
}

#[derive(Clone, Debug)]
pub struct DecimalBytePartsEncoding;

impl ArrayVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn len(array: &DecimalBytePartsArray) -> usize {
        array.msp.len()
    }

    fn dtype(array: &DecimalBytePartsArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &DecimalBytePartsArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn canonicalize(array: &DecimalBytePartsArray) -> Canonical {
        // TODO(joe): support parts len != 1
        let prim = array.msp.to_primitive();
        // Depending on the decimal type and the min/max of the primitive array we can choose
        // the correct buffer size

        match_each_signed_integer_ptype!(prim.ptype(), |P| {
            Canonical::Decimal(DecimalArray::new(
                prim.buffer::<P>(),
                *array.decimal_dtype(),
                prim.validity().clone(),
            ))
        })
    }
}

impl OperationsVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn slice(array: &DecimalBytePartsArray, range: Range<usize>) -> ArrayRef {
        // SAFETY: slicing encoded MSP does not change the encoded values
        unsafe {
            DecimalBytePartsArray::new_unchecked(array.msp.slice(range), *array.decimal_dtype())
                .into_array()
        }
    }

    #[allow(clippy::useless_conversion)]
    fn scalar_at(array: &DecimalBytePartsArray, index: usize) -> Scalar {
        // TODO(joe): support parts len != 1
        let scalar = array.msp.scalar_at(index);

        // Note. values in msp, can only be signed integers upto size i64.
        let primitive_scalar = scalar.as_primitive();
        // TODO(joe): extend this to support multiple parts.
        let value = primitive_scalar.as_::<i64>().vortex_expect("non-null");
        Scalar::new(array.dtype.clone(), DecimalValue::I64(value).into())
    }
}

impl ValidityChild<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn validity_child(array: &DecimalBytePartsArray) -> &dyn Array {
        // validity stored in 0th child
        array.msp.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::arrays::{BoolArray, PrimitiveArray};
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar};

    use crate::DecimalBytePartsArray;

    #[test]
    fn test_scalar_at_decimal_parts() {
        let decimal_dtype = DecimalDType::new(8, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let array = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(
                buffer![100i32, 200i32, 400i32],
                Validity::Array(BoolArray::from_iter(vec![false, true, true]).to_array()),
            )
            .to_array(),
            decimal_dtype,
        )
        .unwrap()
        .to_array();

        assert_eq!(Scalar::null(dtype.clone()), array.scalar_at(0));
        assert_eq!(
            Scalar::new(dtype.clone(), DecimalValue::I64(200).into()),
            array.scalar_at(1)
        );
        assert_eq!(
            Scalar::new(dtype, DecimalValue::I64(400).into()),
            array.scalar_at(2)
        );
    }
}
