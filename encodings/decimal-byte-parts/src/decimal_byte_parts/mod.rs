mod compute;
mod serde;

use itertools::Itertools;
use vortex_array::arrays::DecimalArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityChild,
    ValidityHelper, ValidityVTableFromChild,
};
use vortex_array::{Array, ArrayRef, Canonical, EncodingId, EncodingRef, vtable};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, DecimalDType, PType, match_each_signed_integer_ptype};
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
    lower_parts: Vec<ArrayRef>,
    dtype: DType,
    stats_set: ArrayStats,
}

impl DecimalBytePartsArray {
    pub fn try_new(
        msp: ArrayRef,
        lower_parts: Vec<ArrayRef>,
        decimal_dtype: DecimalDType,
    ) -> VortexResult<Self> {
        if !lower_parts.is_empty() {
            // TODO(joe): remove this constraint.
            vortex_bail!("DecimalBytePartsArray doesn't support lower parts arrays")
        }

        if !(0usize..=3).contains(&lower_parts.iter().len()) {
            vortex_bail!(
                "DecimalBytePartsArray lower_parts must have between 0..=3 children, instead given: {}",
                lower_parts.len()
            )
        }

        if !msp.dtype().is_signed_int() {
            vortex_bail!("decimal bytes parts, first part must be a signed array")
        }

        if lower_parts
            .iter()
            .any(|a| a.dtype() != &DType::Primitive(PType::U64, NonNullable))
        {
            vortex_bail!("decimal bytes parts 2nd to 4th must be non-nullable u64 primitive typed")
        }

        let nullable = msp.dtype().nullability();
        Ok(Self {
            msp,
            lower_parts,
            dtype: DType::Decimal(decimal_dtype, nullable),
            stats_set: Default::default(),
        })
    }

    pub fn decimal_dtype(&self) -> &DecimalDType {
        self.dtype
            .as_decimal()
            .vortex_expect("must be a decimal dtype")
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
    fn canonicalize(array: &DecimalBytePartsArray) -> VortexResult<Canonical> {
        // TODO(joe): support parts len != 1
        assert!(array.lower_parts.is_empty());
        let prim = array.msp.to_canonical()?.into_primitive()?;
        // Depending on the decimal type and the min/max of the primitive array we can choose
        // the correct buffer size

        let res = match_each_signed_integer_ptype!(prim.ptype(), |P| {
            Canonical::Decimal(DecimalArray::new(
                prim.buffer::<P>(),
                *array.decimal_dtype(),
                prim.validity().clone(),
            ))
        });

        Ok(res)
    }
}

impl OperationsVTable<DecimalBytePartsVTable> for DecimalBytePartsVTable {
    fn slice(array: &DecimalBytePartsArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        DecimalBytePartsArray::try_new(
            array.msp.slice(start, stop)?,
            array
                .lower_parts
                .iter()
                .map(|p| p.slice(start, stop))
                .try_collect()?,
            *array.decimal_dtype(),
        )
        .map(|d| d.to_array())
    }

    #[allow(clippy::useless_conversion)]
    fn scalar_at(array: &DecimalBytePartsArray, index: usize) -> VortexResult<Scalar> {
        // TODO(joe): support parts len != 1
        assert!(array.lower_parts.is_empty());
        let scalar = array.msp.scalar_at(index)?;

        // Note. values in msp, can only be signed integers upto size i64.
        let primitive_scalar = scalar.as_primitive();
        // TODO(joe): extend this to support multiple parts.
        let value = match_each_signed_integer_ptype!(primitive_scalar.ptype(), |P| {
            i64::from(
                primitive_scalar
                    .typed_value::<P>()
                    .vortex_expect("scalar must have correct ptype"),
            )
        });
        Ok(Scalar::new(
            array.dtype.clone(),
            DecimalValue::I64(value).into(),
        ))
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
            vec![],
            decimal_dtype,
        )
        .unwrap()
        .to_array();

        assert_eq!(Scalar::null(dtype.clone()), array.scalar_at(0).unwrap());
        assert_eq!(
            Scalar::new(dtype.clone(), DecimalValue::I64(200).into()),
            array.scalar_at(1).unwrap()
        );
        assert_eq!(
            Scalar::new(dtype, DecimalValue::I64(400).into()),
            array.scalar_at(2).unwrap()
        );
    }
}
