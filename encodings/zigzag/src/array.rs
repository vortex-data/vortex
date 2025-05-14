use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityChild,
    ValidityVTableFromChild,
};
use vortex_array::{
    Array, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray, ToCanonical, vtable,
};
use vortex_dtype::{DType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_scalar::{PrimitiveScalar, Scalar};
use zigzag::ZigZag as ExternalZigZag;

use crate::compute::ZigZagEncoded;
use crate::zigzag_decode;

vtable!(ZigZag);

impl VTable for ZigZagVTable {
    type Array = ZigZagArray;
    type Encoding = ZigZagEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.zigzag")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(ZigZagEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct ZigZagArray {
    dtype: DType,
    encoded: ArrayRef,
    stats_set: ArrayStats,
}

#[derive(Clone, Debug)]
pub struct ZigZagEncoding;

impl ZigZagArray {
    pub fn try_new(encoded: ArrayRef) -> VortexResult<Self> {
        let encoded_dtype = encoded.dtype().clone();
        if !encoded_dtype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }

        let dtype = DType::from(PType::try_from(&encoded_dtype)?.to_signed())
            .with_nullability(encoded_dtype.nullability());

        Ok(Self {
            dtype,
            encoded,
            stats_set: Default::default(),
        })
    }

    pub fn ptype(&self) -> PType {
        self.dtype().to_ptype()
    }

    pub fn encoded(&self) -> &ArrayRef {
        &self.encoded
    }
}

impl ArrayVTable<ZigZagVTable> for ZigZagVTable {
    fn len(array: &ZigZagArray) -> usize {
        array.encoded.len()
    }

    fn dtype(array: &ZigZagArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ZigZagArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<ZigZagVTable> for ZigZagVTable {
    fn canonicalize(array: &ZigZagArray) -> VortexResult<Canonical> {
        zigzag_decode(array.encoded().to_primitive()?).map(Canonical::Primitive)
    }
}

impl OperationsVTable<ZigZagVTable> for ZigZagVTable {
    fn slice(array: &ZigZagArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        Ok(ZigZagArray::try_new(array.encoded().slice(start, stop)?)?.into_array())
    }

    fn scalar_at(array: &ZigZagArray, index: usize) -> VortexResult<Scalar> {
        let scalar = array.encoded().scalar_at(index)?;
        if scalar.is_null() {
            return Ok(scalar.reinterpret_cast(array.ptype()));
        }

        let pscalar = PrimitiveScalar::try_from(&scalar)?;
        match_each_unsigned_integer_ptype!(pscalar.ptype(), |$P| {
            Ok(Scalar::primitive(
                <<$P as ZigZagEncoded>::Int>::decode(pscalar.typed_value::<$P>().ok_or_else(|| {
                    vortex_err!(
                        "Cannot decode provided scalar: expected {}, got ptype {}",
                        std::any::type_name::<$P>(),
                        pscalar.ptype()
                    )
                })?),
                array.dtype().nullability(),
            ))
        })
    }
}

impl ValidityChild<ZigZagVTable> for ZigZagVTable {
    fn validity_child(array: &ZigZagArray) -> &dyn Array {
        array.encoded()
    }
}

#[cfg(test)]
mod test {
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_scalar::Scalar;

    use super::*;

    #[test]
    fn test_compute_statistics() {
        let array = buffer![1i32, -5i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
        let canonical = array.to_canonical().unwrap();
        let zigzag = ZigZagEncoding.encode(&canonical, None).unwrap().unwrap();

        assert_eq!(
            zigzag.statistics().compute_max::<i32>(),
            array.statistics().compute_max::<i32>()
        );
        assert_eq!(
            zigzag.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            zigzag.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );

        let sliced = zigzag.slice(0, 2).unwrap();
        let sliced = sliced.as_::<ZigZagVTable>();
        assert_eq!(
            sliced.scalar_at(sliced.len() - 1).unwrap(),
            Scalar::from(-5i32)
        );

        assert_eq!(
            sliced.statistics().compute_min::<i32>(),
            array.statistics().compute_min::<i32>()
        );
        assert_eq!(
            sliced.statistics().compute_null_count(),
            array.statistics().compute_null_count()
        );
        assert_eq!(
            sliced.statistics().compute_is_constant(),
            array.statistics().compute_is_constant()
        );
    }
}
