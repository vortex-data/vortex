use num_traits::cast::FromPrimitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::stats::{ArrayStats, StatsSetRef};
use vortex_array::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityVTable,
    VisitorVTable,
};
use vortex_array::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EncodingId, EncodingRef, vtable,
};
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::{DType, NativePType, PType, match_each_integer_ptype, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_mask::Mask;
use vortex_scalar::{Scalar, ScalarValue};

vtable!(Sequence);

#[derive(Clone, Debug)]
/// An array representing the equation `A[i] = base + i * multiplier`.
pub struct SequenceArray {
    base: ScalarValue,
    multiplier: ScalarValue,
    dtype: DType,
    length: usize,
    stats_set: ArrayStats,
}

impl SequenceArray {
    pub fn typed_new<T: NativePType + Into<ScalarValue>>(
        base: T,
        multiplier: T,
        length: usize,
    ) -> VortexResult<Self> {
        Self::new(base.into(), multiplier.into(), T::PTYPE, length)
    }

    // Constructs a sequence array using two integer values (with the same ptype).
    pub fn new(
        base: ScalarValue,
        multiplier: ScalarValue,
        ptype: PType,
        length: usize,
    ) -> VortexResult<Self> {
        if !ptype.is_int() {
            vortex_bail!("only integer ptype are supported in SequenceArray currently")
        }

        match_each_integer_ptype!(ptype, |$P| {
            let len_t = <$P>::from_usize(length)
                .ok_or_else(|| vortex_err!("cannot convert length {} into {}", length, ptype))?;

            let base = <$P>::try_from(&base)?;
            let multiplier = <$P>::try_from(&multiplier)?;

            if len_t
                .checked_mul(multiplier)
                .and_then(|offset| offset.checked_add(base))
                .is_none()
            {
                vortex_bail!("array value out of range of array type")
            }
        });

        Ok(Self::unchecked_new(base, multiplier, ptype, length))
    }

    pub(crate) fn unchecked_new(
        base: ScalarValue,
        multiplier: ScalarValue,
        ptype: PType,
        length: usize,
    ) -> Self {
        let dtype = DType::Primitive(ptype, NonNullable);
        Self {
            base,
            multiplier,
            dtype,
            length,
            // TODO(joe): add stats, on construct or on use?
            stats_set: Default::default(),
        }
    }

    pub fn ptype(&self) -> PType {
        self.dtype.as_ptype()
    }

    pub fn base(&self) -> &ScalarValue {
        &self.base
    }

    pub fn multiplier(&self) -> &ScalarValue {
        &self.multiplier
    }
}

impl VTable for SequenceVTable {
    type Array = SequenceArray;
    type Encoding = SequenceEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type SerdeVTable = Self;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.sequence")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(SequenceEncoding.as_ref())
    }
}

impl ArrayVTable<SequenceVTable> for SequenceVTable {
    fn len(array: &SequenceArray) -> usize {
        array.length
    }

    fn dtype(array: &SequenceArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &SequenceArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }
}

impl CanonicalVTable<SequenceVTable> for SequenceVTable {
    fn canonicalize(array: &SequenceArray) -> VortexResult<Canonical> {
        let prim = match_each_native_ptype!(array.ptype(), |$P| {
            let base = <$P>::try_from(&array.base)?;
            let multi = <$P>::try_from(&array.multiplier)?;
            PrimitiveArray::from_iter((0..array.len()).map(|i| base + <$P>::from_usize(i).vortex_expect("") * multi))
        });

        Ok(Canonical::Primitive(prim))
    }
}

impl OperationsVTable<SequenceVTable> for SequenceVTable {
    fn slice(array: &SequenceArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let sliced_len = stop - start;
        let ptype = array.ptype();
        let arr = match_each_native_ptype!(array.ptype(), |$P| {
            let base = <$P>::try_from(&array.base)?;
            let multi = <$P>::try_from(&array.multiplier)?;
            let new_base = base + (multi * <$P>::from_usize(start).vortex_expect("must fit"));

            SequenceArray::unchecked_new(new_base.into(), array.multiplier.clone(), ptype, sliced_len)
        });

        Ok(arr.to_array())
    }

    fn scalar_at(array: &SequenceArray, index: usize) -> VortexResult<Scalar> {
        let scalar_value = match_each_native_ptype!(array.ptype(), |$P| {
            let base = <$P>::try_from(&array.base)?;
            let multi = <$P>::try_from(&array.multiplier)?;
            let scalar = base + (multi * <$P>::from_usize(index).vortex_expect("must fit"));
            ScalarValue::from(scalar)
        });

        Ok(Scalar::new(array.dtype().clone(), scalar_value))
    }
}

impl ValidityVTable<SequenceVTable> for SequenceVTable {
    fn is_valid(_array: &SequenceArray, _index: usize) -> VortexResult<bool> {
        Ok(true)
    }

    fn all_valid(_array: &SequenceArray) -> VortexResult<bool> {
        Ok(true)
    }

    fn all_invalid(_array: &SequenceArray) -> VortexResult<bool> {
        Ok(false)
    }

    fn validity_mask(array: &SequenceArray) -> VortexResult<Mask> {
        Ok(Mask::AllTrue(array.len()))
    }
}

impl VisitorVTable<SequenceVTable> for SequenceVTable {
    fn visit_buffers(_array: &SequenceArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // TODO(joe): expose scalar values
    }

    fn visit_children(_array: &SequenceArray, _visitor: &mut dyn ArrayChildVisitor) {}
}

#[derive(Clone, Debug)]
pub struct SequenceEncoding;

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_scalar::{Scalar, ScalarValue};

    use crate::array::SequenceArray;

    #[test]
    fn test_sequence_canonical() {
        let arr = SequenceArray::typed_new(2i64, 3, 4).unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i64 + i * 3));

        assert_eq!(
            arr.to_canonical()
                .unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<i64>(),
            canon.as_slice::<i64>()
        )
    }

    #[test]
    fn test_sequence_slice_canonical() {
        let arr = SequenceArray::typed_new(2i64, 3, 4)
            .unwrap()
            .slice(2, 3)
            .unwrap();

        let canon = PrimitiveArray::from_iter((2..3).map(|i| 2i64 + i * 3));

        assert_eq!(
            arr.to_canonical()
                .unwrap()
                .into_primitive()
                .unwrap()
                .as_slice::<i64>(),
            canon.as_slice::<i64>()
        )
    }

    #[test]
    fn test_sequence_scalar_at() {
        let scalar = SequenceArray::typed_new(2i64, 3, 4)
            .unwrap()
            .scalar_at(2)
            .unwrap();

        assert_eq!(
            scalar,
            Scalar::new(scalar.dtype().clone(), ScalarValue::from(8i64))
        )
    }

    #[test]
    fn test_sequence_too_big() {
        assert!(SequenceArray::typed_new(127i8, 1i8, 2).is_err());
        assert!(SequenceArray::typed_new(-127i8, -1i8, 2).is_err());
    }
}
