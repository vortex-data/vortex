// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;
use std::ops::Range;

use num_traits::One;
use num_traits::cast::FromPrimitive;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::execution::ExecutionCtx;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::ArrayVTable;
use vortex_array::vtable::ArrayVTableExt;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::vtable::EncodeVTable;
use vortex_array::vtable::NotSupported;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::VisitorVTable;
use vortex_buffer::BufferHandle;
use vortex_buffer::BufferMut;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::Nullability::NonNullable;
use vortex_dtype::PType;
use vortex_dtype::match_each_integer_ptype;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_scalar::PValue;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;
use vortex_vector::Vector;
use vortex_vector::primitive::PVector;

vtable!(Sequence);

#[derive(Clone, prost::Message)]
pub struct SequenceMetadata {
    #[prost(message, tag = "1")]
    base: Option<vortex_proto::scalar::ScalarValue>,
    #[prost(message, tag = "2")]
    multiplier: Option<vortex_proto::scalar::ScalarValue>,
}

#[derive(Clone, Debug)]
/// An array representing the equation `A[i] = base + i * multiplier`.
pub struct SequenceArray {
    base: PValue,
    multiplier: PValue,
    dtype: DType,
    pub(crate) length: usize,
    stats_set: ArrayStats,
}

impl SequenceArray {
    pub fn typed_new<T: NativePType + Into<PValue>>(
        base: T,
        multiplier: T,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<Self> {
        Self::new(
            base.into(),
            multiplier.into(),
            T::PTYPE,
            nullability,
            length,
        )
    }

    /// Constructs a sequence array using two integer values (with the same ptype).
    pub fn new(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<Self> {
        if !ptype.is_int() {
            vortex_bail!("only integer ptype are supported in SequenceArray currently")
        }

        Self::try_last(base, multiplier, ptype, length).map_err(|e| {
            e.with_context(format!(
                "final value not expressible, base = {base:?}, multiplier = {multiplier:?}, len = {length} ",
            ))
        })?;

        Ok(Self::unchecked_new(
            base,
            multiplier,
            ptype,
            nullability,
            length,
        ))
    }

    pub(crate) fn unchecked_new(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> Self {
        let dtype = DType::Primitive(ptype, nullability);
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

    pub fn base(&self) -> PValue {
        self.base
    }

    pub fn multiplier(&self) -> PValue {
        self.multiplier
    }

    pub(crate) fn try_last(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        length: usize,
    ) -> VortexResult<PValue> {
        match_each_integer_ptype!(ptype, |P| {
            let len_t = <P>::from_usize(length - 1)
                .ok_or_else(|| vortex_err!("cannot convert length {} into {}", length, ptype))?;

            let base = base.cast::<P>();
            let multiplier = multiplier.cast::<P>();

            let last = len_t
                .checked_mul(multiplier)
                .and_then(|offset| offset.checked_add(base))
                .ok_or_else(|| vortex_err!("last value computation overflows"))?;
            Ok(PValue::from(last))
        })
    }

    pub(crate) fn index_value(&self, idx: usize) -> PValue {
        assert!(idx < self.length, "index_value({idx}): index out of bounds");

        match_each_native_ptype!(self.ptype(), |P| {
            let base = self.base.cast::<P>();
            let multiplier = self.multiplier.cast::<P>();
            let value = base + (multiplier * <P>::from_usize(idx).vortex_expect("must fit"));

            PValue::from(value)
        })
    }

    /// Returns the validated final value of a sequence array
    pub fn last(&self) -> PValue {
        Self::try_last(self.base, self.multiplier, self.ptype(), self.length)
            .vortex_expect("validated array")
    }
}

impl VTable for SequenceVTable {
    type Array = SequenceArray;

    type Metadata = ProstMetadata<SequenceMetadata>;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = Self;
    type OperatorVTable = Self;

    fn id(&self) -> ArrayId {
        ArrayId::new_ref("vortex.sequence")
    }

    fn encoding(_array: &Self::Array) -> ArrayVTable {
        SequenceVTable.as_vtable()
    }

    fn metadata(array: &SequenceArray) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(SequenceMetadata {
            base: Some((&array.base()).into()),
            multiplier: Some((&array.multiplier()).into()),
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(buffer: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<SequenceMetadata> as DeserializeMetadata>::deserialize(buffer)?,
        ))
    }

    fn build(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<SequenceArray> {
        let ptype = dtype.as_ptype();

        // We go via scalar to cast the scalar values into the correct PType
        let base = Scalar::new(
            DType::Primitive(ptype, NonNullable),
            metadata
                .0
                .base
                .as_ref()
                .ok_or_else(|| vortex_err!("base required"))?
                .try_into()?,
        )
        .as_primitive()
        .pvalue()
        .vortex_expect("non-nullable primitive");

        let multiplier = Scalar::new(
            DType::Primitive(ptype, NonNullable),
            metadata
                .0
                .multiplier
                .as_ref()
                .ok_or_else(|| vortex_err!("base required"))?
                .try_into()?,
        )
        .as_primitive()
        .pvalue()
        .vortex_expect("non-nullable primitive");

        Ok(SequenceArray::unchecked_new(
            base,
            multiplier,
            ptype,
            dtype.nullability(),
            len,
        ))
    }

    fn execute(array: &Self::Array, _ctx: &mut dyn ExecutionCtx) -> VortexResult<Vector> {
        Ok(match_each_native_ptype!(array.ptype(), |P| {
            let base = array.base().cast::<P>();
            let multiplier = array.multiplier().cast::<P>();

            let values = if multiplier == <P>::one() {
                BufferMut::from_iter(
                    (0..array.len()).map(|i| base + <P>::from_usize(i).vortex_expect("must fit")),
                )
            } else {
                BufferMut::from_iter(
                    (0..array.len())
                        .map(|i| base + <P>::from_usize(i).vortex_expect("must fit") * multiplier),
                )
            };

            PVector::<P>::new(values.freeze(), Mask::new_true(array.len())).into()
        }))
    }
}

impl BaseArrayVTable<SequenceVTable> for SequenceVTable {
    fn len(array: &SequenceArray) -> usize {
        array.length
    }

    fn dtype(array: &SequenceArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &SequenceArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &SequenceArray,
        state: &mut H,
        _precision: Precision,
    ) {
        array.base.hash(state);
        array.multiplier.hash(state);
        array.dtype.hash(state);
        array.length.hash(state);
    }

    fn array_eq(array: &SequenceArray, other: &SequenceArray, _precision: Precision) -> bool {
        array.base == other.base
            && array.multiplier == other.multiplier
            && array.dtype == other.dtype
            && array.length == other.length
    }
}

impl CanonicalVTable<SequenceVTable> for SequenceVTable {
    fn canonicalize(array: &SequenceArray) -> Canonical {
        let prim = match_each_native_ptype!(array.ptype(), |P| {
            let base = array.base().cast::<P>();
            let multiplier = array.multiplier().cast::<P>();
            let values = BufferMut::from_iter(
                (0..array.len())
                    .map(|i| base + <P>::from_usize(i).vortex_expect("must fit") * multiplier),
            );
            PrimitiveArray::new(values, array.dtype.nullability().into())
        });

        Canonical::Primitive(prim)
    }
}

impl OperationsVTable<SequenceVTable> for SequenceVTable {
    fn slice(array: &SequenceArray, range: Range<usize>) -> ArrayRef {
        SequenceArray::unchecked_new(
            array.index_value(range.start),
            array.multiplier,
            array.ptype(),
            array.dtype().nullability(),
            range.len(),
        )
        .to_array()
    }

    fn scalar_at(array: &SequenceArray, index: usize) -> Scalar {
        Scalar::new(
            array.dtype().clone(),
            ScalarValue::from(array.index_value(index)),
        )
    }
}

impl ValidityVTable<SequenceVTable> for SequenceVTable {
    fn is_valid(_array: &SequenceArray, _index: usize) -> bool {
        true
    }

    fn all_valid(_array: &SequenceArray) -> bool {
        true
    }

    fn all_invalid(_array: &SequenceArray) -> bool {
        false
    }

    fn validity_mask(array: &SequenceArray) -> Mask {
        Mask::AllTrue(array.len())
    }
}

impl VisitorVTable<SequenceVTable> for SequenceVTable {
    fn visit_buffers(_array: &SequenceArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // TODO(joe): expose scalar values
    }

    fn visit_children(_array: &SequenceArray, _visitor: &mut dyn ArrayChildVisitor) {}
}

#[derive(Debug)]
pub struct SequenceVTable;

impl EncodeVTable<SequenceVTable> for SequenceVTable {
    fn encode(
        _vtable: &SequenceVTable,
        _canonical: &Canonical,
        _like: Option<&SequenceArray>,
    ) -> VortexResult<Option<SequenceArray>> {
        // TODO(joe): hook up compressor
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;
    use vortex_scalar::ScalarValue;

    use crate::array::SequenceArray;

    #[test]
    fn test_sequence_canonical() {
        let arr = SequenceArray::typed_new(2i64, 3, Nullability::NonNullable, 4).unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i64 + i * 3));

        assert_eq!(
            arr.to_primitive().as_slice::<i64>(),
            canon.as_slice::<i64>()
        )
    }

    #[test]
    fn test_sequence_slice_canonical() {
        let arr = SequenceArray::typed_new(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .slice(2..3);

        let canon = PrimitiveArray::from_iter((2..3).map(|i| 2i64 + i * 3));

        assert_eq!(
            arr.to_primitive().as_slice::<i64>(),
            canon.as_slice::<i64>()
        )
    }

    #[test]
    fn test_sequence_scalar_at() {
        let scalar = SequenceArray::typed_new(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .scalar_at(2);

        assert_eq!(
            scalar,
            Scalar::new(scalar.dtype().clone(), ScalarValue::from(8i64))
        )
    }

    #[test]
    fn test_sequence_min_max() {
        assert!(SequenceArray::typed_new(-127i8, -1i8, Nullability::NonNullable, 2).is_ok());
        assert!(SequenceArray::typed_new(126i8, -1i8, Nullability::NonNullable, 2).is_ok());
    }

    #[test]
    fn test_sequence_too_big() {
        assert!(SequenceArray::typed_new(127i8, 1i8, Nullability::NonNullable, 2).is_err());
        assert!(SequenceArray::typed_new(-128i8, -1i8, Nullability::NonNullable, 2).is_err());
    }
}
