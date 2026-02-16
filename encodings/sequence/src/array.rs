// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use num_traits::cast::FromPrimitive;
use vortex_array::ArrayBufferVisitor;
use vortex_array::ArrayChildVisitor;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::expr::stats::Precision as StatPrecision;
use vortex_array::expr::stats::Stat;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSet;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::BaseArrayVTable;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::VisitorVTable;
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
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_scalar::PValue;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;
use vortex_session::VortexSession;

use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

vtable!(Sequence);

#[derive(Clone, prost::Message)]
pub struct SequenceMetadata {
    #[prost(message, tag = "1")]
    base: Option<vortex_proto::scalar::ScalarValue>,
    #[prost(message, tag = "2")]
    multiplier: Option<vortex_proto::scalar::ScalarValue>,
}

/// Components of [`SequenceArray`].
pub struct SequenceArrayParts {
    pub base: PValue,
    pub multiplier: PValue,
    pub len: usize,
    pub ptype: PType,
    pub nullability: Nullability,
}

#[derive(Clone, Debug)]
/// An array representing the equation `A[i] = base + i * multiplier`.
pub struct SequenceArray {
    base: PValue,
    multiplier: PValue,
    dtype: DType,
    pub(crate) len: usize,
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

        // A sequence A[i] = base + i * multiplier is sorted iff multiplier >= 0,
        // and strictly sorted iff multiplier > 0.
        let m_int = multiplier.cast::<i64>();
        let is_sorted = m_int >= 0;
        let is_strict_sorted = m_int > 0;

        // SAFETY: we don't have duplicate stats
        let stats_set = unsafe {
            StatsSet::new_unchecked(vec![
                (Stat::IsSorted, StatPrecision::Exact(is_sorted.into())),
                (
                    Stat::IsStrictSorted,
                    StatPrecision::Exact(is_strict_sorted.into()),
                ),
            ])
        };

        Self {
            base,
            multiplier,
            dtype,
            len: length,
            stats_set: ArrayStats::from(stats_set),
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
        assert!(idx < self.len, "index_value({idx}): index out of bounds");

        match_each_native_ptype!(self.ptype(), |P| {
            let base = self.base.cast::<P>();
            let multiplier = self.multiplier.cast::<P>();
            let value = base + (multiplier * <P>::from_usize(idx).vortex_expect("must fit"));

            PValue::from(value)
        })
    }

    /// Returns the validated final value of a sequence array
    pub fn last(&self) -> PValue {
        Self::try_last(self.base, self.multiplier, self.ptype(), self.len)
            .vortex_expect("validated array")
    }

    pub fn into_parts(self) -> SequenceArrayParts {
        SequenceArrayParts {
            base: self.base,
            multiplier: self.multiplier,
            len: self.len,
            ptype: self.dtype.as_ptype(),
            nullability: self.dtype.nullability(),
        }
    }
}

impl VTable for SequenceVTable {
    type Array = SequenceArray;

    type Metadata = ProstMetadata<SequenceMetadata>;

    type ArrayVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = Self;
    type VisitorVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
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

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        Ok(ProstMetadata(
            <ProstMetadata<SequenceMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<SequenceArray> {
        let ptype = dtype.as_ptype();

        // We go via scalar to cast the scalar values into the correct PType
        let base = Scalar::from_proto_value(
            metadata
                .0
                .base
                .as_ref()
                .ok_or_else(|| vortex_err!("base required"))?,
            &DType::Primitive(ptype, NonNullable),
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("non-nullable primitive");

        let multiplier = Scalar::from_proto_value(
            metadata
                .0
                .multiplier
                .as_ref()
                .ok_or_else(|| vortex_err!("multiplier required"))?,
            &DType::Primitive(ptype, NonNullable),
        )?
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

    fn with_children(_array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.is_empty(),
            "SequenceArray expects 0 children, got {}",
            children.len()
        );
        Ok(())
    }

    fn execute(array: &Self::Array, _ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let prim = match_each_native_ptype!(array.ptype(), |P| {
            let base = array.base().cast::<P>();
            let multiplier = array.multiplier().cast::<P>();
            let values = BufferMut::from_iter(
                (0..array.len())
                    .map(|i| base + <P>::from_usize(i).vortex_expect("must fit") * multiplier),
            );
            PrimitiveArray::new(values, array.dtype.nullability().into())
        });

        Ok(prim.into_array())
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: &SequenceArray,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl BaseArrayVTable<SequenceVTable> for SequenceVTable {
    fn len(array: &SequenceArray) -> usize {
        array.len
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
        array.len.hash(state);
    }

    fn array_eq(array: &SequenceArray, other: &SequenceArray, _precision: Precision) -> bool {
        array.base == other.base
            && array.multiplier == other.multiplier
            && array.dtype == other.dtype
            && array.len == other.len
    }
}

impl OperationsVTable<SequenceVTable> for SequenceVTable {
    fn scalar_at(array: &SequenceArray, index: usize) -> VortexResult<Scalar> {
        Scalar::try_new(
            array.dtype().clone(),
            Some(ScalarValue::Primitive(array.index_value(index))),
        )
    }
}

impl ValidityVTable<SequenceVTable> for SequenceVTable {
    fn validity(_array: &SequenceArray) -> VortexResult<Validity> {
        Ok(Validity::AllValid)
    }
}

impl VisitorVTable<SequenceVTable> for SequenceVTable {
    fn visit_buffers(_array: &SequenceArray, _visitor: &mut dyn ArrayBufferVisitor) {
        // TODO(joe): expose scalar values
    }

    fn nbuffers(_array: &SequenceArray) -> usize {
        0
    }

    fn visit_children(_array: &SequenceArray, _visitor: &mut dyn ArrayChildVisitor) {}

    fn nchildren(_array: &SequenceArray) -> usize {
        0
    }
}

#[derive(Debug)]
pub struct SequenceVTable;

impl SequenceVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.sequence");
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::stats::Precision as StatPrecision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_scalar::Scalar;
    use vortex_scalar::ScalarValue;

    use crate::array::SequenceArray;

    #[test]
    fn test_sequence_canonical() {
        let arr = SequenceArray::typed_new(2i64, 3, Nullability::NonNullable, 4).unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon);
    }

    #[test]
    fn test_sequence_slice_canonical() {
        let arr = SequenceArray::typed_new(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .slice(2..3)
            .unwrap();

        let canon = PrimitiveArray::from_iter((2..3).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon);
    }

    #[test]
    fn test_sequence_scalar_at() {
        let scalar = SequenceArray::typed_new(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .scalar_at(2)
            .unwrap();

        assert_eq!(
            scalar,
            Scalar::try_new(scalar.dtype().clone(), Some(ScalarValue::from(8i64))).unwrap()
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

    #[test]
    fn positive_multiplier_is_strict_sorted() -> VortexResult<()> {
        let arr = SequenceArray::typed_new(0i64, 3, Nullability::NonNullable, 4)?;

        let is_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Some(StatPrecision::Exact(true)));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));
        assert_eq!(is_strict_sorted, Some(StatPrecision::Exact(true)));
        Ok(())
    }

    #[test]
    fn zero_multiplier_is_sorted_not_strict() -> VortexResult<()> {
        let arr = SequenceArray::typed_new(5i64, 0, Nullability::NonNullable, 4)?;

        let is_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Some(StatPrecision::Exact(true)));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));
        assert_eq!(is_strict_sorted, Some(StatPrecision::Exact(false)));
        Ok(())
    }

    #[test]
    fn negative_multiplier_not_sorted() -> VortexResult<()> {
        let arr = SequenceArray::typed_new(10i64, -1, Nullability::NonNullable, 4)?;

        let is_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Some(StatPrecision::Exact(false)));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));
        assert_eq!(is_strict_sorted, Some(StatPrecision::Exact(false)));
        Ok(())
    }
}
