// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use num_traits::cast::FromPrimitive;
use vortex_array::Array;
use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Precision as StatPrecision;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_pvalue;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::compress::sequence_decompress;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

vtable!(Sequence, Sequence, SequenceData);

#[derive(Debug, Clone, Copy)]
pub struct SequenceMetadata {
    base: PValue,
    multiplier: PValue,
}

#[derive(Clone, prost::Message)]
pub struct ProstSequenceMetadata {
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

pub(super) const SLOT_NAMES: [&str; 0] = [];

#[derive(Clone, Debug)]
/// An array representing the equation `A[i] = base + i * multiplier`.
pub struct SequenceData {
    base: PValue,
    multiplier: PValue,
    dtype: DType,
    pub(crate) len: usize,
    pub(super) slots: Vec<Option<ArrayRef>>,
    stats_set: ArrayStats,
}

impl SequenceData {
    pub fn try_new_typed<T: NativePType + Into<PValue>>(
        base: T,
        multiplier: T,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<Self> {
        Self::try_new(
            base.into(),
            multiplier.into(),
            T::PTYPE,
            nullability,
            length,
        )
    }

    /// Constructs a sequence array using two integer values (with the same ptype).
    pub fn try_new(
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

        // SAFETY: we just validated that `ptype` is an integer and that the final
        // element is representable via `try_last`.
        Ok(unsafe { Self::new_unchecked(base, multiplier, ptype, nullability, length) })
    }

    /// Constructs a [`SequenceArray`] without validating that the `ptype` is an integer
    /// type or that the final element is representable.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptype` is an integer type (i.e., `ptype.is_int()` returns `true`).
    /// - `base + (length - 1) * multiplier` does not overflow the range of `ptype`.
    ///
    /// Violating the first invariant will cause a panic. Violating the second will
    /// cause silent wraparound when materializing elements, producing incorrect values.
    pub(crate) unsafe fn new_unchecked(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> Self {
        let dtype = DType::Primitive(ptype, nullability);

        // A sequence A[i] = base + i * multiplier is sorted iff multiplier >= 0,
        // and strictly sorted iff multiplier > 0.

        let (is_sorted, is_strict_sorted) = match_each_pvalue!(
            multiplier,
            uint: |v| { (true, v> 0) },
            int: |v| { (v >= 0, v > 0) },
            float: |_v| { unreachable!("float multiplier not supported") }
        );

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
            slots: vec![],
            stats_set: ArrayStats::from(stats_set),
        }
    }

    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the logical data type of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
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

            let base = base.cast::<P>()?;
            let multiplier = multiplier.cast::<P>()?;
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
            let base = self.base.cast::<P>().vortex_expect("must be able to cast");
            let multiplier = self
                .multiplier
                .cast::<P>()
                .vortex_expect("must be able to cast");
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

impl VTable for Sequence {
    type ArrayData = SequenceData;

    type Metadata = SequenceMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::ArrayData) -> &Self {
        &Sequence
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &SequenceData) -> usize {
        array.len
    }

    fn dtype(array: &SequenceData) -> &DType {
        &array.dtype
    }

    fn stats(array: &SequenceData) -> &ArrayStats {
        &array.stats_set
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &SequenceData,
        state: &mut H,
        _precision: Precision,
    ) {
        array.base.hash(state);
        array.multiplier.hash(state);
    }

    fn array_eq(array: &SequenceData, other: &SequenceData, _precision: Precision) -> bool {
        array.base == other.base && array.multiplier == other.multiplier
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("SequenceArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("SequenceArray buffer_name index {idx} out of bounds")
    }

    fn metadata(array: ArrayView<'_, Self>) -> VortexResult<Self::Metadata> {
        Ok(SequenceMetadata {
            base: array.base(),
            multiplier: array.multiplier(),
        })
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        let prost = ProstMetadata(ProstSequenceMetadata {
            base: Some((&metadata.base).into()),
            multiplier: Some((&metadata.multiplier).into()),
        });

        Ok(Some(prost.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let prost =
            <ProstMetadata<ProstSequenceMetadata> as DeserializeMetadata>::deserialize(bytes)?;

        let ptype = dtype.as_ptype();

        // We go via Scalar to validate that the value is valid for the ptype.
        let base = Scalar::from_proto_value(
            prost
                .base
                .as_ref()
                .ok_or_else(|| vortex_err!("base required"))?,
            &DType::Primitive(ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array base should be a non-nullable primitive");

        let multiplier = Scalar::from_proto_value(
            prost
                .multiplier
                .as_ref()
                .ok_or_else(|| vortex_err!("multiplier required"))?,
            &DType::Primitive(ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array multiplier should be a non-nullable primitive");

        Ok(SequenceMetadata { base, multiplier })
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
    ) -> VortexResult<SequenceData> {
        SequenceData::try_new(
            metadata.base,
            metadata.multiplier,
            dtype.as_ptype(),
            dtype.nullability(),
            len,
        )
    }

    fn slots(array: ArrayView<'_, Self>) -> &[Option<ArrayRef>] {
        &array.data().slots
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn with_slots(array: &mut Self::ArrayData, slots: Vec<Option<ArrayRef>>) -> VortexResult<()> {
        vortex_ensure!(
            slots.is_empty(),
            "SequenceArray expects 0 slots, got {}",
            slots.len()
        );
        array.slots = slots;
        Ok(())
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        sequence_decompress(&array).map(ExecutionResult::done)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }
}

impl OperationsVTable<Sequence> for Sequence {
    fn scalar_at(
        array: ArrayView<'_, Sequence>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Scalar::try_new(
            array.dtype().clone(),
            Some(ScalarValue::Primitive(array.index_value(index))),
        )
    }
}

impl ValidityVTable<Sequence> for Sequence {
    fn validity(_array: ArrayView<'_, Sequence>) -> VortexResult<Validity> {
        Ok(Validity::AllValid)
    }
}

#[derive(Clone, Debug)]
pub struct Sequence;

impl Sequence {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.sequence");

    /// Construct a new [`SequenceArray`] from its components.
    pub fn try_new(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<SequenceArray> {
        Array::try_from_data(SequenceData::try_new(
            base,
            multiplier,
            ptype,
            nullability,
            length,
        )?)
    }

    /// Construct a new typed [`SequenceArray`] from base/multiplier values.
    pub fn try_new_typed<T: NativePType + Into<PValue>>(
        base: T,
        multiplier: T,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<SequenceArray> {
        Array::try_from_data(SequenceData::try_new_typed(
            base,
            multiplier,
            nullability,
            length,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::stats::Precision as StatPrecision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_error::VortexResult;

    use crate::Sequence;

    #[test]
    fn test_sequence_canonical() {
        let arr = Sequence::try_new_typed(2i64, 3, Nullability::NonNullable, 4).unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon);
    }

    #[test]
    fn test_sequence_slice_canonical() {
        let arr = Sequence::try_new_typed(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .slice(2..3)
            .unwrap();

        let canon = PrimitiveArray::from_iter((2..3).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon);
    }

    #[test]
    fn test_sequence_scalar_at() {
        let scalar = Sequence::try_new_typed(2i64, 3, Nullability::NonNullable, 4)
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
        assert!(Sequence::try_new_typed(-127i8, -1i8, Nullability::NonNullable, 2).is_ok());
        assert!(Sequence::try_new_typed(126i8, -1i8, Nullability::NonNullable, 2).is_ok());
    }

    #[test]
    fn test_sequence_too_big() {
        assert!(Sequence::try_new_typed(127i8, 1i8, Nullability::NonNullable, 2).is_err());
        assert!(Sequence::try_new_typed(-128i8, -1i8, Nullability::NonNullable, 2).is_err());
    }

    #[test]
    fn positive_multiplier_is_strict_sorted() -> VortexResult<()> {
        let arr = Sequence::try_new_typed(0i64, 3, Nullability::NonNullable, 4)?;

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
        let arr = Sequence::try_new_typed(5i64, 0, Nullability::NonNullable, 4)?;

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
        let arr = Sequence::try_new_typed(10i64, -1, Nullability::NonNullable, 4)?;

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

    // This is regression test for an issue caught by the fuzzer, where SequenceArrays with
    // multiplier > i64::MAX were unable to be constructed.
    #[test]
    fn test_large_multiplier_sorted() -> VortexResult<()> {
        let large_multiplier = (i64::MAX as u64) + 1;
        let arr = Sequence::try_new_typed(0, large_multiplier, Nullability::NonNullable, 2)?;

        let is_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));

        assert_eq!(is_sorted, Some(StatPrecision::Exact(true)));
        assert_eq!(is_strict_sorted, Some(StatPrecision::Exact(true)));

        Ok(())
    }
}
