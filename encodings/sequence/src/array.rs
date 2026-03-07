// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use num_traits::cast::FromPrimitive;
use vortex_array::ArrayCommon;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::ExecutionCtx;
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
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
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

vtable!(Sequence);

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

#[derive(Clone, Debug)]
/// An array representing the equation `A[i] = base + i * multiplier`.
pub struct SequenceArray {
    common: ArrayCommon,
    base: PValue,
    multiplier: PValue,
}

/// Extension trait for [`SequenceArray`] methods.
pub trait SequenceArrayExt: Sized {
    /// Returns the primitive type of the array.
    fn ptype(&self) -> PType;

    /// Returns the base value.
    fn base(&self) -> PValue;

    /// Returns the multiplier value.
    fn multiplier(&self) -> PValue;

    /// Returns the validated final value of a sequence array.
    fn last(&self) -> PValue;

    /// Consumes the array and returns its parts.
    fn into_parts(self) -> SequenceArrayParts;
}

impl SequenceArrayExt for SequenceArray {
    fn ptype(&self) -> PType {
        self.common.dtype().as_ptype()
    }

    fn base(&self) -> PValue {
        self.base
    }

    fn multiplier(&self) -> PValue {
        self.multiplier
    }

    fn last(&self) -> PValue {
        Self::try_last(self.base, self.multiplier, self.ptype(), self.common.len())
            .vortex_expect("validated array")
    }

    fn into_parts(self) -> SequenceArrayParts {
        SequenceArrayParts {
            base: self.base,
            multiplier: self.multiplier,
            len: self.common.len(),
            ptype: self.common.dtype().as_ptype(),
            nullability: self.common.dtype().nullability(),
        }
    }
}

impl SequenceArray {
    /// Constructs a [`SequenceArray`] without validating that the `ptype` is an integer
    /// type or that the final element is representable.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptype` is an integer type (i.e., `ptype.is_int()` returns `true`).
    /// - `base + (length - 1) * multiplier` does not overflow the range of `ptype`.
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
            common: ArrayCommon::new_with_stats(length, dtype, ArrayStats::from(stats_set)),
            base,
            multiplier,
        }
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
        assert!(
            idx < self.common.len(),
            "index_value({idx}): index out of bounds"
        );

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
}

impl VTable for SequenceVTable {
    type Array = SequenceArray;

    type Metadata = SequenceMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(_array: &Self::Array) -> ArrayId {
        Self::ID
    }

    fn len(array: &SequenceArray) -> usize {
        array.common.len()
    }

    fn dtype(array: &SequenceArray) -> &DType {
        array.common.dtype()
    }

    fn stats(array: &SequenceArray) -> StatsSetRef<'_> {
        array.common.stats().to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(
        array: &SequenceArray,
        state: &mut H,
        _precision: Precision,
    ) {
        array.base.hash(state);
        array.multiplier.hash(state);
        array.common.dtype().hash(state);
        array.common.len().hash(state);
    }

    fn array_eq(array: &SequenceArray, other: &SequenceArray, _precision: Precision) -> bool {
        array.base == other.base
            && array.multiplier == other.multiplier
            && array.common.dtype() == other.common.dtype()
            && array.common.len() == other.common.len()
    }

    fn nbuffers(_array: &SequenceArray) -> usize {
        0
    }

    fn buffer(_array: &SequenceArray, idx: usize) -> BufferHandle {
        vortex_panic!("SequenceArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &SequenceArray, idx: usize) -> Option<String> {
        vortex_panic!("SequenceArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(_array: &SequenceArray) -> usize {
        0
    }

    fn child(_array: &SequenceArray, idx: usize) -> ArrayRef {
        vortex_panic!("SequenceArray child index {idx} out of bounds")
    }

    fn child_name(_array: &SequenceArray, idx: usize) -> String {
        vortex_panic!("SequenceArray child_name index {idx} out of bounds")
    }

    fn metadata(array: &SequenceArray) -> VortexResult<Self::Metadata> {
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
    ) -> VortexResult<SequenceArray> {
        Self::try_new(
            metadata.base,
            metadata.multiplier,
            dtype.as_ptype(),
            dtype.nullability(),
            len,
        )
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
        sequence_decompress(array)
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

#[derive(Debug)]
pub struct SequenceVTable;

impl SequenceVTable {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.sequence");

    /// Constructs a sequence array using a typed base and multiplier.
    pub fn try_new_typed<T: NativePType + Into<PValue>>(
        base: T,
        multiplier: T,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<SequenceArray> {
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
    ) -> VortexResult<SequenceArray> {
        if !ptype.is_int() {
            vortex_bail!("only integer ptype are supported in SequenceArray currently")
        }

        SequenceArray::try_last(base, multiplier, ptype, length).map_err(|e| {
            e.with_context(format!(
                "final value not expressible, base = {base:?}, multiplier = {multiplier:?}, len = {length} ",
            ))
        })?;

        // SAFETY: we just validated that `ptype` is an integer and that the final
        // element is representable via `try_last`.
        Ok(unsafe { SequenceArray::new_unchecked(base, multiplier, ptype, nullability, length) })
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

    use crate::array::SequenceVTable;

    #[test]
    fn test_sequence_canonical() {
        let arr = SequenceVTable::try_new_typed(2i64, 3, Nullability::NonNullable, 4).unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon);
    }

    #[test]
    fn test_sequence_slice_canonical() {
        let arr = SequenceVTable::try_new_typed(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .slice(2..3)
            .unwrap();

        let canon = PrimitiveArray::from_iter((2..3).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon);
    }

    #[test]
    fn test_sequence_scalar_at() {
        let scalar = SequenceVTable::try_new_typed(2i64, 3, Nullability::NonNullable, 4)
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
        assert!(SequenceVTable::try_new_typed(-127i8, -1i8, Nullability::NonNullable, 2).is_ok());
        assert!(SequenceVTable::try_new_typed(126i8, -1i8, Nullability::NonNullable, 2).is_ok());
    }

    #[test]
    fn test_sequence_too_big() {
        assert!(SequenceVTable::try_new_typed(127i8, 1i8, Nullability::NonNullable, 2).is_err());
        assert!(SequenceVTable::try_new_typed(-128i8, -1i8, Nullability::NonNullable, 2).is_err());
    }

    #[test]
    fn positive_multiplier_is_strict_sorted() -> VortexResult<()> {
        let arr = SequenceVTable::try_new_typed(0i64, 3, Nullability::NonNullable, 4)?;

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
        let arr = SequenceVTable::try_new_typed(5i64, 0, Nullability::NonNullable, 4)?;

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
        let arr = SequenceVTable::try_new_typed(10i64, -1, Nullability::NonNullable, 4)?;

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
        let arr = SequenceVTable::try_new_typed(0, large_multiplier, Nullability::NonNullable, 2)?;

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
