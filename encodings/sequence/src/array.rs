// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use num_traits::cast::FromPrimitive;
use prost::Message;
use smallvec::smallvec;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
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
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
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
use vortex_session::registry::CachedId;

use crate::compress::sequence_decompress;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

/// A [`Sequence`]-encoded Vortex array.
pub type SequenceArray = Array<Sequence>;

#[derive(Clone, prost::Message)]
pub struct SequenceMetadata {
    #[prost(message, tag = "1")]
    base: Option<vortex_proto::scalar::ScalarValue>,
    #[prost(message, tag = "2")]
    multiplier: Option<vortex_proto::scalar::ScalarValue>,
}

pub(super) const SLOT_NAMES: [&str; 0] = [];

#[derive(Clone, Debug)]
/// An array representing the equation `A[i] = base + i * multiplier`.
pub struct SequenceData {
    base: PValue,
    multiplier: PValue,
}

impl Display for SequenceData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "base: {}, multiplier: {}", self.base, self.multiplier)
    }
}

pub struct SequenceDataParts {
    pub base: PValue,
    pub multiplier: PValue,
    pub ptype: PType,
}

impl SequenceData {
    pub(crate) fn try_new_typed<T: NativePType + Into<PValue>>(
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
    pub(crate) fn try_new(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<Self> {
        let dtype = DType::Primitive(ptype, nullability);
        Self::validate(base, multiplier, &dtype, length)?;
        let (base, multiplier) = Self::normalize(base, multiplier, ptype)?;

        Ok(unsafe { Self::new_unchecked(base, multiplier) })
    }

    pub fn validate(
        base: PValue,
        multiplier: PValue,
        dtype: &DType,
        length: usize,
    ) -> VortexResult<()> {
        let DType::Primitive(ptype, _) = dtype else {
            vortex_bail!("only primitive dtypes are supported in SequenceArray currently");
        };

        if !ptype.is_int() {
            vortex_bail!("only integer ptype are supported in SequenceArray currently")
        }

        vortex_ensure!(length > 0, "SequenceArray length must be greater than zero");
        Self::try_last(base, multiplier, *ptype, length).map_err(|e| {
            e.with_context(format!(
                "final value not expressible, base = {base:?}, multiplier = {multiplier:?}, len = {length} ",
            ))
        })?;

        Ok(())
    }

    fn normalize(base: PValue, multiplier: PValue, ptype: PType) -> VortexResult<(PValue, PValue)> {
        match_each_integer_ptype!(ptype, |P| {
            Ok((
                PValue::from(base.cast::<P>()?),
                PValue::from(multiplier.cast::<P>()?),
            ))
        })
    }

    /// Constructs a [`SequenceArray`] payload without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `base` and `multiplier` are both normalized to the same integer `ptype`.
    /// - they are logically compatible with the outer dtype and len.
    pub(crate) unsafe fn new_unchecked(base: PValue, multiplier: PValue) -> Self {
        Self { base, multiplier }
    }

    pub fn ptype(&self) -> PType {
        self.base.ptype()
    }

    pub fn base(&self) -> PValue {
        self.base
    }

    pub fn multiplier(&self) -> PValue {
        self.multiplier
    }

    pub fn into_parts(self) -> SequenceDataParts {
        SequenceDataParts {
            base: self.base,
            multiplier: self.multiplier,
            ptype: self.base.ptype(),
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

impl ArrayHash for SequenceData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.base.hash(state);
        self.multiplier.hash(state);
    }
}

impl ArrayEq for SequenceData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.base == other.base && self.multiplier == other.multiplier
    }
}

impl VTable for Sequence {
    type TypedArrayData = SequenceData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.sequence");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        SequenceData::validate(data.base, data.multiplier, dtype, len)
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

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let metadata = SequenceMetadata {
            base: Some((&array.base()).into()),
            multiplier: Some((&array.multiplier()).into()),
        };

        Ok(Some(metadata.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_ensure!(
            buffers.is_empty(),
            "SequenceArray expects 0 buffers, got {}",
            buffers.len()
        );
        vortex_ensure!(
            children.is_empty(),
            "SequenceArray expects 0 children, got {}",
            children.len()
        );
        let metadata = SequenceMetadata::decode(metadata)?;

        let ptype = dtype.as_ptype();

        // We go via Scalar to validate that the value is valid for the ptype.
        let base = Scalar::from_proto_value(
            metadata
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
            metadata
                .multiplier
                .as_ref()
                .ok_or_else(|| vortex_err!("multiplier required"))?,
            &DType::Primitive(ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array multiplier should be a non-nullable primitive");

        let data = SequenceData::try_new(base, multiplier, ptype, dtype.nullability(), len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
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
    fn stats(multiplier: PValue) -> StatsSet {
        // A sequence A[i] = base + i * multiplier is sorted iff multiplier >= 0,
        // and strictly sorted iff multiplier > 0.
        let (is_sorted, is_strict_sorted) = match_each_pvalue!(
            multiplier,
            uint: |v| { (true, v > 0) },
            int: |v| { (v >= 0, v > 0) },
            float: |_v| { unreachable!("float multiplier not supported") }
        );

        // SAFETY: we don't have duplicate stats.
        unsafe {
            StatsSet::new_unchecked(smallvec![
                (Stat::IsSorted, StatPrecision::Exact(is_sorted.into())),
                (
                    Stat::IsStrictSorted,
                    StatPrecision::Exact(is_strict_sorted.into()),
                ),
            ])
        }
    }

    /// Construct a new [`SequenceArray`] from pre-validated parts.
    ///
    /// # Safety
    ///
    /// Caller must ensure the sequence is logically compatible with the provided dtype and len.
    pub(crate) unsafe fn new_unchecked(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> SequenceArray {
        let dtype = DType::Primitive(ptype, nullability);
        let (base, multiplier) = SequenceData::normalize(base, multiplier, ptype)
            .vortex_expect("SequenceArray parts must be normalized to the target ptype");
        let stats = Self::stats(multiplier);
        let data = unsafe { SequenceData::new_unchecked(base, multiplier) };
        unsafe { Array::from_parts_unchecked(ArrayParts::new(Sequence, dtype, length, data)) }
            .with_stats_set(stats)
    }

    /// Construct a new [`SequenceArray`] from its components.
    pub fn try_new(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<SequenceArray> {
        let dtype = DType::Primitive(ptype, nullability);
        let data = SequenceData::try_new(base, multiplier, ptype, nullability, length)?;
        let stats = Self::stats(data.multiplier());
        Ok(
            unsafe { Array::from_parts_unchecked(ArrayParts::new(Sequence, dtype, length, data)) }
                .with_stats_set(stats),
        )
    }

    /// Construct a new typed [`SequenceArray`] from base/multiplier values.
    pub fn try_new_typed<T: NativePType + Into<PValue>>(
        base: T,
        multiplier: T,
        nullability: Nullability,
        length: usize,
    ) -> VortexResult<SequenceArray> {
        let ptype = T::PTYPE;
        let dtype = DType::Primitive(ptype, nullability);
        let data = SequenceData::try_new_typed(base, multiplier, nullability, length)?;
        let stats = Self::stats(data.multiplier());
        Ok(
            unsafe { Array::from_parts_unchecked(ArrayParts::new(Sequence, dtype, length, data)) }
                .with_stats_set(stats),
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
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
            .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
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
        assert_eq!(is_sorted, StatPrecision::Exact(true));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));
        assert_eq!(is_strict_sorted, StatPrecision::Exact(true));
        Ok(())
    }

    #[test]
    fn zero_multiplier_is_sorted_not_strict() -> VortexResult<()> {
        let arr = Sequence::try_new_typed(5i64, 0, Nullability::NonNullable, 4)?;

        let is_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, StatPrecision::Exact(true));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));
        assert_eq!(is_strict_sorted, StatPrecision::Exact(false));
        Ok(())
    }

    #[test]
    fn negative_multiplier_not_sorted() -> VortexResult<()> {
        let arr = Sequence::try_new_typed(10i64, -1, Nullability::NonNullable, 4)?;

        let is_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, StatPrecision::Exact(false));

        let is_strict_sorted = arr
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsStrictSorted));
        assert_eq!(is_strict_sorted, StatPrecision::Exact(false));
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

        assert_eq!(is_sorted, StatPrecision::Exact(true));
        assert_eq!(is_strict_sorted, StatPrecision::Exact(true));

        Ok(())
    }
}
