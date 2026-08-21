// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use num_traits::AsPrimitive;
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
use crate::eval;
use crate::eval::SequenceValue;
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

/// An array representing the equation `A[i] = base + i * multiplier`.
///
/// `base` is the array's first value, held in the array's output ptype. `multiplier` is the step
/// between values, which need not be representable in the output ptype - a descending sequence
/// like `100, 90, ..., 60` has a negative step yet is a legal `u8` array - so it is held as an
/// `i64`, or as a `u64` for the steps above `i64::MAX` that no `i64` can express.
///
/// Construction validates the first and last values against the output ptype, so every value of
/// the sequence fits it - see [`crate::eval`] for how values are computed from these two.
#[derive(Clone, Debug)]
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

    /// Constructs a sequence array using two integer values, validated against output `ptype`.
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
            vortex_bail!("only integer ptypes are supported in SequenceArray currently")
        }

        vortex_ensure!(length > 0, "SequenceArray length must be greater than zero");

        Self::narrowed_base(base, *ptype)?;
        Self::ensure_last_expressible(base, multiplier, *ptype, length)
    }

    /// Checks that the last value `base + (length - 1) * multiplier` fits `ptype`, without
    /// computing it: the steps the sequence takes must not exceed the room between `base` and the
    /// ptype's boundary in the step's direction. A ptype's full range spans at most `u64::MAX`,
    /// so the room, like the step's magnitude, is exact in `u64` - no arithmetic wider than 64
    /// bits is needed, even for values above `i64::MAX`.
    fn ensure_last_expressible(
        base: PValue,
        multiplier: PValue,
        ptype: PType,
        length: usize,
    ) -> VortexResult<()> {
        let steps = (length - 1) as u64;
        let (ascending, magnitude) = eval::step_parts(multiplier)
            .ok_or_else(|| vortex_err!("step {multiplier} must be an integer"))?;
        if steps == 0 || magnitude == 0 {
            return Ok(());
        }

        // `base` fits `ptype` (checked in `narrowed_base`), so it casts into the domain of the
        // ptype's signedness: `u64` for unsigned ptypes - keeping bases above `i64::MAX` exact -
        // and `i64` for signed ones.
        let room = if ptype.is_signed_int() {
            let base = base.cast::<i64>()?;
            let max = i64::try_from(ptype.max_value_as_u64())
                .vortex_expect("a signed ptype's max fits i64");
            base.abs_diff(if ascending { max } else { -max - 1 })
        } else {
            let base = base.cast::<u64>()?;
            if ascending {
                ptype.max_value_as_u64() - base
            } else {
                base
            }
        };

        vortex_ensure!(
            steps <= room / magnitude,
            "final value not expressible, base = {base:?}, multiplier = {multiplier:?}, len = {length}"
        );
        Ok(())
    }

    /// The step's ptype: the serialized form preserves its signedness but not its width.
    fn multiplier_ptype_from_proto(
        multiplier: &vortex_proto::scalar::ScalarValue,
    ) -> VortexResult<PType> {
        use vortex_proto::scalar::scalar_value::Kind;
        match multiplier
            .kind
            .as_ref()
            .ok_or_else(|| vortex_err!("multiplier value missing kind"))?
        {
            Kind::Int64Value(_) => Ok(PType::I64),
            Kind::Uint64Value(_) => Ok(PType::U64),
            _ => vortex_bail!("only integer ptypes are supported in SequenceArray currently"),
        }
    }

    /// The array's first value, which has to be expressible in the output ptype.
    fn narrowed_base(base: PValue, ptype: PType) -> VortexResult<PValue> {
        vortex_ensure!(base.ptype().is_int(), "base {base} must be an integer");
        match_each_integer_ptype!(ptype, |P| { Ok(PValue::from(base.cast::<P>()?)) })
    }

    /// Puts `base` into the output ptype and the step into its canonical ptype.
    fn normalize(base: PValue, multiplier: PValue, ptype: PType) -> VortexResult<(PValue, PValue)> {
        let base = Self::narrowed_base(base, ptype)?;

        // The step is canonically an `i64`, or a `u64` when it does not fit one, so that a
        // sequence has one representation whichever ptypes it was built from.
        let multiplier = match_each_pvalue!(
            multiplier,
            uint: |v| {
                let v: u64 = v.as_();
                i64::try_from(v).map(PValue::from).unwrap_or(PValue::U64(v))
            },
            int: |v| {
                let v: i64 = v.as_();
                PValue::from(v)
            },
            float: |v| { vortex_bail!("step {v} must be an integer") }
        );

        Ok((base, multiplier))
    }

    /// Constructs a [`SequenceArray`] payload without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `base` is an integer value in the outer dtype's ptype.
    /// - `multiplier` is an integer value, canonically an `i64` unless the step exceeds
    ///   `i64::MAX`.
    /// - the first and last value of the sequence they describe fit the outer dtype's ptype.
    pub(crate) unsafe fn new_unchecked(base: PValue, multiplier: PValue) -> Self {
        Self { base, multiplier }
    }

    /// The array's output ptype, which `base` is held in - see [`SequenceData`].
    pub fn ptype(&self) -> PType {
        self.base.ptype()
    }

    pub fn base(&self) -> PValue {
        self.base
    }

    pub fn multiplier(&self) -> PValue {
        self.multiplier
    }

    /// `base` and `multiplier` reduced into `O`, the type the values are computed in.
    pub(crate) fn wrapping_parts<O: SequenceValue>(&self) -> VortexResult<(O, O)> {
        eval::wrapping_parts(self.base, self.multiplier).ok_or_else(|| {
            vortex_err!(
                "SequenceArray values must be integers, got base {:?} and step {:?}",
                self.base,
                self.multiplier
            )
        })
    }

    /// The two's-complement bits of `base` and `multiplier`, widened to 64 bits by sign-extending
    /// a negative step.
    ///
    /// A kernel materializing the sequence computes `base + i * multiplier` from these in an
    /// integer type at least as wide as the output ptype, wrapping on overflow, and truncates the
    /// result to the output ptype - see [`crate::eval`].
    pub fn wrapping_bits(&self) -> VortexResult<(u64, u64)> {
        self.wrapping_parts::<u64>()
    }

    pub fn into_parts(self) -> SequenceDataParts {
        SequenceDataParts {
            base: self.base,
            multiplier: self.multiplier,
            ptype: self.ptype(),
        }
    }

    pub(crate) fn index_value(&self, idx: usize) -> PValue {
        match_each_integer_ptype!(self.ptype(), |O| {
            let (base, multiplier) = self
                .wrapping_parts::<O>()
                .vortex_expect("sequence values are integers");
            PValue::from(eval::wrapping_value(base, multiplier, idx))
        })
    }
}

// `base` is held in the array's output ptype and the step canonically as an `i64`, or a `u64`
// above `i64::MAX`, so equal sequences hold identically-tagged values. Comparing the tags first
// also keeps `PValue`'s `PartialEq` - which panics comparing a negative `i64` against a `u64`
// above `i64::MAX` - on same-signedness pairs it can handle.
impl ArrayHash for SequenceData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.base.hash(state);
        self.multiplier.hash(state);
    }
}

impl ArrayEq for SequenceData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.base.ptype() == other.base.ptype()
            && self.multiplier.ptype() == other.multiplier.ptype()
            && self.base == other.base
            && self.multiplier == other.multiplier
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

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
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
        let DType::Primitive(output_ptype, _) = dtype else {
            vortex_bail!(
                "only primitive dtypes are supported in SequenceArray currently, got {dtype}"
            );
        };
        let metadata = SequenceMetadata::decode(metadata)?;

        let base_metadata = metadata
            .base
            .as_ref()
            .ok_or_else(|| vortex_err!("base required"))?;

        let multiplier_metadata = metadata
            .multiplier
            .as_ref()
            .ok_or_else(|| vortex_err!("multiplier required"))?;

        // We go via Scalar to validate that the value is valid for the ptype.
        let base = Scalar::from_proto_value(
            base_metadata,
            &DType::Primitive(*output_ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array base should be a non-nullable primitive");

        // The step is not necessarily representable in the output ptype, only its signedness is
        // recovered from the serialized form.
        let multiplier_ptype = SequenceData::multiplier_ptype_from_proto(multiplier_metadata)?;
        let multiplier = Scalar::from_proto_value(
            multiplier_metadata,
            &DType::Primitive(multiplier_ptype, NonNullable),
            session,
        )?
        .as_primitive()
        .pvalue()
        .vortex_expect("sequence array multiplier should be a non-nullable primitive");

        let data =
            SequenceData::try_new(base, multiplier, *output_ptype, dtype.nullability(), len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        sequence_decompress(&array).map(ExecutionResult::done)
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
    /// `base` and `multiplier` may be of any integer ptype; they are normalized here.
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
            .vortex_expect("SequenceArray parts must be representable in the output ptype");
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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayEq;
    use vortex_array::EqMode;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::stats::Precision as StatPrecision;
    use vortex_array::expr::stats::Stat;
    use vortex_array::expr::stats::StatsProviderExt;
    use vortex_array::scalar::PValue;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::Sequence;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_sequence_canonical() {
        let arr = Sequence::try_new_typed(2i64, 3, Nullability::NonNullable, 4).unwrap();

        let canon = PrimitiveArray::from_iter((0..4).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_sequence_slice_canonical() {
        let arr = Sequence::try_new_typed(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .slice(2..3)
            .unwrap();

        let canon = PrimitiveArray::from_iter((2..3).map(|i| 2i64 + i * 3));

        assert_arrays_eq!(arr, canon, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_sequence_scalar_at() {
        let scalar = Sequence::try_new_typed(2i64, 3, Nullability::NonNullable, 4)
            .unwrap()
            .execute_scalar(2, &mut SESSION.create_execution_ctx())
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

    #[rstest]
    #[case::descending_step_unsigned_output(PValue::from(100i32), PValue::from(-10i32), PType::U8)]
    #[case::narrow_unsigned(PValue::from(1000u32), PValue::from(100u32), PType::U16)]
    #[case::signed_output(PValue::from(0i16), PValue::from(1i16), PType::I32)]
    #[case::signed_step_past_i64_max(PValue::from(0i64), PValue::from(1i64 << 62), PType::U64)]
    #[case::unsigned_step_past_i64_max(PValue::from(0u64), PValue::from(u64::MAX / 4), PType::U64)]
    fn serde_roundtrip_preserves_values(
        #[case] base: PValue,
        #[case] multiplier: PValue,
        #[case] output_ptype: PType,
    ) -> VortexResult<()> {
        let array = Sequence::try_new(base, multiplier, output_ptype, Nullability::NonNullable, 4)?;
        assert_eq!(array.ptype(), output_ptype);

        let dtype = array.dtype().clone();
        let len = array.len();
        let ctx = ArrayContext::empty();
        let serialized =
            array
                .clone()
                .into_array()
                .serialize(&ctx, &SESSION, &SerializeOptions::default())?;

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }

        let decoded = SerializedArray::try_from(concat.freeze())?.decode(
            &dtype,
            len,
            &ReadContext::new(ctx.to_ids()),
            &SESSION,
        )?;

        let decoded_sequence = decoded
            .as_opt::<Sequence>()
            .ok_or_else(|| vortex_err!("decoded array should still be a SequenceArray"))?;
        assert_eq!(decoded_sequence.ptype(), output_ptype);
        assert_eq!(decoded_sequence.multiplier(), array.multiplier());
        assert_eq!(decoded.dtype(), &dtype);
        assert_arrays_eq!(decoded, array, &mut SESSION.create_execution_ctx());

        Ok(())
    }

    /// A descending sequence steps by a negative value yet is a legal unsigned array.
    #[test]
    fn descending_step_unsigned_output() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = Sequence::try_new(
            PValue::from(100i32),
            PValue::from(-10i32),
            PType::U8,
            Nullability::NonNullable,
            5,
        )?;

        assert_arrays_eq!(
            array,
            PrimitiveArray::from_iter([100u8, 90, 80, 70, 60]),
            &mut ctx
        );
        assert_eq!(
            array.clone().into_array().execute_scalar(1, &mut ctx)?,
            Scalar::from(90u8)
        );
        assert_arrays_eq!(
            array.slice(3..5)?,
            PrimitiveArray::from_iter([70u8, 60]),
            &mut ctx
        );

        Ok(())
    }

    /// A step that wraps the whole output range still lands on representable values.
    #[test]
    fn step_spanning_output_range() -> VortexResult<()> {
        let array = Sequence::try_new(
            PValue::from(255u8),
            PValue::from(-255i32),
            PType::U8,
            Nullability::NonNullable,
            2,
        )?;

        assert_arrays_eq!(
            array,
            PrimitiveArray::from_iter([255u8, 0]),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    /// Values above `i64::MAX` are representable, whatever the step's signedness.
    #[test]
    fn values_past_i64_max() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let step = 1u64 << 62;
        let array = Sequence::try_new(
            PValue::from(0i64),
            PValue::from(1i64 << 62),
            PType::U64,
            Nullability::NonNullable,
            4,
        )?;

        assert_arrays_eq!(
            array,
            PrimitiveArray::from_iter([0, step, 2 * step, 3 * step]),
            &mut ctx
        );
        assert_eq!(
            array.into_array().execute_scalar(3, &mut ctx)?,
            Scalar::from(3 * step)
        );

        Ok(())
    }

    /// A negative step and a step above `i64::MAX` are both legal for a `u64` sequence, and
    /// comparing the two must not panic.
    #[test]
    fn eq_across_step_signedness() -> VortexResult<()> {
        // The largest base a step of `1 << 63` still fits in `u64` from.
        let base = PValue::from((1u64 << 63) - 1);
        let descending = Sequence::try_new(
            base,
            PValue::from(-1i64),
            PType::U64,
            Nullability::NonNullable,
            2,
        )?
        .into_array();
        let ascending = Sequence::try_new(
            base,
            PValue::from(1u64 << 63),
            PType::U64,
            Nullability::NonNullable,
            2,
        )?
        .into_array();

        assert!(descending.array_eq(&descending.clone(), EqMode::Value));
        assert!(!descending.array_eq(&ascending, EqMode::Value));

        Ok(())
    }

    /// A constant sequence can be longer than the output ptype's range.
    #[test]
    fn constant_sequence_longer_than_output_range() -> VortexResult<()> {
        let array = Sequence::try_new(
            PValue::from(7u8),
            PValue::from(0i32),
            PType::U8,
            Nullability::NonNullable,
            300,
        )?;

        assert_arrays_eq!(
            array,
            PrimitiveArray::from_iter([7u8; 300]),
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    #[test]
    fn deserialize_rejects_values_outside_output_ptype() -> VortexResult<()> {
        let array = Sequence::try_new_typed(-5i32, 1i32, Nullability::NonNullable, 5)?;
        let len = array.len();
        let ctx = ArrayContext::empty();
        let serialized =
            array
                .into_array()
                .serialize(&ctx, &SESSION, &SerializeOptions::default())?;

        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }

        // The sequence starts at -5, so it is not a valid u8 sequence.
        let decoded = SerializedArray::try_from(concat.freeze())?.decode(
            &DType::Primitive(PType::U8, Nullability::NonNullable),
            len,
            &ReadContext::new(ctx.to_ids()),
            &SESSION,
        );
        assert!(decoded.is_err());

        Ok(())
    }
}
