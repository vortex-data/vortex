// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::TypedArrayRef;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ConsecutiveDeltaMetadata;

/// A first-order consecutive-delta encoded Vortex array of `i64` values.
///
/// Decodes via prefix sum from a stored `seed` (the first element) and a
/// child `primary` array of consecutive differences:
///
/// ```text
/// out[0] = seed
/// out[i] = out[i - 1].wrapping_add(primary[i - 1])  for i in 1..N
/// ```
///
/// The encoding uses wrapping arithmetic, so the round-trip is bit-exact
/// for any `i64` input.
///
/// # Random-access cliff
///
/// [`OperationsVTable::scalar_at`] must replay the prefix sum from element
/// zero and is therefore **O(i)**. This is the first layer of the layered
/// pco stack that breaks element-level random access; the cost is measured
/// in `benches/consecutive_delta.rs`.
///
/// # Restrictions
///
/// Only `i64` is supported in this phase. Nullable input is rejected at
/// encode time — see [`ConsecutiveDelta::encode`].
pub type ConsecutiveDeltaArray = Array<ConsecutiveDelta>;

/// Slot holding the consecutive-difference child.
pub(crate) const PRIMARY_SLOT: usize = 0;
const NUM_SLOTS: usize = 1;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["primary"];

/// Marker type implementing [`VTable`] for the [`ConsecutiveDelta`] array.
///
/// The only per-array state is the `seed` stored in
/// [`ConsecutiveDeltaMetadata`].
#[derive(Clone, Debug)]
pub struct ConsecutiveDelta;

/// Per-array data for [`ConsecutiveDeltaArray`]. Carries only the `seed`.
#[derive(Clone, Debug)]
pub struct ConsecutiveDeltaData {
    seed: i64,
}

impl ConsecutiveDeltaData {
    /// Create new ConsecutiveDelta data from a validated `seed`.
    pub(crate) fn new(seed: i64) -> Self {
        Self { seed }
    }

    /// Returns the first absolute value (`x[0]`) of the encoded stream.
    pub fn seed(&self) -> i64 {
        self.seed
    }
}

impl Display for ConsecutiveDeltaData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "seed: {}", self.seed)
    }
}

impl ArrayHash for ConsecutiveDeltaData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.seed.hash(state);
    }
}

impl ArrayEq for ConsecutiveDeltaData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.seed == other.seed
    }
}

impl VTable for ConsecutiveDelta {
    type TypedArrayData = ConsecutiveDeltaData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.consecutive_delta");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let primary = slots[PRIMARY_SLOT]
            .as_ref()
            .vortex_expect("ConsecutiveDeltaArray primary slot");
        validate_parts(dtype, primary, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ConsecutiveDeltaArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("ConsecutiveDeltaArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            ConsecutiveDeltaMetadata {
                seed: array.data().seed,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = ConsecutiveDeltaMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        ensure_i64_dtype(dtype)?;

        let primary_dtype = DType::Primitive(PType::I64, dtype.nullability());
        let primary_len = primary_len_for(len);
        let primary = children.get(PRIMARY_SLOT, &primary_dtype, primary_len)?;
        let slots = smallvec![Some(primary.clone())];
        let data = ConsecutiveDeltaData::new(metadata.seed);
        validate_parts(dtype, &primary, len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let seed = array.data().seed;
        let len = array.len();
        let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(ExecutionResult::done(
            decode_primitive(primary, seed, len).into_array(),
        ))
    }
}

/// Ensure the parent dtype is `i64` (the only width supported in this phase).
fn ensure_i64_dtype(dtype: &DType) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    if ptype != PType::I64 {
        vortex_bail!("ConsecutiveDeltaArray only supports i64 inputs, got {ptype}");
    }
    Ok(())
}

/// Length of the `primary` child for an array of length `len`. For `N >= 1`
/// the primary holds the `N - 1` consecutive differences; for `N = 0` it is
/// empty.
const fn primary_len_for(len: usize) -> usize {
    len.saturating_sub(1)
}

/// Validate that the `primary` child is an `i64` array of the expected
/// length and nullability for an array of logical length `len`.
fn validate_parts(dtype: &DType, primary: &ArrayRef, len: usize) -> VortexResult<()> {
    ensure_i64_dtype(dtype)?;
    let expected_dtype = DType::Primitive(PType::I64, dtype.nullability());
    vortex_ensure!(
        primary.dtype() == &expected_dtype,
        "ConsecutiveDeltaArray primary dtype {} does not match expected {}",
        primary.dtype(),
        expected_dtype,
    );
    let expected_primary_len = primary_len_for(len);
    vortex_ensure!(
        primary.len() == expected_primary_len,
        "ConsecutiveDeltaArray primary len {} does not match expected {expected_primary_len} \
         (array len {len})",
        primary.len(),
    );
    Ok(())
}

/// Extension methods on any typed reference to a [`ConsecutiveDeltaArray`].
pub trait ConsecutiveDeltaArrayExt: TypedArrayRef<ConsecutiveDelta> {
    /// The first absolute value (`x[0]`) of the encoded stream.
    fn seed(&self) -> i64 {
        ConsecutiveDeltaData::seed(self)
    }

    /// The child array of consecutive differences.
    fn primary(&self) -> &ArrayRef {
        self.as_ref().slots()[PRIMARY_SLOT]
            .as_ref()
            .vortex_expect("ConsecutiveDeltaArray primary slot")
    }
}

impl<T: TypedArrayRef<ConsecutiveDelta>> ConsecutiveDeltaArrayExt for T {}

impl ConsecutiveDelta {
    /// Construct a [`ConsecutiveDeltaArray`] from a validated `primary`
    /// child and a `seed`.
    ///
    /// `primary` must be `Primitive<i64>` of length `len - 1` (or empty when
    /// `len == 0`) with [`Validity::NonNullable`]. The returned array's
    /// logical dtype is `i64` non-nullable; validity flows from the
    /// `primary` child via [`ValidityVTableFromChild`].
    pub fn try_new(
        seed: i64,
        primary: ArrayRef,
        len: usize,
    ) -> VortexResult<ConsecutiveDeltaArray> {
        let dtype = DType::Primitive(PType::I64, primary.dtype().nullability());
        validate_parts(&dtype, &primary, len)?;
        let slots = smallvec![Some(primary)];
        let data = ConsecutiveDeltaData::new(seed);
        // SAFETY: validate_parts above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(ConsecutiveDelta, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode an `i64` primitive array as first-order consecutive deltas.
    ///
    /// Returns a [`ConsecutiveDeltaArray`] whose decode is bit-exact with
    /// the input under wrapping arithmetic.
    ///
    /// # Errors
    ///
    /// - Returns an error if the input dtype is not `i64`.
    /// - Returns an error if the input is nullable (see the
    ///   [crate-level docs](crate) for context).
    ///
    /// # Edge cases
    ///
    /// - Empty input is permitted; the result has `seed = 0` and an empty
    ///   `primary` child.
    /// - Singleton input is permitted; the result has `seed = x[0]` and an
    ///   empty `primary` child.
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ConsecutiveDeltaArray> {
        let ptype = PrimitiveArrayExt::ptype(&parray);
        if ptype != PType::I64 {
            vortex_bail!("ConsecutiveDeltaArray::encode requires i64 input, got {ptype}");
        }
        let validity = PrimitiveArrayExt::validity(&parray);
        if !matches!(validity, Validity::NonNullable) {
            vortex_bail!(
                "ConsecutiveDeltaArray::encode requires non-nullable input; nullable streams \
                 are not supported in this phase"
            );
        }

        let len = parray.array().len();
        let parray = parray.into_owned();
        let buf = parray.into_buffer::<i64>();
        let (seed, primary_buf) = split_buffer(buf);
        let primary = PrimitiveArray::new(primary_buf, Validity::NonNullable).into_array();
        Self::try_new(seed, primary, len)
    }
}

/// Produce `(seed, primary)` from an input `i64` buffer. The primary holds
/// `N - 1` wrapping differences; for `N == 0` we return `seed = 0` and an
/// empty buffer; for `N == 1` we return `seed = x[0]` and an empty buffer.
fn split_buffer(values: Buffer<i64>) -> (i64, Buffer<i64>) {
    let slice = values.as_slice();
    if slice.is_empty() {
        return (0, Buffer::<i64>::empty());
    }
    let seed = slice[0];
    let n_deltas = slice.len() - 1;
    let mut out = BufferMut::<i64>::with_capacity(n_deltas);
    let mut prev = seed;
    for &v in &slice[1..] {
        out.push(v.wrapping_sub(prev));
        prev = v;
    }
    (seed, out.freeze())
}

/// Reconstruct the absolute values from `seed` and the deltas in `primary`.
/// Always returns a non-nullable `PrimitiveArray<i64>` of length `len`.
fn decode_primitive(primary: PrimitiveArray, seed: i64, len: usize) -> PrimitiveArray {
    debug_assert_eq!(primary.len(), primary_len_for(len));
    let mut out = BufferMut::<i64>::with_capacity(len);
    if len == 0 {
        return PrimitiveArray::new(out.freeze(), Validity::NonNullable);
    }
    let deltas = primary.into_buffer::<i64>();
    let mut acc = seed;
    out.push(acc);
    for &d in deltas.as_slice() {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
    PrimitiveArray::new(out.freeze(), Validity::NonNullable)
}

impl OperationsVTable<ConsecutiveDelta> for ConsecutiveDelta {
    /// Returns the absolute value at `index` by replaying the prefix sum.
    ///
    /// **O(i)** in the array length: every call materialises the running
    /// accumulator from element zero. See the type-level docs on
    /// [`ConsecutiveDeltaArray`] for context.
    fn scalar_at(
        array: ArrayView<'_, ConsecutiveDelta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let seed = array.data().seed;
        let nullability = array.dtype().nullability();
        if index == 0 {
            return Ok(Scalar::primitive(seed, nullability));
        }
        // Replay from element zero. We execute the primary child once and
        // walk its slice; reusing the executed buffer avoids one
        // `execute_scalar` call per step.
        let primary = array
            .primary()
            .clone()
            .execute::<PrimitiveArray>(ctx)?
            .into_buffer::<i64>();
        let deltas = primary.as_slice();
        let mut acc = seed;
        for &d in &deltas[..index] {
            acc = acc.wrapping_add(d);
        }
        Ok(Scalar::primitive(acc, nullability))
    }
}

impl ValidityChild<ConsecutiveDelta> for ConsecutiveDelta {
    fn validity_child(array: ArrayView<'_, ConsecutiveDelta>) -> ArrayRef {
        array.primary().clone()
    }
}

#[cfg(test)]
mod tests {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    fn round_trip(values: Vec<i64>) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)?;
        assert_eq!(encoded.dtype(), parray.dtype());
        assert_eq!(encoded.len(), parray.len());
        assert_eq!(encoded.primary().len(), primary_len_for(values.len()));
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[test]
    fn monotone_round_trip_primary_is_constant_step() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<i64> = (0..1024).map(|i: i64| 1_000_000 + i * 7).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)?;

        assert_eq!(encoded.seed(), 1_000_000);
        let primary = encoded
            .primary()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<i64>();
        assert!(primary.as_slice().iter().all(|&d| d == 7));

        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[test]
    fn random_i64_round_trip() -> VortexResult<()> {
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let values: Vec<i64> = (0..1024).map(|_| rng.random::<i64>()).collect();
        round_trip(values)
    }

    #[rstest]
    #[case::boundaries(vec![i64::MIN, i64::MAX, 0, -1, 1])]
    #[case::min_to_max(vec![i64::MIN, i64::MAX])]
    #[case::max_to_min(vec![i64::MAX, i64::MIN])]
    #[case::around_zero(vec![-1, 0, 1, 0, -1])]
    #[case::repeated_min(vec![i64::MIN, i64::MIN, i64::MIN])]
    fn edge_value_round_trip(#[case] values: Vec<i64>) -> VortexResult<()> {
        round_trip(values)
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<i64> = (0..200).map(|i: i64| 10 + i * 3).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)?;
        let sliced = encoded.into_array().slice(40..160)?;
        let expected = PrimitiveArray::from_iter(values[40..160].iter().copied());
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn empty_input_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(Vec::<i64>::new());
        let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)?;
        assert_eq!(encoded.len(), 0);
        assert_eq!(encoded.primary().len(), 0);
        assert_eq!(encoded.seed(), 0);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[test]
    fn singleton_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([42i64]);
        let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)?;
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded.primary().len(), 0);
        assert_eq!(encoded.seed(), 42);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[test]
    fn rejects_nullable_input() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let input = PrimitiveArray::new(
            buffer![1i64, 2, 3, 4, 5],
            Validity::from_iter([true, true, false, true, true]),
        );
        let err = ConsecutiveDelta::encode(input.as_view(), &mut ctx);
        assert!(
            err.is_err(),
            "expected error for nullable input, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn rejects_all_valid_nullable_input() -> VortexResult<()> {
        // `Validity::AllValid` carries a nullable logical dtype even though no
        // value is actually null. We still reject it: the rule is on the
        // logical nullability, not the runtime null count.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let input = PrimitiveArray::new(buffer![1i64, 2, 3], Validity::AllValid);
        let err = ConsecutiveDelta::encode(input.as_view(), &mut ctx);
        assert!(
            err.is_err(),
            "expected error for AllValid input, got {err:?}"
        );
        Ok(())
    }

    #[rstest]
    #[case::f64(PrimitiveArray::from_iter([1.0_f64, 2.0, 3.0]))]
    #[case::u64(PrimitiveArray::from_iter([1u64, 2, 3]))]
    #[case::i32(PrimitiveArray::from_iter([1i32, 2, 3]))]
    fn rejects_non_i64_input(#[case] parray: PrimitiveArray) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let err = ConsecutiveDelta::encode(parray.as_view(), &mut ctx);
        assert!(
            err.is_err(),
            "expected error for non-i64 input, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut rng = SmallRng::seed_from_u64(0xABBA);
        let values: Vec<i64> = (0..512).map(|_| rng.random::<i64>()).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = ConsecutiveDelta::encode(parray.as_view(), &mut ctx)?;
        let arr = encoded.into_array();

        let mut idx_rng = SmallRng::seed_from_u64(0xBEEF);
        for _ in 0..64 {
            let i = idx_rng.random_range(0..values.len());
            let scalar = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(scalar, Scalar::from(values[i]));
        }

        // Also check the boundaries.
        assert_eq!(arr.execute_scalar(0, &mut ctx)?, Scalar::from(values[0]));
        assert_eq!(
            arr.execute_scalar(values.len() - 1, &mut ctx)?,
            Scalar::from(values[values.len() - 1])
        );
        Ok(())
    }
}
