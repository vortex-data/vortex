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

use crate::FloatQuantMetadata;

/// A [`FloatQuant`]-encoded Vortex array.
///
/// Decomposes an `f64` input stream into a `(primary, secondary)` pair of
/// `u64` children that split each value's raw bit pattern at a fixed
/// quantization boundary `k` (`1..=63`):
///
/// - `primary[i] = x[i].to_bits() >> k` — the high `64 - k` bits.
/// - `secondary[i] = x[i].to_bits() & ((1u64 << k) - 1)` — the low `k` bits.
///
/// Decode reconstructs `out[i] = f64::from_bits((primary[i] << k) | secondary[i])`.
/// The round-trip is bit-exact for every `f64` including NaN, both infinities,
/// and `+/- 0.0`. When the low `k` mantissa bits carry no signal `secondary`
/// is highly predictable — often all zero — and compresses well in downstream
/// entropy-coded layers.
pub type FloatQuantArray = Array<FloatQuant>;

/// Slot holding the high `64 - k` bits of each value.
pub(crate) const PRIMARY_SLOT: usize = 0;
/// Slot holding the low `k` bits of each value.
pub(crate) const SECONDARY_SLOT: usize = 1;
const NUM_SLOTS: usize = 2;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["primary", "secondary"];

/// Minimum quantization boundary (inclusive). `k = 0` is the identity mode
/// and offers no compression value here.
const MIN_K: u32 = 1;
/// Maximum quantization boundary (inclusive). `k = 64` would collapse
/// `primary` to zero and is rejected.
const MAX_K: u32 = 63;

/// Marker type implementing [`VTable`] for the [`FloatQuant`] mode array.
///
/// FloatQuant is parameter-less in the type system; the only state it carries
/// at construction time is the quantization boundary `k` stored in
/// [`FloatQuantMetadata`].
#[derive(Clone, Debug)]
pub struct FloatQuant;

/// Per-array data for [`FloatQuantArray`]. Carries only the boundary `k`.
#[derive(Clone, Debug)]
pub struct FloatQuantData {
    k: u32,
}

impl FloatQuantData {
    /// Create new FloatQuant data from a validated boundary.
    pub(crate) fn new(k: u32) -> Self {
        Self { k }
    }

    /// Returns the quantization boundary (number of low bits in `secondary`).
    pub fn k(&self) -> u32 {
        self.k
    }
}

impl Display for FloatQuantData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "k: {}", self.k)
    }
}

impl ArrayHash for FloatQuantData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.k.hash(state);
    }
}

impl ArrayEq for FloatQuantData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.k == other.k
    }
}

impl VTable for FloatQuant {
    type TypedArrayData = FloatQuantData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.float_quant");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let primary = slots[PRIMARY_SLOT]
            .as_ref()
            .vortex_expect("FloatQuantArray primary slot");
        let secondary = slots[SECONDARY_SLOT]
            .as_ref()
            .vortex_expect("FloatQuantArray secondary slot");
        validate_children(data.k, dtype, primary, secondary, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FloatQuantArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FloatQuantArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            FloatQuantMetadata { k: array.data().k }.encode_to_vec(),
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
        let metadata = FloatQuantMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        ensure_f64_dtype(dtype)?;
        ensure_k_valid(metadata.k)?;

        let child_dtype = DType::Primitive(PType::U64, dtype.nullability());
        let primary = children.get(PRIMARY_SLOT, &child_dtype, len)?;
        let secondary = children.get(SECONDARY_SLOT, &child_dtype, len)?;
        let slots = smallvec![Some(primary.clone()), Some(secondary.clone())];
        let data = FloatQuantData::new(metadata.k);
        validate_children(metadata.k, dtype, &primary, &secondary, len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let k = array.data().k;
        let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
        let secondary = array.secondary().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(ExecutionResult::done(
            decode_primitive(primary, secondary, k).into_array(),
        ))
    }
}

/// Ensure the parent dtype is `f64` (the only width FloatQuant supports here).
fn ensure_f64_dtype(dtype: &DType) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    if ptype != PType::F64 {
        vortex_bail!("FloatQuantArray only supports f64 inputs, got {ptype}");
    }
    Ok(())
}

/// Ensure `k` is in the inclusive range `[1, 63]`.
fn ensure_k_valid(k: u32) -> VortexResult<()> {
    if !(MIN_K..=MAX_K).contains(&k) {
        vortex_bail!("FloatQuant k must be in 1..=63, got {k}");
    }
    Ok(())
}

/// Validate that `primary` and `secondary` children are `u64` of the same
/// nullability and length as the array, and that `k` is well-formed.
fn validate_children(
    k: u32,
    dtype: &DType,
    primary: &ArrayRef,
    secondary: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    ensure_f64_dtype(dtype)?;
    ensure_k_valid(k)?;
    let child_dtype = DType::Primitive(PType::U64, dtype.nullability());
    vortex_ensure!(
        primary.dtype() == &child_dtype,
        "FloatQuantArray primary dtype {} does not match expected {}",
        primary.dtype(),
        child_dtype,
    );
    vortex_ensure!(
        secondary.dtype() == &child_dtype,
        "FloatQuantArray secondary dtype {} does not match expected {}",
        secondary.dtype(),
        child_dtype,
    );
    vortex_ensure!(
        primary.len() == len,
        "FloatQuantArray primary len {} does not match array len {len}",
        primary.len(),
    );
    vortex_ensure!(
        secondary.len() == len,
        "FloatQuantArray secondary len {} does not match array len {len}",
        secondary.len(),
    );
    Ok(())
}

/// Extension methods on any typed reference to a [`FloatQuantArray`].
pub trait FloatQuantArrayExt: TypedArrayRef<FloatQuant> {
    /// The quantization boundary `k` (number of low bits in `secondary`).
    fn k(&self) -> u32 {
        // `TypedArrayRef` derefs to `FloatQuantData`.
        FloatQuantData::k(self)
    }

    /// The high-bits child (u64) carrying `bits >> k`.
    fn primary(&self) -> &ArrayRef {
        self.as_ref().slots()[PRIMARY_SLOT]
            .as_ref()
            .vortex_expect("FloatQuantArray primary slot")
    }

    /// The low-bits child (u64) carrying `bits & ((1 << k) - 1)`.
    fn secondary(&self) -> &ArrayRef {
        self.as_ref().slots()[SECONDARY_SLOT]
            .as_ref()
            .vortex_expect("FloatQuantArray secondary slot")
    }
}

impl<T: TypedArrayRef<FloatQuant>> FloatQuantArrayExt for T {}

impl FloatQuant {
    /// Construct a [`FloatQuantArray`] from validated `primary` and `secondary`
    /// `u64` children. The returned array's logical dtype is `f64` with
    /// nullability inherited from the children.
    ///
    /// Validity flows from the `primary` child via [`ValidityVTableFromChild`].
    pub fn try_new(
        primary: ArrayRef,
        secondary: ArrayRef,
        k: u32,
    ) -> VortexResult<FloatQuantArray> {
        let dtype = DType::Primitive(PType::F64, primary.dtype().nullability());
        let len = primary.len();
        validate_children(k, &dtype, &primary, &secondary, len)?;
        let slots = smallvec![Some(primary), Some(secondary)];
        let data = FloatQuantData::new(k);
        // SAFETY: validate_children above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FloatQuant, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode an `f64` primitive array using pco's FloatQuant mode.
    ///
    /// `k` must be in `1..=63`. `k == 0` is just the identity mode and not
    /// useful here; `k == 64` would collapse `primary` to zero and is
    /// rejected. The decomposition is bit-exact: `decode(encode(x)) == x`
    /// element-by-element for every `f64` including NaN, infinities, and
    /// `+/- 0.0`.
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        k: u32,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<FloatQuantArray> {
        let ptype = PrimitiveArrayExt::ptype(&parray);
        if ptype != PType::F64 {
            vortex_bail!("FloatQuantArray::encode requires f64 input, got {ptype}");
        }
        ensure_k_valid(k)?;

        let parray = parray.into_owned();
        let validity = PrimitiveArrayExt::validity(&parray);
        let (primary, secondary) = split_buffer(parray.into_buffer::<f64>(), k);

        Self::try_new(
            PrimitiveArray::new(primary, validity.clone()).into_array(),
            PrimitiveArray::new(secondary, validity).into_array(),
            k,
        )
    }
}

/// Split an `f64` buffer into `(primary: u64, secondary: u64)` according to
/// the FloatQuant decomposition at boundary `k`.
fn split_buffer(values: Buffer<f64>, k: u32) -> (Buffer<u64>, Buffer<u64>) {
    let slice = values.as_slice();
    let len = slice.len();
    let mut primary = BufferMut::<u64>::with_capacity(len);
    let mut secondary = BufferMut::<u64>::with_capacity(len);
    let mask = low_mask(k);
    for &x in slice {
        let bits = x.to_bits();
        primary.push(bits >> k);
        secondary.push(bits & mask);
    }
    (primary.freeze(), secondary.freeze())
}

/// Mask isolating the low `k` bits. `k` must satisfy `1 <= k <= 63` so the
/// shift never overflows.
#[inline]
fn low_mask(k: u32) -> u64 {
    (1u64 << k) - 1
}

/// Reconstruct a single `f64` from a `(primary, secondary)` pair at boundary `k`.
#[inline]
fn decode_one(primary: u64, secondary: u64, k: u32) -> f64 {
    f64::from_bits((primary << k) | secondary)
}

/// Recompose an `f64` `PrimitiveArray` from the two `u64` children. Validity
/// is taken from `primary`.
fn decode_primitive(primary: PrimitiveArray, secondary: PrimitiveArray, k: u32) -> PrimitiveArray {
    let validity = PrimitiveArrayExt::validity(&primary);
    let p_buf = primary.into_buffer::<u64>();
    let s_buf = secondary.into_buffer::<u64>();
    let p_slice = p_buf.as_slice();
    let s_slice = s_buf.as_slice();
    debug_assert_eq!(p_slice.len(), s_slice.len());
    let mut out = BufferMut::<f64>::with_capacity(p_slice.len());
    for (p, s) in p_slice.iter().zip(s_slice.iter()) {
        out.push(decode_one(*p, *s, k));
    }
    PrimitiveArray::new(out.freeze(), validity)
}

impl OperationsVTable<FloatQuant> for FloatQuant {
    fn scalar_at(
        array: ArrayView<'_, FloatQuant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let k = array.data().k;
        let nullability = array.dtype().nullability();
        let primary_scalar = array.primary().execute_scalar(index, ctx)?;
        if primary_scalar.is_null() {
            return Scalar::try_new(array.dtype().clone(), None);
        }
        let secondary_scalar = array.secondary().execute_scalar(index, ctx)?;
        let primary = primary_scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("FloatQuant primary scalar must be u64");
        let secondary = secondary_scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("FloatQuant secondary scalar must be u64");
        Ok(Scalar::primitive(
            decode_one(primary, secondary, k),
            nullability,
        ))
    }
}

impl ValidityChild<FloatQuant> for FloatQuant {
    fn validity_child(array: ArrayView<'_, FloatQuant>) -> ArrayRef {
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

    /// Encode + decode round-trip an `f64` array, checking the decode matches
    /// the input bit-for-bit.
    fn round_trip_bits(input: &[f64], k: u32) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(input.iter().copied());
        let encoded = FloatQuant::encode(parray.as_view(), k, &mut ctx)?;
        assert_eq!(encoded.dtype(), parray.dtype());
        assert_eq!(encoded.len(), parray.len());
        assert_eq!(encoded.k(), k);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        let out = decoded.into_buffer::<f64>();
        assert_eq!(out.len(), input.len());
        for (i, (got, want)) in out.as_slice().iter().zip(input.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "mismatch at {i}: got {got:?} ({:#x}), want {want:?} ({:#x})",
                got.to_bits(),
                want.to_bits(),
            );
        }
        Ok(())
    }

    /// FloatQuant-favorable input: random f64 values with their low `k` bits
    /// cleared. The `secondary` child should be all zeros and the round-trip
    /// must still be bit-exact.
    #[test]
    fn round_trip_favorable() -> VortexResult<()> {
        let k = 16u32;
        let mask = !((1u64 << k) - 1);
        let mut rng = SmallRng::seed_from_u64(0xF00D);
        let values: Vec<f64> = (0..512)
            .map(|_| f64::from_bits(rng.random::<u64>() & mask))
            .collect();
        round_trip_bits(&values, k)?;

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = FloatQuant::encode(parray.as_view(), k, &mut ctx)?;
        let secondary = encoded
            .secondary()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u64>();
        assert!(
            secondary.as_slice().iter().all(|&s| s == 0),
            "favorable input should yield all-zero secondary"
        );
        Ok(())
    }

    /// Random arbitrary `f64` inputs round-trip bit-for-bit even when the
    /// low `k` bits are non-zero.
    #[test]
    fn round_trip_arbitrary_random() -> VortexResult<()> {
        let mut rng = SmallRng::seed_from_u64(0xBEEF);
        let values: Vec<f64> = (0..1024)
            .map(|_| f64::from_bits(rng.random::<u64>()))
            .collect();
        round_trip_bits(&values, 16)
    }

    /// Special values (NaN, +/-inf, +/-0, subnormals, MAX) round-trip
    /// bit-for-bit at the same `k`.
    #[test]
    fn round_trip_special_values() -> VortexResult<()> {
        let values = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0_f64,
            -0.0_f64,
            f64::MIN_POSITIVE,
            f64::MAX,
        ];
        round_trip_bits(&values, 16)
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let k = 8u32;
        let mask = !((1u64 << k) - 1);
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let values: Vec<f64> = (0..50)
            .map(|_| f64::from_bits(rng.random::<u64>() & mask))
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = FloatQuant::encode(parray.as_view(), k, &mut ctx)?;
        let sliced = encoded.into_array().slice(10..30)?;
        let expected = PrimitiveArray::from_iter(values[10..30].iter().copied());
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn nullable_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let k = 16u32;
        let input = PrimitiveArray::new(
            buffer![0.0_f64, 1.5, 2.25, -3.75, 4.5, f64::NAN],
            Validity::from_iter([true, false, true, false, true, true]),
        );
        let encoded = FloatQuant::encode(input.as_view(), k, &mut ctx)?;

        let s1 = encoded.clone().into_array().execute_scalar(1, &mut ctx)?;
        let s3 = encoded.clone().into_array().execute_scalar(3, &mut ctx)?;
        assert!(s1.is_null());
        assert!(s3.is_null());

        let s0 = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(
            s0.as_primitive().typed_value::<f64>().map(f64::to_bits),
            Some(0.0_f64.to_bits())
        );
        let s2 = encoded.clone().into_array().execute_scalar(2, &mut ctx)?;
        assert_eq!(
            s2.as_primitive().typed_value::<f64>().map(f64::to_bits),
            Some(2.25_f64.to_bits())
        );

        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input);
        Ok(())
    }

    #[rstest]
    #[case::zero(0)]
    #[case::sixty_four(64)]
    #[case::one_hundred(100)]
    fn rejects_invalid_k(#[case] k: u32) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([1.0_f64, 2.0, 3.0]);
        let err = FloatQuant::encode(parray.as_view(), k, &mut ctx);
        assert!(err.is_err(), "expected error for k {k}, got {err:?}");
        Ok(())
    }

    #[test]
    fn rejects_non_f64_input() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([1.0_f32, 2.0, 3.0]);
        let err = FloatQuant::encode(parray.as_view(), 16, &mut ctx);
        assert!(err.is_err(), "expected error for f32 input, got {err:?}");

        let parray = PrimitiveArray::from_iter([1i64, 2, 3]);
        let err = FloatQuant::encode(parray.as_view(), 16, &mut ctx);
        assert!(err.is_err(), "expected error for i64 input, got {err:?}");
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let k = 16u32;
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let values: Vec<f64> = (0..256)
            .map(|_| f64::from_bits(rng.random::<u64>()))
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = FloatQuant::encode(parray.as_view(), k, &mut ctx)?;
        let arr = encoded.into_array();

        let decoded = arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<f64>();

        let mut idx_rng = SmallRng::seed_from_u64(0xD00D);
        let indices: Vec<usize> = (0..32).map(|_| idx_rng.random_range(0..256)).collect();
        for &i in &indices {
            let scalar = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(scalar, Scalar::from(decoded.as_slice()[i]));
        }
        Ok(())
    }

    #[rstest]
    #[case::k_one(1)]
    #[case::k_sixty_three(63)]
    fn round_trip_boundary_k(#[case] k: u32) -> VortexResult<()> {
        let mut rng = SmallRng::seed_from_u64(0xBA5E);
        let values: Vec<f64> = (0..256)
            .map(|_| f64::from_bits(rng.random::<u64>()))
            .collect();
        round_trip_bits(&values, k)
    }
}
