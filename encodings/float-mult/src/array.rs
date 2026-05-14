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

use crate::FloatMultMetadata;

/// A [`FloatMult`]-encoded Vortex array.
///
/// Decomposes an `f64` input stream into a `(primary, secondary)` pair of
/// `i64` children related by a fixed `base: f64`:
///
/// - `primary[i] = round(x[i] / base) as i64`
/// - `secondary[i] = (x[i].to_bits() as i64).wrapping_sub((base * primary[i] as f64).to_bits() as i64)`
///
/// Decode reconstructs `out[i] = f64::from_bits(((base * primary[i] as f64).to_bits() as i64).wrapping_add(secondary[i]) as u64)`.
/// The round-trip is bit-exact for every `f64` including NaN and the
/// infinities. For data with a true decimal scale, `primary` has small
/// magnitude and `secondary` is tiny (often zero); both compress well in
/// downstream entropy-coded layers.
pub type FloatMultArray = Array<FloatMult>;

/// Slot holding the integer multiplier (high-order part) child.
pub(crate) const PRIMARY_SLOT: usize = 0;
/// Slot holding the signed ULP-offset (low-order part) child.
pub(crate) const SECONDARY_SLOT: usize = 1;
const NUM_SLOTS: usize = 2;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["primary", "secondary"];

/// Marker type implementing [`VTable`] for the [`FloatMult`] mode array.
///
/// FloatMult is parameter-less in the type system; the only state it carries
/// at construction time is the `base` stored in [`FloatMultMetadata`].
#[derive(Clone, Debug)]
pub struct FloatMult;

/// Per-array data for [`FloatMultArray`]. Carries only the `base`.
#[derive(Clone, Debug)]
pub struct FloatMultData {
    base: f64,
}

impl FloatMultData {
    /// Create new FloatMult data from a validated base.
    pub(crate) fn new(base: f64) -> Self {
        Self { base }
    }

    /// Returns the multiplier used to relate primary and secondary.
    pub fn base(&self) -> f64 {
        self.base
    }
}

impl Display for FloatMultData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "base: {}", self.base)
    }
}

impl ArrayHash for FloatMultData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.base.to_bits().hash(state);
    }
}

impl ArrayEq for FloatMultData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.base.to_bits() == other.base.to_bits()
    }
}

impl VTable for FloatMult {
    type TypedArrayData = FloatMultData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.float_mult");
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
            .vortex_expect("FloatMultArray primary slot");
        let secondary = slots[SECONDARY_SLOT]
            .as_ref()
            .vortex_expect("FloatMultArray secondary slot");
        validate_children(data.base, dtype, primary, secondary, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("FloatMultArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("FloatMultArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            FloatMultMetadata {
                base: array.data().base,
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
        let metadata = FloatMultMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }
        ensure_f64_dtype(dtype)?;
        ensure_base_valid(metadata.base)?;

        let child_dtype = DType::Primitive(PType::I64, dtype.nullability());
        let primary = children.get(PRIMARY_SLOT, &child_dtype, len)?;
        let secondary = children.get(SECONDARY_SLOT, &child_dtype, len)?;
        let slots = smallvec![Some(primary.clone()), Some(secondary.clone())];
        let data = FloatMultData::new(metadata.base);
        validate_children(metadata.base, dtype, &primary, &secondary, len)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let base = array.data().base;
        let primary = array.primary().clone().execute::<PrimitiveArray>(ctx)?;
        let secondary = array.secondary().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(ExecutionResult::done(
            decode_primitive(primary, secondary, base).into_array(),
        ))
    }
}

/// Ensure the parent dtype is `f64` (the only width FloatMult supports here).
fn ensure_f64_dtype(dtype: &DType) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    if ptype != PType::F64 {
        vortex_bail!("FloatMultArray only supports f64 inputs, got {ptype}");
    }
    Ok(())
}

/// Ensure `base` is finite, strictly positive, and not subnormal.
fn ensure_base_valid(base: f64) -> VortexResult<()> {
    if base.is_nan() {
        vortex_bail!("FloatMult base must not be NaN");
    }
    if !base.is_finite() {
        vortex_bail!("FloatMult base must be finite, got {base}");
    }
    if base <= 0.0 {
        vortex_bail!("FloatMult base must be > 0, got {base}");
    }
    if !base.is_normal() {
        vortex_bail!("FloatMult base must be a normal f64, got subnormal {base}");
    }
    Ok(())
}

/// Validate that `primary` and `secondary` children are `i64` of the same
/// nullability and length as the array, and that `base` is well-formed.
fn validate_children(
    base: f64,
    dtype: &DType,
    primary: &ArrayRef,
    secondary: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    ensure_f64_dtype(dtype)?;
    ensure_base_valid(base)?;
    let child_dtype = DType::Primitive(PType::I64, dtype.nullability());
    vortex_ensure!(
        primary.dtype() == &child_dtype,
        "FloatMultArray primary dtype {} does not match expected {}",
        primary.dtype(),
        child_dtype,
    );
    vortex_ensure!(
        secondary.dtype() == &child_dtype,
        "FloatMultArray secondary dtype {} does not match expected {}",
        secondary.dtype(),
        child_dtype,
    );
    vortex_ensure!(
        primary.len() == len,
        "FloatMultArray primary len {} does not match array len {len}",
        primary.len(),
    );
    vortex_ensure!(
        secondary.len() == len,
        "FloatMultArray secondary len {} does not match array len {len}",
        secondary.len(),
    );
    Ok(())
}

/// Extension methods on any typed reference to a [`FloatMultArray`].
pub trait FloatMultArrayExt: TypedArrayRef<FloatMult> {
    /// The multiplier relating `primary` and `secondary`.
    fn base(&self) -> f64 {
        // `TypedArrayRef` derefs to `FloatMultData`.
        FloatMultData::base(self)
    }

    /// The integer multiplier child (i64) where `primary[i] ≈ round(x[i] / base)`.
    fn primary(&self) -> &ArrayRef {
        self.as_ref().slots()[PRIMARY_SLOT]
            .as_ref()
            .vortex_expect("FloatMultArray primary slot")
    }

    /// The signed ULP-offset child (i64) carrying the residual bit pattern.
    fn secondary(&self) -> &ArrayRef {
        self.as_ref().slots()[SECONDARY_SLOT]
            .as_ref()
            .vortex_expect("FloatMultArray secondary slot")
    }
}

impl<T: TypedArrayRef<FloatMult>> FloatMultArrayExt for T {}

impl FloatMult {
    /// Construct a [`FloatMultArray`] from validated `primary` and `secondary`
    /// `i64` children. The returned array's logical dtype is `f64` with
    /// nullability inherited from the children.
    ///
    /// Validity flows from the `primary` child via [`ValidityVTableFromChild`].
    pub fn try_new(
        primary: ArrayRef,
        secondary: ArrayRef,
        base: f64,
    ) -> VortexResult<FloatMultArray> {
        let dtype = DType::Primitive(PType::F64, primary.dtype().nullability());
        let len = primary.len();
        validate_children(base, &dtype, &primary, &secondary, len)?;
        let slots = smallvec![Some(primary), Some(secondary)];
        let data = FloatMultData::new(base);
        // SAFETY: validate_children above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(FloatMult, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode an `f64` primitive array using pco's FloatMult mode.
    ///
    /// `base` must be finite, strictly positive, and not subnormal. The
    /// decomposition is bit-exact: `decode(encode(x)) == x` element-by-element
    /// for every `f64` including NaN and infinities. (For NaN/inf inputs,
    /// `primary` is `0` and `secondary` stores the full bit pattern.)
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        base: f64,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<FloatMultArray> {
        let ptype = PrimitiveArrayExt::ptype(&parray);
        if ptype != PType::F64 {
            vortex_bail!("FloatMultArray::encode requires f64 input, got {ptype}");
        }
        ensure_base_valid(base)?;

        let parray = parray.into_owned();
        let validity = PrimitiveArrayExt::validity(&parray);
        let (primary, secondary) = split_buffer(parray.into_buffer::<f64>(), base);

        Self::try_new(
            PrimitiveArray::new(primary, validity.clone()).into_array(),
            PrimitiveArray::new(secondary, validity).into_array(),
            base,
        )
    }
}

/// Split an `f64` buffer into `(primary: i64, secondary: i64)` according to
/// the FloatMult decomposition.
fn split_buffer(values: Buffer<f64>, base: f64) -> (Buffer<i64>, Buffer<i64>) {
    let slice = values.as_slice();
    let len = slice.len();
    let mut primary = BufferMut::<i64>::with_capacity(len);
    let mut secondary = BufferMut::<i64>::with_capacity(len);
    for &x in slice {
        let (prim, sec) = encode_one(x, base);
        primary.push(prim);
        secondary.push(sec);
    }
    (primary.freeze(), secondary.freeze())
}

/// Decompose a single `f64` `x` into `(primary, secondary)` for `base`.
///
/// `primary = round(x / base) as i64` for finite `x` within range; for
/// NaN/inf inputs (and quotients that overflow i64) `primary = 0`.
/// `secondary` is the signed ULP offset that makes the reconstruction
/// bit-exact.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "intentional saturation; secondary compensates losslessly"
)]
fn encode_one(x: f64, base: f64) -> (i64, i64) {
    let q = x / base;
    // For NaN/inf inputs, `q` is non-finite and `q.round() as i64` would
    // saturate. Pin those to primary=0 so secondary carries the entire bit
    // pattern (since approx_bits = (base * 0.0).to_bits() = 0).
    let primary = if q.is_finite() { q.round() as i64 } else { 0 };
    let approx_bits = (base * primary as f64).to_bits() as i64;
    let secondary = (x.to_bits() as i64).wrapping_sub(approx_bits);
    (primary, secondary)
}

/// Reconstruct a single `f64` from a `(primary, secondary)` pair and `base`.
#[inline]
fn decode_one(primary: i64, secondary: i64, base: f64) -> f64 {
    let approx_bits = (base * primary as f64).to_bits() as i64;
    f64::from_bits(approx_bits.wrapping_add(secondary) as u64)
}

/// Recompose an `f64` `PrimitiveArray` from the two `i64` children. Validity
/// is taken from `primary`.
fn decode_primitive(
    primary: PrimitiveArray,
    secondary: PrimitiveArray,
    base: f64,
) -> PrimitiveArray {
    let validity = PrimitiveArrayExt::validity(&primary);
    let p_buf = primary.into_buffer::<i64>();
    let s_buf = secondary.into_buffer::<i64>();
    let p_slice = p_buf.as_slice();
    let s_slice = s_buf.as_slice();
    debug_assert_eq!(p_slice.len(), s_slice.len());
    let mut out = BufferMut::<f64>::with_capacity(p_slice.len());
    for (p, s) in p_slice.iter().zip(s_slice.iter()) {
        out.push(decode_one(*p, *s, base));
    }
    PrimitiveArray::new(out.freeze(), validity)
}

impl OperationsVTable<FloatMult> for FloatMult {
    fn scalar_at(
        array: ArrayView<'_, FloatMult>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let base = array.data().base;
        let nullability = array.dtype().nullability();
        let primary_scalar = array.primary().execute_scalar(index, ctx)?;
        if primary_scalar.is_null() {
            return Scalar::try_new(array.dtype().clone(), None);
        }
        let secondary_scalar = array.secondary().execute_scalar(index, ctx)?;
        let p = primary_scalar
            .as_primitive()
            .typed_value::<i64>()
            .vortex_expect("FloatMult primary scalar must be i64");
        let s = secondary_scalar
            .as_primitive()
            .typed_value::<i64>()
            .vortex_expect("FloatMult secondary scalar must be i64");
        Ok(Scalar::primitive(decode_one(p, s, base), nullability))
    }
}

impl ValidityChild<FloatMult> for FloatMult {
    fn validity_child(array: ArrayView<'_, FloatMult>) -> ArrayRef {
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
    fn round_trip_bits(input: &[f64], base: f64) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(input.iter().copied());
        let encoded = FloatMult::encode(parray.as_view(), base, &mut ctx)?;
        assert_eq!(encoded.dtype(), parray.dtype());
        assert_eq!(encoded.len(), parray.len());
        assert_eq!(encoded.base().to_bits(), base.to_bits());
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

    /// FloatMult-favorable input: `x[i] = base * k_i + tiny_ulp_offset_i`.
    #[test]
    fn round_trip_favorable() -> VortexResult<()> {
        let base = 0.01_f64;
        let ks: [i64; 8] = [0, 1, -3, 17, -200, 12_345, -1_000_000, 1_000_000];
        let ulp_offsets: [i64; 8] = [0, 1, -1, 2, -5, 7, -3, 0];
        let values: Vec<f64> = ks
            .iter()
            .zip(ulp_offsets.iter())
            .map(|(k, off)| {
                let approx = base * (*k as f64);
                let bits = (approx.to_bits() as i64).wrapping_add(*off) as u64;
                f64::from_bits(bits)
            })
            .collect();
        round_trip_bits(&values, base)?;

        // The favorable construction means primary should match `ks` exactly.
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = FloatMult::encode(parray.as_view(), base, &mut ctx)?;
        let primary = encoded
            .primary()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<i64>();
        assert_eq!(primary.as_slice(), &ks[..]);
        Ok(())
    }

    /// Random `f64` inputs round-trip bit-for-bit even when primary is large.
    #[test]
    fn round_trip_arbitrary_random() -> VortexResult<()> {
        let mut rng = SmallRng::seed_from_u64(0xBEEF);
        let values: Vec<f64> = (0..1024)
            .map(|_| rng.random_range(-1e6_f64..1e6_f64))
            .collect();
        round_trip_bits(&values, 0.01)
    }

    /// Special values (NaN, +/-inf, +/-0) round-trip bit-for-bit.
    #[test]
    fn round_trip_special_values() -> VortexResult<()> {
        let values = [
            1.0_f64,
            -2.5,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
        ];
        round_trip_bits(&values, 0.01)
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base = 0.01_f64;
        let values: Vec<f64> = (0..50).map(|i| base * i as f64).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = FloatMult::encode(parray.as_view(), base, &mut ctx)?;
        let sliced = encoded.into_array().slice(10..30)?;
        let expected = PrimitiveArray::from_iter(values[10..30].iter().copied());
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn nullable_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base = 0.01_f64;
        let input = PrimitiveArray::new(
            buffer![0.0_f64, 0.07, 0.14, 0.21, 0.28, 0.35],
            Validity::from_iter([true, false, true, false, true, true]),
        );
        let encoded = FloatMult::encode(input.as_view(), base, &mut ctx)?;

        let s1 = encoded.clone().into_array().execute_scalar(1, &mut ctx)?;
        let s3 = encoded.clone().into_array().execute_scalar(3, &mut ctx)?;
        assert!(s1.is_null());
        assert!(s3.is_null());

        let s0 = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(s0.as_primitive().typed_value::<f64>(), Some(0.0));
        let s2 = encoded.clone().into_array().execute_scalar(2, &mut ctx)?;
        // 0.14 round-trips bit-exactly.
        assert_eq!(
            s2.as_primitive().typed_value::<f64>().map(f64::to_bits),
            Some(0.14_f64.to_bits())
        );

        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input);
        Ok(())
    }

    #[rstest]
    #[case::nan(f64::NAN)]
    #[case::pos_inf(f64::INFINITY)]
    #[case::neg_inf(f64::NEG_INFINITY)]
    #[case::zero(0.0_f64)]
    #[case::negative(-1.0_f64)]
    #[case::subnormal(f64::from_bits(1))]
    fn rejects_invalid_base(#[case] base: f64) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([1.0_f64, 2.0, 3.0]);
        let err = FloatMult::encode(parray.as_view(), base, &mut ctx);
        assert!(
            err.is_err(),
            "expected error for base {base:?}, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn rejects_non_f64_input() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([1.0_f32, 2.0, 3.0]);
        let err = FloatMult::encode(parray.as_view(), 0.01, &mut ctx);
        assert!(err.is_err(), "expected error for f32 input, got {err:?}");

        let parray = PrimitiveArray::from_iter([1i64, 2, 3]);
        let err = FloatMult::encode(parray.as_view(), 0.01, &mut ctx);
        assert!(err.is_err(), "expected error for i64 input, got {err:?}");
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base = 0.01_f64;
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let values: Vec<f64> = (0..256)
            .map(|_| rng.random_range(-1e6_f64..1e6_f64))
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = FloatMult::encode(parray.as_view(), base, &mut ctx)?;
        let arr = encoded.into_array();

        let decoded = arr
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<f64>();

        let indices = [0usize, 1, 7, 42, 100, 128, 200, 255];
        for &i in &indices {
            let scalar = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(scalar, Scalar::from(decoded.as_slice()[i]));
        }
        Ok(())
    }
}
