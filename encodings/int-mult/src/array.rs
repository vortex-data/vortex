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
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
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

use crate::IntMultMetadata;

/// An [`IntMult`]-encoded Vortex array.
///
/// Decomposes an unsigned latent stream of type `L ∈ {u8, u16, u32, u64}`
/// into a `(primary, secondary)` pair such that
/// `value = base.wrapping_mul(primary) + secondary` in the latent type.
/// This is the layered equivalent of pco's `IntMult` mode and shaves
/// `log2(base)` bits off the magnitude of typical primary values when the
/// data has a true integer scale (e.g. millicents, microseconds).
pub type IntMultArray = Array<IntMult>;

/// Slot holding the multiplier (high-order part) child.
pub(crate) const PRIMARY_SLOT: usize = 0;
/// Slot holding the adjustment (low-order part) child.
pub(crate) const SECONDARY_SLOT: usize = 1;
const NUM_SLOTS: usize = 2;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["primary", "secondary"];

/// Marker type implementing [`VTable`] for the [`IntMult`] mode array.
///
/// IntMult is parameter-less in the type system; the only state it carries
/// at construction time is the `base` stored in [`IntMultMetadata`].
#[derive(Clone, Debug)]
pub struct IntMult;

/// Per-array data for [`IntMultArray`]. Carries only the `base`.
#[derive(Clone, Debug)]
pub struct IntMultData {
    base: u64,
}

impl IntMultData {
    /// Create new IntMult data from a validated base.
    pub(crate) fn new(base: u64) -> Self {
        Self { base }
    }

    /// Returns the multiplier used to relate primary and secondary.
    pub fn base(&self) -> u64 {
        self.base
    }
}

impl Display for IntMultData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "base: {}", self.base)
    }
}

impl ArrayHash for IntMultData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _precision: Precision) {
        self.base.hash(state);
    }
}

impl ArrayEq for IntMultData {
    fn array_eq(&self, other: &Self, _precision: Precision) -> bool {
        self.base == other.base
    }
}

impl VTable for IntMult {
    type TypedArrayData = IntMultData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.int_mult");
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
            .vortex_expect("IntMultArray primary slot");
        let secondary = slots[SECONDARY_SLOT]
            .as_ref()
            .vortex_expect("IntMultArray secondary slot");
        validate_children(data.base, dtype, primary, secondary, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("IntMultArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("IntMultArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            IntMultMetadata {
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
        let metadata = IntMultMetadata::decode(metadata)?;
        if children.len() != NUM_SLOTS {
            vortex_bail!("Expected {NUM_SLOTS} children, got {}", children.len());
        }

        let ptype = PType::try_from(dtype)?;
        ensure_unsigned_latent(ptype)?;
        ensure_base_fits(metadata.base, ptype)?;

        let child_dtype = dtype.clone();
        let primary = children.get(PRIMARY_SLOT, &child_dtype, len)?;
        let secondary = children.get(SECONDARY_SLOT, &child_dtype, len)?;
        let slots = smallvec![Some(primary.clone()), Some(secondary.clone())];
        let data = IntMultData::new(metadata.base);
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
            decode_primitive(primary, secondary, base)?.into_array(),
        ))
    }
}

/// Ensure the dtype is one of `u8`/`u16`/`u32`/`u64`.
fn ensure_unsigned_latent(ptype: PType) -> VortexResult<()> {
    if !ptype.is_unsigned_int() {
        vortex_bail!(
            "IntMultArray latent must be an unsigned integer, got {}",
            ptype
        );
    }
    Ok(())
}

/// Ensure `base >= 2` and that it fits in the given latent ptype.
fn ensure_base_fits(base: u64, ptype: PType) -> VortexResult<()> {
    if base < 2 {
        vortex_bail!("IntMult base must be >= 2, got {base}");
    }
    let max = match ptype {
        PType::U8 => u8::MAX as u64,
        PType::U16 => u16::MAX as u64,
        PType::U32 => u32::MAX as u64,
        PType::U64 => u64::MAX,
        _ => vortex_bail!(
            "IntMultArray latent must be an unsigned integer, got {}",
            ptype
        ),
    };
    if base > max {
        vortex_bail!("IntMult base {base} does not fit in latent ptype {ptype} (max {max})");
    }
    Ok(())
}

/// Validate that `primary` and `secondary` children share the array's dtype
/// and length, and that the dtype is one of `u8`/`u16`/`u32`/`u64`.
fn validate_children(
    base: u64,
    dtype: &DType,
    primary: &ArrayRef,
    secondary: &ArrayRef,
    len: usize,
) -> VortexResult<()> {
    let ptype = PType::try_from(dtype)?;
    ensure_unsigned_latent(ptype)?;
    ensure_base_fits(base, ptype)?;
    vortex_ensure!(
        primary.dtype() == dtype,
        "IntMultArray primary dtype {} does not match array dtype {}",
        primary.dtype(),
        dtype,
    );
    vortex_ensure!(
        secondary.dtype() == dtype,
        "IntMultArray secondary dtype {} does not match array dtype {}",
        secondary.dtype(),
        dtype,
    );
    vortex_ensure!(
        primary.len() == len,
        "IntMultArray primary len {} does not match array len {len}",
        primary.len(),
    );
    vortex_ensure!(
        secondary.len() == len,
        "IntMultArray secondary len {} does not match array len {len}",
        secondary.len(),
    );
    Ok(())
}

/// Extension methods on any typed reference to an [`IntMultArray`].
pub trait IntMultArrayExt: TypedArrayRef<IntMult> {
    /// The multiplier relating `primary` and `secondary`.
    fn base(&self) -> u64 {
        // `TypedArrayRef` derefs to `IntMultData`.
        IntMultData::base(self)
    }

    /// The high-order child array of multipliers.
    fn primary(&self) -> &ArrayRef {
        self.as_ref().slots()[PRIMARY_SLOT]
            .as_ref()
            .vortex_expect("IntMultArray primary slot")
    }

    /// The low-order child array of adjustments.
    fn secondary(&self) -> &ArrayRef {
        self.as_ref().slots()[SECONDARY_SLOT]
            .as_ref()
            .vortex_expect("IntMultArray secondary slot")
    }

    /// The latent [`PType`] of the array (always unsigned).
    fn ptype(&self) -> PType {
        PType::try_from(self.as_ref().dtype()).vortex_expect("IntMultArray dtype")
    }
}

impl<T: TypedArrayRef<IntMult>> IntMultArrayExt for T {}

impl IntMult {
    /// Construct an [`IntMultArray`] from validated `primary` and `secondary`
    /// children. Both children must share an unsigned-integer dtype.
    ///
    /// The returned array's dtype is the shared child dtype; validity flows
    /// from the `primary` child via [`ValidityVTableFromChild`].
    pub fn try_new(
        primary: ArrayRef,
        secondary: ArrayRef,
        base: u64,
    ) -> VortexResult<IntMultArray> {
        let dtype = primary.dtype().clone();
        let len = primary.len();
        validate_children(base, &dtype, &primary, &secondary, len)?;
        let slots = smallvec![Some(primary), Some(secondary)];
        let data = IntMultData::new(base);
        // SAFETY: validate_children above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(IntMult, dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode an ordered-latent primitive array using pco's IntMult mode.
    ///
    /// Splits each value `n` into `(n / base, n % base)`. The input dtype
    /// must be one of `u8`/`u16`/`u32`/`u64`; floats and signed integers
    /// should first be passed through `OrderedLatentArray::encode` and then
    /// unwrapped into a `Primitive<L>`.
    ///
    /// `base` must be `>= 2` and must fit in the latent type.
    pub fn encode(
        latent: ArrayView<'_, Primitive>,
        base: u64,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<IntMultArray> {
        let ptype = PrimitiveArrayExt::ptype(&latent);
        ensure_unsigned_latent(ptype)?;
        ensure_base_fits(base, ptype)?;

        let parray = latent.into_owned();
        let validity = PrimitiveArrayExt::validity(&parray);

        let (primary, secondary) = match_each_unsigned_integer_ptype!(ptype, |L| {
            // Safe: `ensure_base_fits` above checked that `base <= L::MAX`.
            let base_l = L::try_from(base).vortex_expect("base fits in latent ptype");
            let (p_buf, s_buf) = split_buffer::<L>(parray.into_buffer::<L>(), base_l);
            (
                PrimitiveArray::new(p_buf, validity.clone()).into_array(),
                PrimitiveArray::new(s_buf, validity).into_array(),
            )
        });

        Self::try_new(primary, secondary, base)
    }
}

/// Split a buffer of latent values into `(primary, secondary)` buffers
/// where `primary[i] = values[i] / base` and `secondary[i] = values[i] % base`.
fn split_buffer<L>(values: Buffer<L>, base: L) -> (Buffer<L>, Buffer<L>)
where
    L: NativePType + std::ops::Div<Output = L> + std::ops::Rem<Output = L>,
{
    let slice = values.as_slice();
    let n = slice.len();
    let mut primary = BufferMut::<L>::with_capacity(n);
    let mut secondary = BufferMut::<L>::with_capacity(n);
    for v in slice {
        primary.push(*v / base);
        secondary.push(*v % base);
    }
    (primary.freeze(), secondary.freeze())
}

/// Compose `out[i] = base.wrapping_mul(primary[i]) + secondary[i]` in
/// wrapping arithmetic. Validity is taken from `primary`.
fn decode_primitive(
    primary: PrimitiveArray,
    secondary: PrimitiveArray,
    base: u64,
) -> VortexResult<PrimitiveArray> {
    let ptype = primary.ptype();
    let validity = PrimitiveArrayExt::validity(&primary);
    Ok(match_each_unsigned_integer_ptype!(ptype, |L| {
        // Safe: validate_children above checked that `base <= L::MAX`.
        let base_l = L::try_from(base).vortex_expect("base fits in latent ptype");
        let buffer = compose_buffer::<L>(
            primary.into_buffer_mut::<L>(),
            secondary.into_buffer::<L>(),
            base_l,
        );
        PrimitiveArray::new(buffer, validity)
    }))
}

fn compose_buffer<L>(mut primary: BufferMut<L>, secondary: Buffer<L>, base: L) -> Buffer<L>
where
    L: NativePType + WrappingMulAdd,
{
    let s = secondary.as_slice();
    debug_assert_eq!(primary.len(), s.len());
    for (dst, src) in primary.as_mut_slice().iter_mut().zip(s) {
        *dst = L::wrapping_mul_add(base, *dst, *src);
    }
    primary.freeze()
}

/// Trait providing wrapping `base * primary + secondary` for the four
/// unsigned latent widths supported by IntMult.
trait WrappingMulAdd: Copy {
    fn wrapping_mul_add(base: Self, primary: Self, secondary: Self) -> Self;
}

macro_rules! impl_wrapping_mul_add {
    ($T:ty) => {
        impl WrappingMulAdd for $T {
            #[inline]
            fn wrapping_mul_add(base: Self, primary: Self, secondary: Self) -> Self {
                base.wrapping_mul(primary).wrapping_add(secondary)
            }
        }
    };
}

impl_wrapping_mul_add!(u8);
impl_wrapping_mul_add!(u16);
impl_wrapping_mul_add!(u32);
impl_wrapping_mul_add!(u64);

impl OperationsVTable<IntMult> for IntMult {
    fn scalar_at(
        array: ArrayView<'_, IntMult>,
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

        let pscalar = primary_scalar.as_primitive();
        let sscalar = secondary_scalar.as_primitive();
        Ok(match_each_unsigned_integer_ptype!(pscalar.ptype(), |L| {
            let p = pscalar
                .typed_value::<L>()
                .vortex_expect("IntMult primary scalar must match latent ptype");
            let s = sscalar
                .typed_value::<L>()
                .vortex_expect("IntMult secondary scalar must match latent ptype");
            // Safe: validate_children at construction ensured `base <= L::MAX`.
            let base_l = L::try_from(base).vortex_expect("base fits in latent ptype");
            Scalar::primitive(L::wrapping_mul_add(base_l, p, s), nullability)
        }))
    }
}

impl ValidityChild<IntMult> for IntMult {
    fn validity_child(array: ArrayView<'_, IntMult>) -> ArrayRef {
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
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    /// Encode + decode round-trip an unsigned latent array, checking the
    /// decode matches the input element-wise.
    fn round_trip<L>(input: Vec<L>, base: u64) -> VortexResult<()>
    where
        L: NativePType + std::fmt::Debug,
    {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter(input.iter().copied());
        let encoded = IntMult::encode(parray.as_view(), base, &mut ctx)?;
        assert_eq!(encoded.dtype(), parray.dtype());
        assert_eq!(encoded.len(), parray.len());
        assert_eq!(encoded.base(), base);
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, parray);
        Ok(())
    }

    #[rstest]
    #[case::u8(vec![0u8, 1, 2, 7, 14, 100, 250], 7)]
    #[case::u16(vec![0u16, 1, 7, 14, 1000, 30_000, u16::MAX], 7)]
    #[case::u32(vec![0u32, 1, 7, 14, 1_000_000, u32::MAX], 7)]
    #[case::u64(vec![0u64, 1, 7, 14, 1_000_000_000, u64::MAX], 7)]
    fn round_trip_each_latent<L>(#[case] input: Vec<L>, #[case] base: u64) -> VortexResult<()>
    where
        L: NativePType + std::fmt::Debug,
    {
        round_trip(input, base)
    }

    /// Synthetic IntMult-favorable input: `latent[i] = base * k_i + r_i`.
    /// Encoded primary and secondary children must be exactly the `k_i` and
    /// `r_i` we put in.
    #[test]
    fn split_matches_known_layout() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base: u32 = 1000;
        let ks: [u32; 8] = [0, 1, 2, 5, 17, 999, 12_345, 100_000];
        let rs: [u32; 8] = [0, 1, 2, 3, 999, 500, 7, 42];
        let values: Vec<u32> = ks
            .iter()
            .zip(rs.iter())
            .map(|(k, r)| base * k + r)
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = IntMult::encode(parray.as_view(), base as u64, &mut ctx)?;

        let primary = encoded
            .primary()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u32>();
        let secondary = encoded
            .secondary()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?
            .into_buffer::<u32>();
        assert_eq!(primary.as_slice(), &ks[..]);
        assert_eq!(secondary.as_slice(), &rs[..]);
        Ok(())
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base: u32 = 100;
        let values: Vec<u32> = (0u32..50).map(|i| base * i + (i % base)).collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = IntMult::encode(parray.as_view(), base as u64, &mut ctx)?;
        let sliced = encoded.into_array().slice(10..30)?;
        let expected = PrimitiveArray::from_iter(values[10..30].iter().copied());
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn nullable_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base: u64 = 1000;
        let input = PrimitiveArray::new(
            buffer![0u64, 7000, 14_001, 21_002, 28_003, 35_004],
            Validity::from_iter([true, false, true, false, true, true]),
        );
        let encoded = IntMult::encode(input.as_view(), base, &mut ctx)?;

        // Null positions retained.
        let s1 = encoded.clone().into_array().execute_scalar(1, &mut ctx)?;
        let s3 = encoded.clone().into_array().execute_scalar(3, &mut ctx)?;
        assert!(s1.is_null());
        assert!(s3.is_null());

        // Non-null positions decode correctly.
        let s0 = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(s0.as_primitive().typed_value::<u64>(), Some(0));
        let s2 = encoded.clone().into_array().execute_scalar(2, &mut ctx)?;
        assert_eq!(s2.as_primitive().typed_value::<u64>(), Some(14_001));

        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input);
        Ok(())
    }

    #[rstest]
    #[case::zero(0u64)]
    #[case::one(1u64)]
    fn rejects_small_base(#[case] base: u64) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([0u32, 1, 2]);
        let err = IntMult::encode(parray.as_view(), base, &mut ctx);
        assert!(err.is_err(), "expected error for base {base}, got {err:?}");
        Ok(())
    }

    #[test]
    fn rejects_base_exceeding_latent_max() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        // Latent is u8, but base of 300 does not fit in u8.
        let parray = PrimitiveArray::from_iter([0u8, 1, 2]);
        let err = IntMult::encode(parray.as_view(), 300u64, &mut ctx);
        assert!(err.is_err(), "expected error for u8 base 300, got {err:?}");

        // u16 base that overflows.
        let parray = PrimitiveArray::from_iter([0u16, 1, 2]);
        let err = IntMult::encode(parray.as_view(), 100_000u64, &mut ctx);
        assert!(
            err.is_err(),
            "expected error for u16 base 100_000, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn rejects_signed_latent() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let parray = PrimitiveArray::from_iter([1i32, 2, 3]);
        let err = IntMult::encode(parray.as_view(), 7, &mut ctx);
        assert!(
            err.is_err(),
            "expected error for signed latent, got {err:?}"
        );
        Ok(())
    }

    #[test]
    fn scalar_at_matches_canonical_decode() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let base: u64 = 1000;
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let values: Vec<u64> = (0..256)
            .map(|_| rng.random_range(0u64..1_000_000) * base + rng.random_range(0u64..base))
            .collect();
        let parray = PrimitiveArray::from_iter(values.iter().copied());
        let encoded = IntMult::encode(parray.as_view(), base, &mut ctx)?;
        let arr = encoded.into_array();

        // Spot-check at a handful of indices.
        let indices = [0usize, 1, 7, 42, 100, 128, 200, 255];
        for &i in &indices {
            let scalar = arr.execute_scalar(i, &mut ctx)?;
            assert_eq!(scalar, Scalar::from(values[i]));
        }
        Ok(())
    }
}
