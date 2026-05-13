// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;

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
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_native_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::transforms::OrderedLatentNumber;

/// An [`OrderedLatent`]-encoded Vortex array.
///
/// Stores its values as an unsigned latent that preserves order with respect
/// to the parent's primitive `T`. The parent dtype is preserved, so a slice or
/// canonicalize round-trips back to the original `Primitive<T>` values.
pub type OrderedLatentArray = Array<OrderedLatent>;

/// Slot holding the unsigned latent child array.
const ENCODED_SLOT: usize = 0;
const NUM_SLOTS: usize = 1;
const SLOT_NAMES: [&str; NUM_SLOTS] = ["encoded"];

/// Marker type implementing [`VTable`] for the order-preserving latent encoding.
///
/// The encoding maps every pco-supported primitive `T` (any of `u8`/`u16`/`u32`/
/// `u64`/`i8`/`i16`/`i32`/`i64`/`f16`/`f32`/`f64`) to an unsigned latent `L` of
/// the same byte width. The mapping is bijective and preserves order with
/// respect to IEEE-754 total ordering for floats.
#[derive(Clone, Debug)]
pub struct OrderedLatent;

/// Per-array data for [`OrderedLatentArray`]. The encoding is stateless beyond
/// the dtype carried by [`Array`] itself, so this struct holds no fields.
#[derive(Clone, Debug, Default)]
pub struct OrderedLatentData;

impl Display for OrderedLatentData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArrayHash for OrderedLatentData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _precision: Precision) {}
}

impl ArrayEq for OrderedLatentData {
    fn array_eq(&self, _other: &Self, _precision: Precision) -> bool {
        true
    }
}

impl OrderedLatentData {
    /// Construct a fresh, empty data instance.
    pub fn new() -> Self {
        Self
    }

    /// Compute the parent dtype implied by an encoded child dtype.
    ///
    /// Because the parent → latent mapping is ambiguous on the unsigned-integer
    /// case (`u8` → `u8`, `i8` → `u8`, etc.), this helper is used only when the
    /// parent dtype is already known and we need to validate it against the
    /// encoded dtype's width.
    fn validate_widths(parent_dtype: &DType, encoded_dtype: &DType) -> VortexResult<()> {
        let parent_ptype = PType::try_from(parent_dtype)?;
        let encoded_ptype = PType::try_from(encoded_dtype)?;
        if !encoded_ptype.is_unsigned_int() {
            vortex_bail!(MismatchedTypes: "unsigned int", encoded_dtype);
        }
        vortex_ensure!(
            parent_ptype.byte_width() == encoded_ptype.byte_width(),
            "OrderedLatentArray: parent ptype {parent_ptype} byte width does not match \
             encoded ptype {encoded_ptype} byte width"
        );
        vortex_ensure!(
            encoded_ptype == latent_ptype(parent_ptype),
            "OrderedLatentArray: expected latent {} for parent {}, got {}",
            latent_ptype(parent_ptype),
            parent_ptype,
            encoded_ptype,
        );
        vortex_ensure!(
            parent_dtype.nullability() == encoded_dtype.nullability(),
            "OrderedLatentArray: parent nullability {} does not match encoded nullability {}",
            parent_dtype.nullability(),
            encoded_dtype.nullability(),
        );
        Ok(())
    }
}

/// Returns the latent unsigned [`PType`] corresponding to a parent [`PType`].
#[inline]
fn latent_ptype(parent: PType) -> PType {
    match parent {
        PType::U8 | PType::I8 => PType::U8,
        PType::U16 | PType::I16 | PType::F16 => PType::U16,
        PType::U32 | PType::I32 | PType::F32 => PType::U32,
        PType::U64 | PType::I64 | PType::F64 => PType::U64,
    }
}

impl VTable for OrderedLatent {
    type TypedArrayData = OrderedLatentData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.ordered_latent");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let encoded = slots[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("OrderedLatentArray encoded slot");
        OrderedLatentData::validate_widths(dtype, encoded.dtype())?;
        vortex_ensure!(
            encoded.len() == len,
            "expected len {len}, got {}",
            encoded.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("OrderedLatentArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("OrderedLatentArray buffer_name index {idx} out of bounds")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
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
        if !metadata.is_empty() {
            vortex_bail!(
                "OrderedLatentArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }

        let parent_ptype = PType::try_from(dtype)?;
        let encoded_type = DType::Primitive(latent_ptype(parent_ptype), dtype.nullability());
        let encoded = children.get(0, &encoded_type, len)?;
        let slots = smallvec![Some(encoded.clone())];
        let data = OrderedLatentData::new();
        OrderedLatentData::validate_widths(dtype, encoded.dtype())?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let parent_dtype = array.dtype().clone();
        let encoded = array.encoded().clone().execute::<PrimitiveArray>(ctx)?;
        Ok(ExecutionResult::done(
            decode_primitive(&encoded, &parent_dtype)?.into_array(),
        ))
    }
}

/// Extension methods on any typed reference to an [`OrderedLatentArray`].
pub trait OrderedLatentArrayExt: TypedArrayRef<OrderedLatent> {
    /// Returns the underlying unsigned latent child array.
    fn encoded(&self) -> &ArrayRef {
        self.as_ref().slots()[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("OrderedLatentArray encoded slot")
    }

    /// Returns the parent [`PType`] this array decodes back to.
    fn ptype(&self) -> PType {
        PType::try_from(self.as_ref().dtype()).vortex_expect("OrderedLatentArray dtype")
    }

    /// Returns the latent (unsigned) [`PType`] used in storage.
    fn latent_ptype(&self) -> PType {
        latent_ptype(self.ptype())
    }
}

impl<T: TypedArrayRef<OrderedLatent>> OrderedLatentArrayExt for T {}

impl OrderedLatent {
    /// Construct an [`OrderedLatentArray`] from an already-encoded unsigned
    /// child array and the original primitive dtype.
    ///
    /// The encoded child must be a `Primitive<L>` where `L` is the unsigned
    /// integer type whose byte width matches `parent_dtype`'s ptype, and whose
    /// nullability matches.
    pub fn try_new(encoded: ArrayRef, parent_dtype: DType) -> VortexResult<OrderedLatentArray> {
        OrderedLatentData::validate_widths(&parent_dtype, encoded.dtype())?;
        let len = encoded.len();
        let slots = smallvec![Some(encoded)];
        let data = OrderedLatentData::new();
        // SAFETY: validate_widths above checked all type/length invariants.
        Ok(unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(OrderedLatent, parent_dtype, len, data).with_slots(slots),
            )
        })
    }

    /// Encode a primitive array into its ordered-latent representation.
    ///
    /// The returned [`OrderedLatentArray`] has the *same logical dtype* as the
    /// input; only the storage is the unsigned latent.
    pub fn encode(
        parray: ArrayView<'_, Primitive>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<OrderedLatentArray> {
        let parent_dtype = parray.dtype().clone();
        let parray = parray.into_owned();
        let validity = PrimitiveArrayExt::validity(&parray);
        let ptype = PrimitiveArrayExt::ptype(&parray);
        let encoded = match_each_native_ptype!(ptype, |T| {
            encode_buffer::<T>(parray.into_buffer_mut::<T>(), validity)
        });
        Self::try_new(encoded.into_array(), parent_dtype)
    }
}

/// Apply [`OrderedLatentNumber::to_latent_ordered`] to every element of a
/// buffer, producing a `Primitive<L>` array with the original validity.
fn encode_buffer<T>(values: BufferMut<T>, validity: Validity) -> PrimitiveArray
where
    T: OrderedLatentNumber,
{
    let encoded = values.map_each_in_place(T::to_latent_ordered);
    PrimitiveArray::new(encoded.freeze(), validity)
}

/// Apply [`OrderedLatentNumber::from_latent_ordered`] to every element of an
/// already-canonical [`PrimitiveArray`], producing a `Primitive<T>` array
/// matching the supplied `parent_dtype`.
fn decode_primitive(
    encoded: &PrimitiveArray,
    parent_dtype: &DType,
) -> VortexResult<PrimitiveArray> {
    let parent_ptype = PType::try_from(parent_dtype)?;
    let validity = PrimitiveArrayExt::validity(encoded);
    let encoded = encoded.clone();
    Ok(match_each_native_ptype!(parent_ptype, |T| {
        decode_buffer::<T>(
            encoded.into_buffer_mut::<<T as OrderedLatentNumber>::Latent>(),
            validity,
        )
    }))
}

fn decode_buffer<T>(values: BufferMut<T::Latent>, validity: Validity) -> PrimitiveArray
where
    T: OrderedLatentNumber,
{
    let decoded = values.map_each_in_place(T::from_latent_ordered);
    PrimitiveArray::new(decoded.freeze(), validity)
}

impl OperationsVTable<OrderedLatent> for OrderedLatent {
    fn scalar_at(
        array: ArrayView<'_, OrderedLatent>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let parent_ptype = OrderedLatentArrayExt::ptype(&array);
        let nullability = array.dtype().nullability();
        let encoded_scalar = array.encoded().execute_scalar(index, ctx)?;
        if encoded_scalar.is_null() {
            return null_scalar(parent_ptype, nullability);
        }

        let pscalar = encoded_scalar.as_primitive();
        Ok(match_each_native_ptype!(parent_ptype, |T| {
            let latent_value = pscalar
                .typed_value::<<T as OrderedLatentNumber>::Latent>()
                .vortex_expect("ordered-latent encoded scalar must match latent ptype");
            Scalar::primitive(T::from_latent_ordered(latent_value), nullability)
        }))
    }
}

/// Construct a typed null scalar for the parent [`PType`].
fn null_scalar(parent_ptype: PType, nullability: Nullability) -> VortexResult<Scalar> {
    Scalar::try_new(DType::Primitive(parent_ptype, nullability), None)
}

impl ValidityChild<OrderedLatent> for OrderedLatent {
    fn validity_child(array: ArrayView<'_, OrderedLatent>) -> ArrayRef {
        array.encoded().clone()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::half::f16;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    fn round_trip(input: PrimitiveArray) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let encoded = OrderedLatent::encode(input.as_view(), &mut ctx)?;
        assert_eq!(encoded.dtype(), input.dtype());
        assert_eq!(encoded.len(), input.len());
        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input);
        Ok(())
    }

    #[rstest]
    #[case::u8(PrimitiveArray::from_iter([0u8, 1, 2, 127, 200, u8::MAX]))]
    #[case::u16(PrimitiveArray::from_iter([0u16, 1, 1000, u16::MAX]))]
    #[case::u32(PrimitiveArray::from_iter([0u32, 1, 1_000_000, u32::MAX]))]
    #[case::u64(PrimitiveArray::from_iter([0u64, 1, 1_000_000_000, u64::MAX]))]
    #[case::i8(PrimitiveArray::from_iter([i8::MIN, -1, 0, 1, i8::MAX]))]
    #[case::i16(PrimitiveArray::from_iter([i16::MIN, -100, 0, 100, i16::MAX]))]
    #[case::i32(PrimitiveArray::from_iter([i32::MIN, -100_000, 0, 100_000, i32::MAX]))]
    #[case::i64(PrimitiveArray::from_iter([i64::MIN, -1_000_000_000, 0, 1_000_000_000, i64::MAX]))]
    #[case::f32(PrimitiveArray::from_iter([
        f32::NEG_INFINITY, -1.5_f32, -0.0, 0.0, 1.5, f32::INFINITY,
    ]))]
    #[case::f64(PrimitiveArray::from_iter([
        f64::NEG_INFINITY, -1.5_f64, -0.0, 0.0, 1.5, f64::INFINITY,
    ]))]
    fn round_trip_each_type(#[case] input: PrimitiveArray) -> VortexResult<()> {
        round_trip(input)
    }

    #[test]
    fn round_trip_f16() -> VortexResult<()> {
        let xs: Vec<f16> = [-1.5_f32, -0.0, 0.0, 1.0, 1.5, 3.0]
            .iter()
            .copied()
            .map(f16::from_f32)
            .collect();
        round_trip(PrimitiveArray::from_iter(xs))
    }

    /// The latent buffer of a sorted input must be monotone non-decreasing.
    #[rstest]
    #[case::sorted_i32(buffer![i32::MIN, -1_000, -1, 0, 1, 1_000, i32::MAX].into_array())]
    #[case::sorted_f64(
        PrimitiveArray::from_iter([
            f64::NEG_INFINITY, -1e10_f64, -1.0, -0.0, 0.0, 1.0, 1e10, f64::INFINITY
        ]).into_array()
    )]
    #[case::sorted_u32(buffer![0u32, 1, 100, 1_000_000, u32::MAX].into_array())]
    fn latent_is_monotone_for_sorted_input(#[case] input: ArrayRef) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let primitive = input.execute::<PrimitiveArray>(&mut ctx)?;
        let encoded = OrderedLatent::encode(primitive.as_view(), &mut ctx)?;
        let latent_array = encoded
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        match latent_array.ptype() {
            PType::U8 => assert_buffer_sorted(&latent_array.into_buffer::<u8>()),
            PType::U16 => assert_buffer_sorted(&latent_array.into_buffer::<u16>()),
            PType::U32 => assert_buffer_sorted(&latent_array.into_buffer::<u32>()),
            PType::U64 => assert_buffer_sorted(&latent_array.into_buffer::<u64>()),
            other => vortex_bail!("unexpected latent ptype {other}"),
        }
        Ok(())
    }

    fn assert_buffer_sorted<T: NativePType + Ord>(buf: &Buffer<T>) {
        let slice = buf.as_slice();
        for w in slice.windows(2) {
            assert!(
                w[0] <= w[1],
                "latent buffer not monotone: {:?} > {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn nullable_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let input = PrimitiveArray::new(
            buffer![-128i32, 0, 42, 99, 100],
            Validity::from_iter([true, false, true, false, true]),
        );
        let encoded = OrderedLatent::encode(input.as_view(), &mut ctx)?;

        // Null positions retained through the encoding boundary.
        let s1 = encoded.clone().into_array().execute_scalar(1, &mut ctx)?;
        let s3 = encoded.clone().into_array().execute_scalar(3, &mut ctx)?;
        assert!(s1.is_null());
        assert!(s3.is_null());

        // Non-null values decode back to the original.
        let s0 = encoded.clone().into_array().execute_scalar(0, &mut ctx)?;
        assert_eq!(s0, Scalar::primitive(-128i32, Nullability::Nullable));

        let decoded = encoded.into_array().execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(decoded, input);
        Ok(())
    }

    #[test]
    fn slice_round_trip() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_iter([-10i32, -5, 0, 5, 10, 15, 20]);
        let encoded = OrderedLatent::encode(input.as_view(), &mut ctx)?;
        let sliced = encoded.into_array().slice(2..6)?;
        let expected = PrimitiveArray::from_iter([0i32, 5, 10, 15]);
        assert_arrays_eq!(sliced, expected);
        Ok(())
    }

    #[test]
    fn scalar_at_floats() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_iter([-1.5_f64, 0.0, 1.5]);
        let encoded = OrderedLatent::encode(input.as_view(), &mut ctx)?;
        let arr = encoded.into_array();
        assert_eq!(arr.execute_scalar(0, &mut ctx)?, Scalar::from(-1.5_f64));
        assert_eq!(arr.execute_scalar(1, &mut ctx)?, Scalar::from(0.0_f64));
        assert_eq!(arr.execute_scalar(2, &mut ctx)?, Scalar::from(1.5_f64));
        Ok(())
    }
}
