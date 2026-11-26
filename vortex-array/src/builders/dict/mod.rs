// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Reverse;

use bytes::bytes_dict_builder;
use bytes::bytes_dict_builder_with_counts;
use decimal::decimal_dict_builder;
use decimal::decimal_dict_builder_with_counts;
use primitive::primitive_dict_builder;
use primitive::primitive_dict_builder_with_counts;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::validity::Validity;

mod bytes;
mod decimal;
mod primitive;
mod values;

#[derive(Clone)]
pub struct DictConstraints {
    /// Limits the encoded storage for dictionary values.
    pub max_bytes: usize,
    /// Limits the number of dictionary values.
    pub max_len: usize,
}

pub const UNCONSTRAINED: DictConstraints = DictConstraints {
    max_bytes: usize::MAX,
    max_len: usize::MAX,
};

/// Returns the narrowest unsigned code type that can index `max_len` dictionary values.
///
/// Codes count from zero, so a code type indexes one more value than its maximum: `u8` codes
/// index up to 256 values.
pub const fn codes_ptype_for_max_len(max_len: u64) -> PType {
    match max_len {
        len if len <= 1 << u8::BITS => PType::U8,
        len if len <= 1 << u16::BITS => PType::U16,
        len if len <= 1 << u32::BITS => PType::U32,
        _ => PType::U64,
    }
}

pub trait DictEncoder: Send {
    /// Assign dictionary codes to the given input array.
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray>;

    /// Clear the encoder state to make it ready for a new round of decoding.
    fn reset(&mut self) -> ArrayRef;

    /// Returns the PType of the codes this encoder produces.
    fn codes_ptype(&self) -> PType;
}

/// Internal extension of [`DictEncoder`] that also returns per-entry occurrence counts accumulated
/// during encoding.
pub(super) trait CountingDictEncoder: DictEncoder {
    fn reset_with_counts(&mut self) -> (ArrayRef, Vec<u64>);
}

fn dict_encoder_with_counts_in(
    array: &ArrayRef,
    constraints: &DictConstraints,
    allocator: BufferAllocatorRef,
) -> Box<dyn CountingDictEncoder> {
    match array.dtype() {
        DType::Primitive(ptype, nullability) => match_each_native_ptype!(*ptype, |P| {
            primitive_dict_builder_with_counts::<P>(*nullability, constraints, allocator)
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            bytes_dict_builder_with_counts(array.dtype().clone(), constraints, allocator)
        }
        DType::Decimal(decimal_dtype, nullability) => {
            match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |D| {
                decimal_dict_builder_with_counts::<D>(
                    *decimal_dtype,
                    *nullability,
                    constraints,
                    allocator,
                )
            })
        }
        dtype => {
            vortex_panic!(
                "Can only dictionary encode primitive, decimal or binary arrays, got {dtype}"
            )
        }
    }
}

/// Create a dictionary encoder using the default allocator.
#[deprecated(note = "use `dict_encoder_in` with an explicit allocator")]
pub fn dict_encoder(array: &ArrayRef, constraints: &DictConstraints) -> Box<dyn DictEncoder> {
    dict_encoder_in(
        array,
        constraints,
        BufferAllocatorRef::statically_allocated(),
    )
}

/// Create a dictionary encoder using `allocator`.
pub fn dict_encoder_in(
    array: &ArrayRef,
    constraints: &DictConstraints,
    allocator: BufferAllocatorRef,
) -> Box<dyn DictEncoder> {
    // Choose by dtype: the encoders execute an input that is not already canonical, such as a
    // lazy slice.
    match array.dtype() {
        DType::Primitive(ptype, nullability) => match_each_native_ptype!(*ptype, |P| {
            primitive_dict_builder::<P>(*nullability, constraints, allocator)
        }),
        DType::Utf8(_) | DType::Binary(_) => {
            bytes_dict_builder(array.dtype().clone(), constraints, allocator)
        }
        DType::Decimal(decimal_dtype, nullability) => {
            match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |D| {
                decimal_dict_builder::<D>(*decimal_dtype, *nullability, constraints, allocator)
            })
        }
        dtype => {
            vortex_panic!(
                "Can only dictionary encode primitive, decimal or binary arrays, got {dtype}"
            )
        }
    }
}

/// Encode an array as a `DictArray` subject to the given constraints.
///
/// The codes use the narrowest unsigned integer type that can address the dictionary, so callers
/// have nothing left to narrow.
///
/// Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.
pub fn dict_encode_with_constraints(
    array: &ArrayRef,
    constraints: &DictConstraints,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    // Every row contributes at most one dictionary entry, so the array bounds the dictionary
    // length whatever the caller asked for. Encoders pick their code width up front from this
    // bound, so tightening it keeps a one-shot encode from writing `u64` codes for a dictionary
    // that cannot hold more than a handful of entries.
    let constraints = DictConstraints {
        max_bytes: constraints.max_bytes,
        max_len: constraints.max_len.min(array.len().max(1)),
    };
    let mut encoder = dict_encoder_in(array, &constraints, ctx.allocator().clone());
    let codes = encoder.encode(array, ctx)?;
    let values = encoder.reset();
    let codes = narrow_codes(codes, values.len(), ctx)?;
    // SAFETY: The encoding process will produce a value set of codes and values
    // All values in the dictionary are guaranteed to be referenced by at least one code
    // since we build the dictionary from the codes we observe during encoding
    unsafe {
        Ok(DictArray::new_unchecked(codes.into_array(), values).set_all_values_referenced(true))
    }
}

/// Narrow codes addressing a `dict_len`-entry dictionary to the smallest type that can hold them.
///
/// Encoders hand out consecutive codes and only ever add an entry they immediately use, so the
/// largest code is `dict_len - 1`. That makes the narrowest code type known without the scan
/// [`PrimitiveArrayExt::narrow`] would otherwise run over the codes.
fn narrow_codes(
    codes: PrimitiveArray,
    dict_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let ptype = codes_ptype_for_max_len(dict_len as u64);
    if codes.ptype() == ptype {
        return Ok(codes);
    }
    codes
        .as_ref()
        .cast(DType::Primitive(ptype, Nullability::NonNullable))?
        .execute::<PrimitiveArray>(ctx)
}

pub fn dict_encode(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<DictArray> {
    let dict_array = dict_encode_with_constraints(array, &UNCONSTRAINED, ctx)?;
    if dict_array.len() != array.len() {
        vortex_bail!(
            "must have encoded all {} elements, but only encoded {}",
            array.len(),
            dict_array.len(),
        );
    }
    Ok(dict_array)
}

/// Encode an array as a [`DictArray`] subject to `constraints`, ordering dictionary values by
/// descending occurrence count so the hottest entries receive codes `0, 1, ...`.
///
/// Returns `Ok(None)` without sorting or remapping if `array` exceeds `constraints`.
/// Counts are tracked during dictionary construction, and code remapping is fused with code
/// narrowing (or performed in place when the code width does not change). Ties are broken stably
/// by first-seen order.
pub fn dict_encode_by_frequency_with_constraints(
    array: &ArrayRef,
    constraints: &DictConstraints,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<DictArray>> {
    let constraints = DictConstraints {
        max_bytes: constraints.max_bytes,
        max_len: constraints.max_len.min(array.len().max(1)),
    };
    let mut encoder = dict_encoder_with_counts_in(array, &constraints, ctx.allocator().clone());
    let codes = encoder.encode(array, ctx)?;
    if codes.len() < array.len() {
        return Ok(None);
    }
    let (values, counts) = encoder.reset_with_counts();
    let dict_len = values.len();

    if dict_len <= 1 || codes.is_empty() {
        let codes = narrow_codes(codes, dict_len, ctx)?;
        // SAFETY: the encoder only emits codes in `0..dict_len` and references every entry.
        return Ok(Some(unsafe {
            DictArray::new_unchecked(codes.into_array(), values).set_all_values_referenced(true)
        }));
    }

    debug_assert_eq!(counts.len(), dict_len);
    let mut order: Vec<usize> = (0..dict_len).collect();
    order.sort_by_key(|&old_code| Reverse(counts[old_code]));
    if order
        .iter()
        .enumerate()
        .all(|(new_code, &old_code)| new_code == old_code)
    {
        let codes = narrow_codes(codes, dict_len, ctx)?;
        // SAFETY: the encoder only emits codes in `0..dict_len` and references every entry.
        return Ok(Some(unsafe {
            DictArray::new_unchecked(codes.into_array(), values).set_all_values_referenced(true)
        }));
    }

    let target_ptype = codes_ptype_for_max_len(dict_len as u64);
    let new_codes = remap_and_narrow_codes(codes, &order, target_ptype, ctx);

    let take_indices = Buffer::<u64>::from_iter(order.into_iter().map(|i| i as u64)).into_array();
    let new_values = DictArray::try_new(take_indices, values)?
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array();

    // SAFETY: `order` is a bijection on `0..dict_len`, every code in `new_codes` was remapped
    // through its inverse, and every dictionary entry was referenced during encoding.
    Ok(Some(unsafe {
        DictArray::new_unchecked(new_codes.into_array(), new_values).set_all_values_referenced(true)
    }))
}

/// Encode an entire array as a frequency-ordered [`DictArray`].
pub fn dict_encode_by_frequency(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    match dict_encode_by_frequency_with_constraints(array, &UNCONSTRAINED, ctx)? {
        Some(dict_array) => Ok(dict_array),
        None => vortex_bail!(
            "must have encoded all {} elements under unconstrained dictionary limits",
            array.len(),
        ),
    }
}

/// Remaps `codes` through the inverse of `order` while narrowing to `target_ptype` in one pass.
///
/// When `codes.ptype() == target_ptype`, remaps the uniquely owned code buffer in place.
fn remap_and_narrow_codes(
    codes: PrimitiveArray,
    order: &[usize],
    target_ptype: PType,
    ctx: &ExecutionCtx,
) -> PrimitiveArray {
    match (codes.ptype(), target_ptype) {
        (PType::U8, PType::U8) => remap_codes_in_place::<u8>(codes, order),
        (PType::U16, PType::U16) => remap_codes_in_place::<u16>(codes, order),
        (PType::U32, PType::U32) => remap_codes_in_place::<u32>(codes, order),
        (PType::U64, PType::U64) => remap_codes_in_place::<u64>(codes, order),
        (PType::U16, PType::U8) => remap_and_narrow_typed::<u16, u8>(codes, order, ctx),
        (PType::U32, PType::U8) => remap_and_narrow_typed::<u32, u8>(codes, order, ctx),
        (PType::U32, PType::U16) => remap_and_narrow_typed::<u32, u16>(codes, order, ctx),
        (PType::U64, PType::U8) => remap_and_narrow_typed::<u64, u8>(codes, order, ctx),
        (PType::U64, PType::U16) => remap_and_narrow_typed::<u64, u16>(codes, order, ctx),
        (PType::U64, PType::U32) => remap_and_narrow_typed::<u64, u32>(codes, order, ctx),
        (src, dst) => vortex_panic!("unexpected dictionary code narrowing from {src} to {dst}"),
    }
}

fn build_inverse_map<Dst: UnsignedPType>(order: &[usize]) -> Vec<Dst> {
    let mut new_code_of = vec![Dst::default(); order.len()];
    for (new_code, &old_code) in order.iter().enumerate() {
        new_code_of[old_code] =
            Dst::from_usize(new_code).vortex_expect("new code fits in target code type");
    }
    new_code_of
}

fn remap_codes_in_place<Code: UnsignedPType>(
    codes: PrimitiveArray,
    order: &[usize],
) -> PrimitiveArray {
    let new_code_of = build_inverse_map::<Code>(order);
    let mut buf = codes.into_buffer_mut::<Code>();
    for code in buf.as_mut_slice() {
        // SAFETY: the encoder only emits codes in `0..order.len() == new_code_of.len()`.
        *code = unsafe { *new_code_of.get_unchecked(code.as_()) };
    }
    // SAFETY: encoder codes are always non-nullable.
    unsafe { PrimitiveArray::new_unchecked(buf.freeze(), Validity::NonNullable) }
}

fn remap_and_narrow_typed<Src: UnsignedPType, Dst: UnsignedPType>(
    codes: PrimitiveArray,
    order: &[usize],
    ctx: &ExecutionCtx,
) -> PrimitiveArray {
    let new_code_of = build_inverse_map::<Dst>(order);
    let src_slice = codes.as_slice::<Src>();
    let mut dst = BufferMut::<Dst>::with_capacity_in(src_slice.len(), ctx.allocator().clone());
    for &old_code in src_slice {
        // SAFETY: `dst` has capacity `src_slice.len()`, and the encoder only emits codes in
        // `0..order.len() == new_code_of.len()`.
        unsafe { dst.push_unchecked(*new_code_of.get_unchecked(old_code.as_())) };
    }
    // SAFETY: encoder codes are always non-nullable.
    unsafe { PrimitiveArray::new_unchecked(dst.freeze(), Validity::NonNullable) }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::DictConstraints;
    use super::dict_encode;
    use super::dict_encode_by_frequency;
    use super::dict_encode_by_frequency_with_constraints;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArrayExt;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::primitive::PrimitiveArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// `narrow_codes` picks the code type from the dictionary length instead of scanning the
    /// codes, so it must land on the same type a scan would at every width boundary.
    #[rstest]
    #[case::empty(0)]
    #[case::single(1)]
    #[case::fills_u8(256)]
    #[case::overflows_u8(257)]
    #[case::fills_u16(65_536)]
    #[case::overflows_u16(65_537)]
    fn codes_are_as_narrow_as_a_scan_makes_them(#[case] distinct: u32) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter((0..distinct).chain(0..distinct)).into_array();

        let codes = dict_encode(&array, &mut ctx)?
            .codes()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_eq!(codes.ptype(), codes.narrow(&mut ctx)?.ptype());
        Ok(())
    }

    #[test]
    fn dict_encode_by_frequency_orders_descending_with_stable_ties() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // "rare_b" (1), "tie_first" (2), "hot" (4), "tie_second" (2)
        let array = VarBinViewArray::from_iter_str([
            "rare_b",
            "tie_first",
            "tie_second",
            "hot",
            "tie_first",
            "hot",
            "tie_second",
            "hot",
            "hot",
        ])
        .into_array();

        let sorted = dict_encode_by_frequency(&array, &mut ctx)?;
        assert!(sorted.has_all_values_referenced());
        assert_arrays_eq!(
            sorted.values(),
            VarBinViewArray::from_iter_str(["hot", "tie_first", "tie_second", "rare_b"]),
            &mut ctx
        );
        assert_eq!(
            sorted.codes().as_::<Primitive>().as_slice::<u8>(),
            &[3, 1, 2, 0, 1, 0, 2, 0, 0]
        );
        assert_arrays_eq!(sorted, array, &mut ctx);
        Ok(())
    }

    #[test]
    fn dict_encode_by_frequency_handles_nullable_and_narrowing() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Nullable values: null occurs 4 times, "a" twice, "b" once -> null gets code 0.
        let nullable_str = VarBinViewArray::from_iter(
            [Some("b"), Some("a"), None, None, Some("a"), None, None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let sorted_str = dict_encode_by_frequency(&nullable_str, &mut ctx)?;
        assert_eq!(
            sorted_str.codes().as_::<Primitive>().as_slice::<u8>(),
            &[2, 1, 0, 0, 1, 0, 0]
        );
        assert_arrays_eq!(sorted_str, nullable_str, &mut ctx);

        // Input > 256 rows with 2 distinct values exercises fused u16 -> u8 narrow + remap.
        let mut values = Vec::with_capacity(300);
        values.push(99i32);
        values.extend(std::iter::repeat_n(42i32, 299));
        let prim = PrimitiveArray::from_iter(values).into_array();
        let sorted_prim = dict_encode_by_frequency(&prim, &mut ctx)?;
        assert_eq!(
            sorted_prim.codes().dtype(),
            &DType::Primitive(PType::U8, Nullability::NonNullable)
        );
        assert_eq!(
            sorted_prim.values().as_::<Primitive>().as_slice::<i32>(),
            &[42, 99]
        );
        assert_arrays_eq!(sorted_prim, prim, &mut ctx);

        // Constrained encode with max_len <= 256 exercises in-place u8 -> u8 remap.
        let constraints = DictConstraints {
            max_bytes: usize::MAX,
            max_len: 16,
        };
        let constrained = dict_encode_by_frequency_with_constraints(&prim, &constraints, &mut ctx)?
            .expect("2 distinct values fit within max_len=16");
        assert_eq!(
            constrained.values().as_::<Primitive>().as_slice::<i32>(),
            &[42, 99]
        );
        assert_arrays_eq!(constrained, prim, &mut ctx);

        // Exceeding constraints returns Ok(None).
        let tight = DictConstraints {
            max_bytes: usize::MAX,
            max_len: 1,
        };
        assert!(dict_encode_by_frequency_with_constraints(&prim, &tight, &mut ctx)?.is_none());
        Ok(())
    }
}
