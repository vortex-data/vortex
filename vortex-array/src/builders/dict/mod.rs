// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Reverse;

use bytes::bytes_dict_builder;
use decimal::decimal_dict_builder;
use primitive::primitive_dict_builder;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
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

/// Reorders a [`DictArray`]'s dictionary values by descending code occurrence count and remaps
/// its codes accordingly, so the hottest entries receive codes `0, 1, ...`.
///
/// Ties are broken stably by original code index (preserving first-seen order when `dict` came
/// from [`dict_encode`] or [`dict_encode_with_constraints`]).
pub fn sort_dict_by_frequency(dict: DictArray, ctx: &mut ExecutionCtx) -> VortexResult<DictArray> {
    if dict.values().len() <= 1 || dict.is_empty() {
        return Ok(dict);
    }
    let codes = dict.codes().clone().execute::<PrimitiveArray>(ctx)?;
    match codes.ptype() {
        PType::U8 => sort_dict_by_frequency_typed::<u8>(dict, codes, ctx),
        PType::U16 => sort_dict_by_frequency_typed::<u16>(dict, codes, ctx),
        PType::U32 => sort_dict_by_frequency_typed::<u32>(dict, codes, ctx),
        PType::U64 => sort_dict_by_frequency_typed::<u64>(dict, codes, ctx),
        ptype => vortex_bail!("expected unsigned dictionary codes, got {ptype}"),
    }
}

fn sort_dict_by_frequency_typed<Code: UnsignedPType>(
    dict: DictArray,
    codes: PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DictArray> {
    let dict_len = dict.values().len();
    let slice = codes.as_slice::<Code>();
    let mut counts = vec![0u64; dict_len];

    match codes.validity()? {
        Validity::NonNullable | Validity::AllValid => {
            for &code in slice {
                if let Some(count) = counts.get_mut(code.as_()) {
                    *count += 1;
                }
            }
        }
        Validity::AllInvalid => return Ok(dict),
        validity => {
            let mask = validity.execute_mask(codes.len(), ctx)?;
            match mask.bit_buffer() {
                AllOr::All => {
                    for &code in slice {
                        if let Some(count) = counts.get_mut(code.as_()) {
                            *count += 1;
                        }
                    }
                }
                AllOr::None => return Ok(dict),
                AllOr::Some(bits) => {
                    for idx in bits.set_indices() {
                        if let Some(count) = counts.get_mut(slice[idx].as_()) {
                            *count += 1;
                        }
                    }
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..dict_len).collect();
    order.sort_by_key(|&old_code| Reverse(counts[old_code]));
    if order
        .iter()
        .enumerate()
        .all(|(new_code, &old_code)| new_code == old_code)
    {
        return Ok(dict);
    }

    let mut new_code_of = vec![Code::default(); dict_len];
    for (new_code, &old_code) in order.iter().enumerate() {
        if counts[old_code] == 0 {
            break;
        }
        new_code_of[old_code] =
            Code::from_usize(new_code).vortex_expect("referenced entry fits in original code type");
    }

    let mut remapped = BufferMut::<Code>::with_capacity_in(slice.len(), ctx.allocator().clone());
    for &old_code in slice {
        let mapped = new_code_of.get(old_code.as_()).copied().unwrap_or_default();
        // SAFETY: `remapped` was allocated with capacity `slice.len()`.
        unsafe { remapped.push_unchecked(mapped) };
    }
    let new_codes = PrimitiveArray::new(remapped.freeze(), codes.validity()?).into_array();

    let take_indices = Buffer::<u64>::from_iter(order.into_iter().map(|i| i as u64)).into_array();
    let new_values = DictArray::try_new(take_indices, dict.values().clone())?
        .into_array()
        .execute::<Canonical>(ctx)?
        .into_array();

    let all_referenced = dict.has_all_values_referenced();
    // SAFETY: `order` is a bijection on `0..dict_len`, and every valid code in `new_codes`
    // was mapped through its inverse `new_code_of`, preserving logical values and `all_values_referenced`.
    Ok(unsafe {
        DictArray::new_unchecked(new_codes, new_values).set_all_values_referenced(all_referenced)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::dict_encode;
    use super::sort_dict_by_frequency;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::DictArray;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArrayExt;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::primitive::PrimitiveArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

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
    fn sort_dict_by_frequency_orders_descending_with_stable_ties() -> VortexResult<()> {
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

        let dict = dict_encode(&array, &mut ctx)?;
        let sorted = sort_dict_by_frequency(dict, &mut ctx)?;
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
    fn sort_dict_by_frequency_handles_nullable_values_and_nullable_codes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Nullable values via `dict_encode`: null occurs 4 times, "a" twice, "b" once -> null gets code 0.
        let nullable_str = VarBinViewArray::from_iter(
            [Some("b"), Some("a"), None, None, Some("a"), None, None],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let sorted_str = sort_dict_by_frequency(dict_encode(&nullable_str, &mut ctx)?, &mut ctx)?;
        assert_eq!(
            sorted_str.codes().as_::<Primitive>().as_slice::<u8>(),
            &[2, 1, 0, 0, 1, 0, 0]
        );
        assert_arrays_eq!(sorted_str, nullable_str, &mut ctx);

        // Nullable codes where invalid slots carry an out-of-bounds payload (`255`) or a rare valid code (`0`):
        // valid counts: code 0 -> 1, code 1 -> 3. Invalid slots must not inflate code 0 or panic on 255.
        let codes = PrimitiveArray::new(
            buffer![0u8, 0, 255, 1, 1, 1],
            Validity::Array(
                BoolArray::from_iter([true, false, false, true, true, true]).into_array(),
            ),
        )
        .into_array();
        let values = buffer![10i32, 20].into_array();
        let dict = DictArray::try_new(codes, values)?;
        let sorted = sort_dict_by_frequency(dict.clone(), &mut ctx)?;
        assert_eq!(
            sorted.values().as_::<Primitive>().as_slice::<i32>(),
            &[20, 10]
        );
        assert_arrays_eq!(sorted, dict, &mut ctx);
        Ok(())
    }

    #[test]
    fn sort_dict_by_frequency_allows_unused_entries_beyond_code_width() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let codes = buffer![0u8, 1, 1, 1].into_array();
        let values = PrimitiveArray::from_iter(0i32..300).into_array();
        let dict = DictArray::try_new(codes, values)?;
        let sorted = sort_dict_by_frequency(dict.clone(), &mut ctx)?;
        assert_eq!(
            &sorted.values().as_::<Primitive>().as_slice::<i32>()[..2],
            &[1, 0]
        );
        assert_eq!(
            sorted.codes().as_::<Primitive>().as_slice::<u8>(),
            &[1, 0, 0, 0]
        );
        assert_arrays_eq!(sorted, dict, &mut ctx);
        Ok(())
    }
}
