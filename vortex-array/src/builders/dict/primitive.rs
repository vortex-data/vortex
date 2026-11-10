// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::OnceCell;
use std::hash::Hash;
use std::mem;

use rustc_hash::FxBuildHasher;
use vortex_buffer::Alignment;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;

use super::DictConstraints;
use super::DictEncoder;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::NativeValue;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::validity::Validity;

/// Creates an encoder with a fixed code width based on the value domain and dictionary length limit.
///
/// The bound includes a possible null entry, so nullable values may require wider codes even when
/// few distinct values occur. [`super::dict_encode_with_constraints`] narrows the codes after
/// encoding; callers using this encoder directly retain the selected width.
pub fn primitive_dict_builder<T: NativePType>(
    nullability: Nullability,
    constraints: &DictConstraints,
    allocator: BufferAllocatorRef,
) -> Box<dyn DictEncoder>
where
    NativeValue<T>: Hash + Eq,
{
    // Codes start at zero, and nullable values need an additional dictionary entry for null.
    let max_code = match T::PTYPE.bit_width() {
        8 => u8::MAX as u64,
        16 => u16::MAX as u64,
        32 => u32::MAX as u64,
        64 => u64::MAX,
        width => vortex_panic!("invalid bit_width: {width}"),
    }
    .saturating_add(u64::from(nullability.is_nullable()));
    let max_code = max_code.min(constraints.max_len.saturating_sub(1) as u64);
    match max_code {
        max if max <= u8::MAX as u64 => Box::new(PrimitiveDictBuilder::<T, u8>::new_in(
            nullability,
            constraints,
            allocator,
        )),
        max if max <= u16::MAX as u64 => Box::new(PrimitiveDictBuilder::<T, u16>::new_in(
            nullability,
            constraints,
            allocator,
        )),
        max if max <= u32::MAX as u64 => Box::new(PrimitiveDictBuilder::<T, u32>::new_in(
            nullability,
            constraints,
            allocator,
        )),
        _ => Box::new(PrimitiveDictBuilder::<T, u64>::new_in(
            nullability,
            constraints,
            allocator,
        )),
    }
}

impl<T, Code> PrimitiveDictBuilder<T, Code>
where
    T: NativePType,
    NativeValue<T>: Hash + Eq,
    Code: UnsignedPType,
{
    pub fn new_in(
        nullability: Nullability,
        constraints: &DictConstraints,
        allocator: BufferAllocatorRef,
    ) -> Self {
        let max_dict_len = constraints
            .max_len
            .min(constraints.max_bytes / T::PTYPE.byte_width());
        Self {
            lookup: HashMap::with_hasher(FxBuildHasher),
            null_code: OnceCell::new(),
            values: BufferMut::<T>::empty_aligned_in(Alignment::of::<T>(), allocator.clone()),
            values_nulls: BitBufferMut::empty_in(allocator.clone()),
            allocator,
            nullability,
            max_dict_len,
        }
    }

    /// Returns `None` when assigning a code would exceed the dictionary constraints,
    /// and callers should stop encoding after the current prefix.
    fn encode_value(&mut self, v: T) -> Option<Code> {
        match self.lookup.entry(NativeValue(v)) {
            Entry::Occupied(o) => Some(*o.get()),
            Entry::Vacant(vac) => {
                // Admit the first entry even if it exceeds the budget, so the caller makes progress.
                if !self.values.is_empty() && self.values.len() >= self.max_dict_len {
                    return None;
                }
                let next_code = Code::from_usize(self.values.len()).unwrap_or_else(|| {
                    vortex_panic!("{} has to fit into {}", self.values.len(), Code::PTYPE)
                });
                self.values.push(v);
                self.values_nulls.append_true();
                Some(*vac.insert(next_code))
            }
        }
    }

    /// Returns `None` when assigning the null code would exceed the dictionary constraints,
    /// and callers should stop encoding after the current prefix.
    fn encode_null(&mut self) -> Option<Code> {
        if let Some(code) = self.null_code.get() {
            return Some(*code);
        }

        if !self.values.is_empty() && self.values.len() >= self.max_dict_len {
            return None;
        }

        let code = Code::from_usize(self.values.len()).unwrap_or_else(|| {
            vortex_panic!("{} has to fit into {}", self.values.len(), Code::PTYPE)
        });
        self.values.push(T::default());
        self.values_nulls.append_false();
        self.null_code
            .set(code)
            .ok()
            .vortex_expect("null code is initialized once");
        Some(code)
    }
}

/// Dictionary encode primitive array with given PType.
///
/// Null values are stored in the values of the dictionary such that codes are always non-null.
pub struct PrimitiveDictBuilder<T, Code> {
    lookup: HashMap<NativeValue<T>, Code, FxBuildHasher>,
    null_code: OnceCell<Code>,
    values: BufferMut<T>,
    values_nulls: BitBufferMut,
    nullability: Nullability,
    max_dict_len: usize,
    allocator: BufferAllocatorRef,
}

impl<T, Code> DictEncoder for PrimitiveDictBuilder<T, Code>
where
    T: NativePType,
    NativeValue<T>: Hash + Eq,
    Code: UnsignedPType,
{
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        let mut codes = BufferMut::<Code>::with_capacity_in(array.len(), self.allocator.clone());

        let prim = array.clone().execute::<PrimitiveArray>(ctx)?;
        match prim.validity()?.execute_mask(array.len(), ctx)? {
            Mask::AllTrue(_) => {
                for &value in prim.as_slice::<T>() {
                    let Some(code) = self.encode_value(value) else {
                        break;
                    };
                    unsafe { codes.push_unchecked(code) }
                }
            }
            Mask::AllFalse(_) => {
                if let Some(code) = self.encode_null() {
                    unsafe { codes.push_n_unchecked(code, array.len()) }
                }
            }
            Mask::Values(v) => {
                let bit_buff = v.bit_buffer();
                for (&value, valid) in prim.as_slice::<T>().iter().zip(bit_buff) {
                    if !valid {
                        let Some(code) = self.encode_null() else {
                            break;
                        };
                        unsafe { codes.push_unchecked(code) }
                    } else {
                        let Some(code) = self.encode_value(value) else {
                            break;
                        };
                        unsafe { codes.push_unchecked(code) }
                    }
                }
            }
        }

        Ok(PrimitiveArray::new(codes, Validity::NonNullable))
    }

    fn reset(&mut self) -> ArrayRef {
        self.lookup.clear();
        self.null_code = OnceCell::new();
        let values = mem::replace(
            &mut self.values,
            BufferMut::empty_aligned_in(Alignment::of::<T>(), self.allocator.clone()),
        );
        let nulls = mem::replace(
            &mut self.values_nulls,
            BitBufferMut::empty_in(self.allocator.clone()),
        )
        .freeze();
        PrimitiveArray::new(values, Validity::from_bit_buffer(nulls, self.nullability)).into_array()
    }

    fn codes_ptype(&self) -> PType {
        Code::PTYPE
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::primitive_dict_builder;
    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::dict::DictConstraints;
    use crate::builders::dict::UNCONSTRAINED;
    use crate::builders::dict::dict_encode;
    use crate::builders::dict::dict_encoder_in;
    use crate::builders::dict::primitive::PrimitiveArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::match_each_integer_ptype;
    use crate::match_each_native_ptype;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);
    const USIZE_PTYPE: PType = PType::min_unsigned_ptype_for_value(usize::MAX as u64);

    #[test]
    fn encode_primitive() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = buffer![1, 1, 3, 3, 3].into_array();
        let dict = dict_encode(&arr, &mut SESSION.create_execution_ctx()).unwrap();

        let expected_codes = buffer![0u8, 0, 1, 1, 1].into_array();
        assert_arrays_eq!(dict.codes(), expected_codes, &mut ctx);

        let expected_values = buffer![1i32, 3].into_array();
        assert_arrays_eq!(dict.values(), expected_values, &mut ctx);
    }

    #[test]
    fn encode_primitive_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = PrimitiveArray::from_option_iter([
            Some(1),
            Some(1),
            None,
            Some(3),
            Some(3),
            None,
            Some(3),
            None,
        ])
        .into_array();
        let dict = dict_encode(&arr, &mut SESSION.create_execution_ctx()).unwrap();

        let expected_codes = buffer![0u8, 0, 1, 2, 2, 1, 2, 1].into_array();
        assert_arrays_eq!(dict.codes(), expected_codes, &mut ctx);

        let expected_values =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        assert_arrays_eq!(dict.values(), expected_values, &mut ctx);
    }

    #[test]
    fn reset_clears_dict() {
        let mut ctx = SESSION.create_execution_ctx();
        let first = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let constraints = DictConstraints {
            max_len: u32::MAX as usize,
            ..UNCONSTRAINED
        };
        let mut encoder = dict_encoder_in(&first, &constraints, ctx.allocator().clone());

        assert_arrays_eq!(
            encoder.encode(&first, &mut ctx).unwrap(),
            buffer![0u32, 1, 2].into_array(),
            &mut ctx
        );
        assert_arrays_eq!(encoder.reset(), first, &mut ctx);

        let second = PrimitiveArray::from_option_iter([Some(4i32), Some(5)]).into_array();
        assert_arrays_eq!(
            encoder.encode(&second, &mut ctx).unwrap(),
            buffer![0u32, 1].into_array(),
            &mut ctx
        );
        assert_arrays_eq!(encoder.reset(), second, &mut ctx);
    }

    #[rstest]
    #[case::u8(PType::U8, PType::U16)]
    #[case::i8(PType::I8, PType::U16)]
    #[case::u16(PType::U16, PType::U32)]
    #[case::i16(PType::I16, PType::U32)]
    fn encode_every_value_and_a_null(
        #[case] ptype: PType,
        #[case] codes_ptype: PType,
        #[values(false, true)] null_first: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = match_each_integer_ptype!(ptype, |T| {
            let mut values = (T::MIN..=T::MAX).map(Some).collect::<Vec<_>>();
            values.insert(if null_first { 0 } else { values.len() }, None);
            PrimitiveArray::from_option_iter(values).into_array()
        });
        let dict = dict_encode(&arr, &mut ctx)?;

        assert_eq!(
            dict.codes().dtype(),
            &DType::Primitive(codes_ptype, NonNullable)
        );
        assert_eq!(dict.values().len(), arr.len());
        assert_arrays_eq!(dict, arr, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::u8(PType::U8, NonNullable, usize::MAX, PType::U8)]
    #[case::u16(PType::U16, NonNullable, usize::MAX, PType::U16)]
    #[case::u32(PType::U32, NonNullable, usize::MAX, PType::U32)]
    #[case::nullable_u32(PType::U32, Nullable, usize::MAX, USIZE_PTYPE)]
    #[case::nullable_u64(PType::U64, Nullable, usize::MAX, USIZE_PTYPE)]
    #[case::nullable_f16(PType::F16, Nullable, usize::MAX, PType::U32)]
    #[case::nullable_f32(PType::F32, Nullable, usize::MAX, USIZE_PTYPE)]
    #[case::empty_budget(PType::U16, Nullable, 0, PType::U8)]
    #[case::full_u8_codes(PType::U16, Nullable, 256, PType::U8)]
    #[case::widen_u8_codes(PType::U16, Nullable, 257, PType::U16)]
    #[case::full_u16_codes(PType::U32, Nullable, 65_536, PType::U16)]
    #[case::widen_u16_codes(PType::U32, Nullable, 65_537, PType::U32)]
    fn codes_fit_dictionary_bounds(
        #[case] ptype: PType,
        #[case] nullability: Nullability,
        #[case] max_len: usize,
        #[case] expected_codes_ptype: PType,
    ) {
        let ctx = SESSION.create_execution_ctx();
        let constraints = DictConstraints {
            max_len,
            ..UNCONSTRAINED
        };
        let encoder = match_each_native_ptype!(ptype, |T| {
            primitive_dict_builder::<T>(nullability, &constraints, ctx.allocator().clone())
        });
        assert_eq!(encoder.codes_ptype(), expected_codes_ptype);
    }
}
