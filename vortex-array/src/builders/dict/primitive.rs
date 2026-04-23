// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::OnceCell;
use std::hash::Hash;
use std::mem;

use rustc_hash::FxBuildHasher;
use vortex_buffer::BitBufferMut;
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

pub fn primitive_dict_builder<T: NativePType>(
    nullability: Nullability,
    constraints: &DictConstraints,
) -> Box<dyn DictEncoder>
where
    NativeValue<T>: Hash + Eq,
{
    // bound constraints with the cardinality of the primitive type
    let max_possible_len = (constraints.max_len as u64).min(match T::PTYPE.bit_width() {
        8 => u8::MAX as u64,
        16 => u16::MAX as u64,
        32 => u32::MAX as u64,
        64 => u64::MAX,
        width => vortex_panic!("invalid bit_width: {width}"),
    });
    match max_possible_len {
        max if max <= u8::MAX as u64 => {
            Box::new(PrimitiveDictBuilder::<T, u8>::new(nullability, constraints))
        }
        max if max <= u16::MAX as u64 => Box::new(PrimitiveDictBuilder::<T, u16>::new(
            nullability,
            constraints,
        )),
        max if max <= u32::MAX as u64 => Box::new(PrimitiveDictBuilder::<T, u32>::new(
            nullability,
            constraints,
        )),
        _ => Box::new(PrimitiveDictBuilder::<T, u64>::new(
            nullability,
            constraints,
        )),
    }
}

impl<T, Code> PrimitiveDictBuilder<T, Code>
where
    T: NativePType,
    NativeValue<T>: Hash + Eq,
    Code: UnsignedPType,
{
    pub fn new(nullability: Nullability, constraints: &DictConstraints) -> Self {
        let max_dict_len = constraints
            .max_len
            .min(constraints.max_bytes / T::PTYPE.byte_width());
        Self {
            lookup: HashMap::with_hasher(FxBuildHasher),
            null_code: OnceCell::new(),
            values: BufferMut::<T>::empty(),
            values_nulls: BitBufferMut::empty(),
            nullability,
            max_dict_len,
        }
    }

    fn encode_value(&mut self, v: T) -> Option<Code> {
        match self.lookup.entry(NativeValue(v)) {
            Entry::Occupied(o) => Some(*o.get()),
            Entry::Vacant(vac) => {
                if self.values.len() >= self.max_dict_len {
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
}

impl<T, Code> DictEncoder for PrimitiveDictBuilder<T, Code>
where
    T: NativePType,
    NativeValue<T>: Hash + Eq,
    Code: UnsignedPType,
{
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        let mut codes = BufferMut::<Code>::with_capacity(array.len());

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
                self.values.push(T::default());
                self.values_nulls.append_false();
                unsafe {
                    codes.push_n_unchecked(
                        Code::from_usize(0).vortex_expect("must fit 0"),
                        array.len(),
                    )
                }
            }
            Mask::Values(v) => {
                let bit_buff = v.bit_buffer();
                for (&value, valid) in prim.as_slice::<T>().iter().zip(bit_buff) {
                    if !valid {
                        let code = self.null_code.get_or_init(|| {
                            let code = self.values.len();
                            self.values.push(T::default());
                            self.values_nulls.append_false();
                            Code::from_usize(code).unwrap_or_else(|| {
                                vortex_panic!("{} has to fit into {}", code, Code::PTYPE)
                            })
                        });
                        unsafe { codes.push_unchecked(*code) }
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
        PrimitiveArray::new(
            self.values.clone(),
            Validity::from_bit_buffer(mem::take(&mut self.values_nulls).freeze(), self.nullability),
        )
        .into_array()
    }

    fn codes_ptype(&self) -> PType {
        Code::PTYPE
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray as _;
    use crate::VortexSessionExecute;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::assert_arrays_eq;
    use crate::builders::dict::dict_encode;
    use crate::builders::dict::primitive::PrimitiveArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

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
}
