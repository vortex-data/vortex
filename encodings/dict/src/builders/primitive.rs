use std::hash::Hash;

use arrow_buffer::NullBufferBuilder;
use num_traits::{AsPrimitive, Unsigned};
use rustc_hash::FxBuildHasher;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::arrays::{NativeValue, PrimitiveArray};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, Nullability, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_panic};

use super::DictConstraints;
use crate::builders::DictEncoder;

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

impl<T: NativePType, Code> PrimitiveDictBuilder<T, Code>
where
    NativeValue<T>: Hash + Eq,
    Code: Unsigned + AsPrimitive<usize> + NativePType,
{
    pub fn new(nullability: Nullability, constraints: &DictConstraints) -> Self {
        let max_dict_len = constraints
            .max_len
            .min(constraints.max_bytes / T::PTYPE.byte_width());
        Self {
            lookup: HashMap::with_hasher(FxBuildHasher),
            values: BufferMut::<T>::empty(),
            nullability,
            max_dict_len,
        }
    }

    #[inline]
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
                vac.insert(next_code);
                self.values.push(v);
                Some(next_code)
            }
        }
    }
}

/// Dictionary encode primitive array with given PType.
/// Null values in the original array are encoded in the dictionary.
pub struct PrimitiveDictBuilder<T, Codes> {
    lookup: HashMap<NativeValue<T>, Codes, FxBuildHasher>,
    values: BufferMut<T>,
    nullability: Nullability,
    max_dict_len: usize,
}

impl<T: NativePType, Code> DictEncoder for PrimitiveDictBuilder<T, Code>
where
    NativeValue<T>: Hash + Eq,
    Code: Unsigned + AsPrimitive<usize> + NativePType,
{
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef> {
        if T::PTYPE != PType::try_from(array.dtype())? {
            vortex_bail!("Can only encode arrays of {}", T::PTYPE);
        }
        let mut codes = BufferMut::<Code>::with_capacity(array.len());
        let primitive = array.to_primitive()?;

        let codes = if array.dtype().is_nullable() {
            let mut null_buf = NullBufferBuilder::new(array.len());
            primitive.with_iterator(|it| {
                for value in it {
                    let (code, validity) = match value {
                        Some(v) => match self.encode_value(*v) {
                            Some(code) => (code, true),
                            None => break,
                        },
                        None => (Code::zero(), false),
                    };
                    null_buf.append(validity);
                    unsafe { codes.push_unchecked(code) }
                }
            })?;
            PrimitiveArray::new(
                codes,
                null_buf
                    .finish()
                    .map(Validity::from)
                    .unwrap_or(Validity::AllValid),
            )
        } else {
            primitive.with_iterator(|it| {
                for value in it {
                    let Some(code) = self.encode_value(
                        *value.vortex_expect("Dict encode null value in non-nullable array"),
                    ) else {
                        break;
                    };
                    unsafe { codes.push_unchecked(code) }
                }
            })?;
            PrimitiveArray::new(codes, Validity::NonNullable)
        };

        Ok(codes.into_array())
    }

    fn values(&mut self) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.values.clone().freeze(), self.nullability.into()).into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::builders::dict_encode;

    #[test]
    fn encode_primitive() {
        let arr = PrimitiveArray::from_iter([1, 1, 3, 3, 3]);
        let dict = dict_encode(&arr).unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u8>(),
            &[0, 0, 1, 1, 1]
        );
        assert_eq!(
            dict.values().to_primitive().unwrap().as_slice::<i32>(),
            &[1, 3]
        );
    }

    #[test]
    fn encode_primitive_nulls() {
        let arr = PrimitiveArray::from_option_iter([
            Some(1),
            Some(1),
            None,
            Some(3),
            Some(3),
            None,
            Some(3),
            None,
        ]);
        let dict = dict_encode(&arr).unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u8>(),
            &[0, 0, 0, 1, 1, 0, 1, 0]
        );
        let dict_values = dict.values();
        assert_eq!(
            dict_values.scalar_at(0).unwrap(),
            Scalar::primitive(1, Nullable)
        );
        assert_eq!(
            dict_values.scalar_at(1).unwrap(),
            Scalar::primitive(3, Nullable)
        );
    }
}
