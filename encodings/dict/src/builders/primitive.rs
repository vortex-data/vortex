use std::hash::Hash;

use arrow_buffer::NullBufferBuilder;
use num_traits::AsPrimitive;
use rustc_hash::FxBuildHasher;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::aliases::hash_map::{Entry, HashMap};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, ToCanonical};
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, Nullability, PType};
use vortex_error::{vortex_bail, VortexExpect, VortexResult};

use crate::builders::DictEncoder;

impl<T: NativePType> PrimitiveDictBuilder<T>
where
    private::Value<T>: Hash + Eq,
{
    pub fn new(nullability: Nullability) -> Self {
        Self {
            lookup: HashMap::with_hasher(FxBuildHasher),
            values: BufferMut::<T>::empty(),
            nullability,
        }
    }

    #[inline]
    fn encode_value(&mut self, v: T) -> u64 {
        match self.lookup.entry(private::Value(v)) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(vac) => {
                let next_code = self.values.len() as u64;
                vac.insert(next_code.as_());
                self.values.push(v);
                next_code
            }
        }
    }
}

/// Dictionary encode primitive array with given PType.
/// Null values in the original array are encoded in the dictionary.
pub struct PrimitiveDictBuilder<T> {
    lookup: HashMap<private::Value<T>, u64, FxBuildHasher>,
    values: BufferMut<T>,
    nullability: Nullability,
}

impl<T: NativePType> DictEncoder for PrimitiveDictBuilder<T>
where
    private::Value<T>: Hash + Eq,
{
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef> {
        if T::PTYPE != PType::try_from(array.dtype())? {
            vortex_bail!("Can only encode arrays of {}", T::PTYPE);
        }

        let mut codes = BufferMut::<u64>::with_capacity(array.len());
        let primitive = array.to_primitive()?;

        let codes = if array.dtype().is_nullable() {
            let mut null_buf = NullBufferBuilder::new(array.len());
            primitive.with_iterator(|it| {
                for value in it {
                    let (code, validity) = value
                        .map(|v| (self.encode_value(*v), true))
                        .unwrap_or((0, false));
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
                    let code = self.encode_value(
                        *value.vortex_expect("Dict encode null value in non-nullable array"),
                    );
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

mod private {
    use vortex_dtype::{half, NativePType};

    /// Value serves as a wrapper type to allow us to implement Hash and Eq on all primitive types.
    ///
    /// Rust does not define Hash/Eq for any of the float types due to the presence of
    /// NaN and +/- 0. We don't care about storing multiple NaNs or zeros in our dictionaries,
    /// so we define simple bit-wise Hash/Eq for the Value-wrapped versions of these types.
    #[derive(Debug)]
    pub struct Value<T>(pub T);

    impl<T> PartialEq<Value<T>> for Value<T>
    where
        T: NativePType,
    {
        fn eq(&self, other: &Value<T>) -> bool {
            self.0.is_eq(other.0)
        }
    }

    impl<T> Eq for Value<T> where T: NativePType {}

    macro_rules! prim_value {
        ($typ:ty) => {
            impl core::hash::Hash for Value<$typ> {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    self.0.hash(state);
                }
            }
        };
    }

    macro_rules! float_value {
        ($typ:ty) => {
            impl core::hash::Hash for Value<$typ> {
                fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                    self.0.to_bits().hash(state);
                }
            }
        };
    }

    prim_value!(u8);
    prim_value!(u16);
    prim_value!(u32);
    prim_value!(u64);
    prim_value!(i8);
    prim_value!(i16);
    prim_value!(i32);
    prim_value!(i64);
    float_value!(half::f16);
    float_value!(f32);
    float_value!(f64);
}
#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::scalar_at;
    use vortex_array::ToCanonical;
    use vortex_dtype::Nullability::Nullable;
    use vortex_scalar::Scalar;

    use crate::builders::dict_encode;

    #[test]
    fn encode_primitive() {
        let arr = PrimitiveArray::from_iter([1, 1, 3, 3, 3]);
        let dict = dict_encode(&arr).unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u64>(),
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
            dict.codes().to_primitive().unwrap().as_slice::<u64>(),
            &[0, 0, 0, 1, 1, 0, 1, 0]
        );
        let dict_values = dict.values();
        assert_eq!(
            scalar_at(dict_values, 0).unwrap(),
            Scalar::primitive(1, Nullable)
        );
        assert_eq!(
            scalar_at(dict_values, 1).unwrap(),
            Scalar::primitive(3, Nullable)
        );
    }
}
