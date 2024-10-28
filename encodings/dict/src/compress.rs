use std::hash::{BuildHasher, Hash, Hasher};

use hashbrown::hash_map::{Entry, RawEntryMut};
use hashbrown::{DefaultHashBuilder, HashMap};
use num_traits::AsPrimitive;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::array::{PrimitiveArray, VarBinArray, VarBinViewArray};
use vortex_array::validity::Validity;
use vortex_array::{ArrayDType, IntoArray, IntoCanonical};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, ToBytes};
use vortex_error::{VortexExpect as _, VortexUnwrap};

/// Statically assigned code for a null value.
pub const NULL_CODE: u64 = 0;

#[derive(Debug)]
struct Value<T>(T);

impl<T: ToBytes> Hash for Value<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_le_bytes().hash(state)
    }
}

impl<T: ToBytes> PartialEq<Self> for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_le_bytes().eq(other.0.to_le_bytes())
    }
}

impl<T: ToBytes> Eq for Value<T> {}

pub fn dict_encode_primitive(array: &PrimitiveArray) -> (PrimitiveArray, PrimitiveArray) {
    match_each_native_ptype!(array.ptype(), |$P| {
        dict_encode_typed_primitive::<$P>(array)
    })
}

/// Dictionary encode primitive array with given PType.
/// Null values in the original array are encoded in the dictionary.
pub fn dict_encode_typed_primitive<T: NativePType>(
    array: &PrimitiveArray,
) -> (PrimitiveArray, PrimitiveArray) {
    let mut lookup_dict: HashMap<Value<T>, u64> = HashMap::new();
    let mut codes: Vec<u64> = Vec::new();
    let mut values: Vec<T> = Vec::new();

    if array.dtype().is_nullable() {
        values.push(T::zero());
    }

    ArrayAccessor::<T>::with_iterator(array, |iter| {
        for ov in iter {
            match ov {
                None => codes.push(NULL_CODE),
                Some(&v) => {
                    let code = match lookup_dict.entry(Value(v)) {
                        Entry::Occupied(o) => *o.get(),
                        Entry::Vacant(vac) => {
                            let next_code = values.len() as u64;
                            vac.insert(next_code.as_());
                            values.push(v);
                            next_code
                        }
                    };
                    codes.push(code);
                }
            }
        }
    })
    .vortex_expect("Failed to dictionary encode primitive array");

    let values_validity = if array.dtype().is_nullable() {
        let mut validity = vec![true; values.len()];
        validity[0] = false;

        validity.into()
    } else {
        Validity::NonNullable
    };

    (
        PrimitiveArray::from(codes),
        PrimitiveArray::from_vec(values, values_validity),
    )
}

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub fn dict_encode_varbin(array: &VarBinArray) -> (PrimitiveArray, VarBinArray) {
    array
        .with_iterator(|iter| dict_encode_typed_varbin(array.dtype().clone(), iter))
        .vortex_expect("Failed to dictionary encode varbin array")
}

/// Dictionary encode a VarbinViewArray.
pub fn dict_encode_varbinview(array: &VarBinViewArray) -> (PrimitiveArray, VarBinViewArray) {
    let (codes, values) = array
        .with_iterator(|iter| dict_encode_typed_varbin(array.dtype().clone(), iter))
        .vortex_unwrap();
    (
        codes,
        values
            .into_canonical()
            .vortex_expect("VarBin to canonical")
            .into_varbinview()
            .vortex_expect("VarBinView"),
    )
}

fn lookup_bytes<'a, T: NativePType + AsPrimitive<usize>>(
    offsets: &'a [T],
    bytes: &'a [u8],
    idx: usize,
) -> &'a [u8] {
    let begin: usize = offsets[idx].as_();
    let end: usize = offsets[idx + 1].as_();
    &bytes[begin..end]
}

fn dict_encode_typed_varbin<I, U>(dtype: DType, values: I) -> (PrimitiveArray, VarBinArray)
where
    I: Iterator<Item = Option<U>>,
    U: AsRef<[u8]>,
{
    let (lower, _) = values.size_hint();
    let hasher = DefaultHashBuilder::default();
    let mut lookup_dict: HashMap<u64, (), ()> = HashMap::with_hasher(());
    let mut codes: Vec<u64> = Vec::with_capacity(lower);
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();
    offsets.push(0);

    if dtype.is_nullable() {
        offsets.push(0);
    }

    for o_val in values {
        match o_val {
            None => codes.push(0),
            Some(val) => {
                let byte_ref = val.as_ref();
                let value_hash = hasher.hash_one(byte_ref);
                let raw_entry = lookup_dict.raw_entry_mut().from_hash(value_hash, |idx| {
                    byte_ref == lookup_bytes(offsets.as_slice(), bytes.as_slice(), idx.as_())
                });

                let code = match raw_entry {
                    RawEntryMut::Occupied(o) => *o.into_key(),
                    RawEntryMut::Vacant(vac) => {
                        let next_code = offsets.len() as u64 - 1;
                        bytes.extend_from_slice(byte_ref);
                        offsets.push(bytes.len() as u32);
                        vac.insert_with_hasher(value_hash, next_code, (), |idx| {
                            hasher.hash_one(lookup_bytes(
                                offsets.as_slice(),
                                bytes.as_slice(),
                                idx.as_(),
                            ))
                        });
                        next_code
                    }
                };
                codes.push(code)
            }
        }
    }

    let values_validity = if dtype.is_nullable() {
        let mut validity = Vec::with_capacity(offsets.len() - 1);
        validity.push(false);
        validity.extend(vec![true; offsets.len() - 2]);

        validity.into()
    } else {
        Validity::NonNullable
    };

    (
        PrimitiveArray::from(codes),
        VarBinArray::try_new(
            PrimitiveArray::from(offsets).into_array(),
            PrimitiveArray::from(bytes).into_array(),
            dtype,
            values_validity,
        )
        .vortex_expect("Failed to create VarBinArray dictionary during encoding"),
    )
}

#[cfg(test)]
mod test {
    use std::str;

    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{PrimitiveArray, VarBinArray};
    use vortex_array::compute::unary::scalar_at;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, PType};
    use vortex_scalar::Scalar;

    use crate::compress::{dict_encode_typed_primitive, dict_encode_varbin};

    #[test]
    fn encode_primitive() {
        let arr = PrimitiveArray::from(vec![1, 1, 3, 3, 3]);
        let (codes, values) = dict_encode_typed_primitive::<i32>(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[0, 0, 1, 1, 1]);
        assert_eq!(values.maybe_null_slice::<i32>(), &[1, 3]);
    }

    #[test]
    fn encode_primitive_nulls() {
        let arr = PrimitiveArray::from_nullable_vec(vec![
            Some(1),
            Some(1),
            None,
            Some(3),
            Some(3),
            None,
            Some(3),
            None,
        ]);
        let (codes, values) = dict_encode_typed_primitive::<i32>(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[1, 1, 0, 2, 2, 0, 2, 0]);
        assert_eq!(
            scalar_at(&values, 0).unwrap(),
            Scalar::null(DType::Primitive(PType::I32, Nullable))
        );
        assert_eq!(
            scalar_at(&values, 1).unwrap(),
            Scalar::primitive(1, Nullable)
        );
        assert_eq!(
            scalar_at(&values, 2).unwrap(),
            Scalar::primitive(3, Nullable)
        );
    }

    #[test]
    fn encode_varbin() {
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "again", "world"]);
        let (codes, values) = dict_encode_varbin(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[0, 1, 0, 2, 1]);
        values
            .with_iterator(|iter| {
                assert_eq!(
                    iter.flatten()
                        .map(|b| unsafe { str::from_utf8_unchecked(b) })
                        .collect::<Vec<_>>(),
                    vec!["hello", "world", "again"]
                );
            })
            .unwrap();
    }

    #[test]
    fn encode_varbin_nulls() {
        let arr: VarBinArray = vec![
            Some("hello"),
            None,
            Some("world"),
            Some("hello"),
            None,
            Some("again"),
            Some("world"),
            None,
        ]
        .into_iter()
        .collect();
        let (codes, values) = dict_encode_varbin(&arr);
        assert_eq!(codes.maybe_null_slice::<u64>(), &[1, 0, 2, 1, 0, 3, 2, 0]);
        assert_eq!(str::from_utf8(&values.bytes_at(0).unwrap()).unwrap(), "");
        values
            .with_iterator(|iter| {
                assert_eq!(
                    iter.map(|b| b.map(|v| unsafe { str::from_utf8_unchecked(v) }))
                        .collect::<Vec<_>>(),
                    vec![None, Some("hello"), Some("world"), Some("again")]
                );
            })
            .unwrap();
    }

    #[test]
    fn repeated_values() {
        let arr = VarBinArray::from(vec!["a", "a", "b", "b", "a", "b", "a", "b"]);
        let (codes, values) = dict_encode_varbin(&arr);
        values
            .with_iterator(|iter| {
                assert_eq!(
                    iter.flatten()
                        .map(|b| unsafe { str::from_utf8_unchecked(b) })
                        .collect::<Vec<_>>(),
                    vec!["a", "b"]
                );
            })
            .unwrap();
        assert_eq!(
            codes.maybe_null_slice::<u64>(),
            &[0u64, 0, 1, 1, 0, 1, 0, 1]
        );
    }
}
