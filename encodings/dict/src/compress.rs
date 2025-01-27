use std::hash::{BuildHasher, Hash, Hasher};

use num_traits::AsPrimitive;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::aliases::hash_map::{DefaultHashBuilder, Entry, HashMap, HashTable, RandomState};
use vortex_array::array::{
    BinaryView, ConstantArray, PrimitiveArray, VarBinArray, VarBinViewArray,
};
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::{BufferMut, ByteBufferMut};
use vortex_dtype::{match_each_native_ptype, DType, NativePType, Nullability, PType, ToBytes};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult, VortexUnwrap};
use vortex_scalar::Scalar;
use vortex_sparse::SparseArray;

use crate::DictArray;

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

pub fn dict_encode(array: &ArrayData) -> VortexResult<DictArray> {
    let dict_builder: &mut dyn DictEncoder = if let Some(pa) = PrimitiveArray::maybe_from(array) {
        match_each_native_ptype!(pa.ptype(), |$P| {
            &mut PrimitiveDictBuilder::<$P>::new(pa.dtype().nullability())
        })
    } else if let Some(vbv) = VarBinViewArray::maybe_from(array) {
        &mut BytesDictBuilder::new(vbv.dtype().clone())
    } else if let Some(vb) = VarBinArray::maybe_from(array) {
        &mut BytesDictBuilder::new(vb.dtype().clone())
    } else {
        vortex_bail!("Can only encode primitive or varbin/view arrays")
    };
    let codes = dict_builder.encode_array(array)?;
    DictArray::try_new(codes, dict_builder.values())
}

pub trait DictEncoder {
    fn encode_array(&mut self, array: &ArrayData) -> VortexResult<ArrayData>;

    fn values(&mut self) -> ArrayData;
}

/// Dictionary encode primitive array with given PType.
/// Null values in the original array are encoded in the dictionary.
pub struct PrimitiveDictBuilder<T> {
    lookup: HashMap<Value<T>, u64>,
    values: BufferMut<T>,
    nullability: Nullability,
}

impl<T: NativePType> PrimitiveDictBuilder<T> {
    pub fn new(nullability: Nullability) -> Self {
        let mut values = BufferMut::<T>::empty();

        if nullability == Nullability::Nullable {
            values.push(T::zero());
        };

        Self {
            lookup: HashMap::new(),
            values,
            nullability,
        }
    }

    #[inline]
    fn encode_value(&mut self, v: T) -> u64 {
        match self.lookup.entry(Value(v)) {
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

impl<T: NativePType> DictEncoder for PrimitiveDictBuilder<T> {
    fn encode_array(&mut self, array: &ArrayData) -> VortexResult<ArrayData> {
        if array.dtype().is_nullable() && self.nullability == Nullability::NonNullable {
            vortex_bail!("Cannot encode nullable array into non nullable dictionary")
        }

        if T::PTYPE != PType::try_from(array.dtype())? {
            vortex_bail!("Can only encode arrays of {}", T::PTYPE);
        }

        let mut codes = BufferMut::<u64>::with_capacity(array.len());

        let primitive = array.clone().into_primitive()?;
        primitive.with_iterator(|it| {
            for value in it {
                let code = if let Some(&v) = value {
                    self.encode_value(v)
                } else {
                    NULL_CODE
                };
                unsafe { codes.push_unchecked(code) }
            }
        })?;

        Ok(PrimitiveArray::new(codes, Validity::NonNullable).into_array())
    }

    fn values(&mut self) -> ArrayData {
        let values_validity = dict_values_validity(self.nullability.into(), self.values.len());

        PrimitiveArray::new(self.values.clone().freeze(), values_validity).into_array()
    }
}

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub struct BytesDictBuilder {
    lookup: Option<HashTable<u64>>,
    views: BufferMut<BinaryView>,
    values: ByteBufferMut,
    hasher: RandomState,
    dtype: DType,
}

impl BytesDictBuilder {
    pub fn new(dtype: DType) -> Self {
        let mut views = BufferMut::<BinaryView>::empty();
        if dtype.is_nullable() {
            views.push(BinaryView::new_inlined(&[]));
        }

        Self {
            lookup: Some(HashTable::new()),
            views,
            values: BufferMut::empty(),
            hasher: DefaultHashBuilder::default(),
            dtype,
        }
    }

    #[inline]
    fn lookup_bytes(&self, idx: usize) -> &[u8] {
        let bin_view = &self.views[idx];
        if bin_view.is_inlined() {
            bin_view.as_inlined().value()
        } else {
            &self.values[bin_view.as_view().to_range()]
        }
    }

    #[inline]
    fn encode_value(&mut self, lookup: &mut HashTable<u64>, val: &[u8]) -> u64 {
        *lookup
            .entry(
                self.hasher.hash_one(val),
                |idx| val == self.lookup_bytes(idx.as_()),
                |idx| self.hasher.hash_one(self.lookup_bytes(idx.as_())),
            )
            .or_insert_with(|| {
                let next_code = self.views.len() as u64;
                if val.len() <= BinaryView::MAX_INLINED_SIZE {
                    self.views.push(BinaryView::new_inlined(val));
                } else {
                    self.views.push(BinaryView::new_view(
                        u32::try_from(val.len()).vortex_unwrap(),
                        val[0..4].try_into().vortex_unwrap(),
                        0,
                        u32::try_from(self.values.len()).vortex_unwrap(),
                    ));
                    self.values.extend_from_slice(val);
                }
                next_code
            })
            .get()
    }

    fn encode_bytes<A: ArrayAccessor<[u8]>>(
        &mut self,
        accessor: A,
        len: usize,
    ) -> VortexResult<ArrayData> {
        let mut local_lookup = self.lookup.take().vortex_expect("Must have a lookup dict");
        let mut codes: BufferMut<u64> = BufferMut::with_capacity(len);

        accessor.with_iterator(|it| {
            for value in it {
                let code = if let Some(v) = value {
                    self.encode_value(&mut local_lookup, v)
                } else {
                    NULL_CODE
                };
                unsafe { codes.push_unchecked(code) }
            }
        })?;

        // Restore lookup dictionary back into the struct
        self.lookup = Some(local_lookup);
        Ok(PrimitiveArray::new(codes, Validity::NonNullable).into_array())
    }
}

impl DictEncoder for BytesDictBuilder {
    fn encode_array(&mut self, array: &ArrayData) -> VortexResult<ArrayData> {
        if array.dtype().is_nullable() && !self.dtype.is_nullable() {
            vortex_bail!("Cannot encode nullable array into non nullable dictionary")
        }

        if !self.dtype.eq_ignore_nullability(array.dtype()) {
            vortex_bail!("Can only encode string or binary arrays");
        }

        let len = array.len();
        if let Some(varbinview) = VarBinViewArray::maybe_from(array) {
            return self.encode_bytes(varbinview, len);
        } else if let Some(varbin) = VarBinArray::maybe_from(array) {
            return self.encode_bytes(varbin, len);
        }

        vortex_bail!("Can only dictionary encode VarBin and VarBinView arrays");
    }

    fn values(&mut self) -> ArrayData {
        let values_validity = dict_values_validity(self.dtype.is_nullable(), self.views.len());
        VarBinViewArray::try_new(
            self.views.clone().freeze(),
            vec![self.values.clone().freeze()],
            self.dtype.clone(),
            values_validity,
        )
        .vortex_unwrap()
        .into_array()
    }
}

fn dict_values_validity(nullable: bool, len: usize) -> Validity {
    if nullable {
        Validity::Array(
            SparseArray::try_new(
                ConstantArray::new(0u64, 1).into_array(),
                ConstantArray::new(false, 1).into_array(),
                len,
                Scalar::from(true),
            )
            .vortex_unwrap()
            .into_array(),
        )
    } else {
        Validity::NonNullable
    }
}

#[cfg(test)]
mod test {
    use std::str;

    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::array::{PrimitiveArray, VarBinArray};
    use vortex_array::compute::scalar_at;
    use vortex_array::IntoArrayVariant;
    use vortex_dtype::Nullability::Nullable;
    use vortex_dtype::{DType, PType};
    use vortex_scalar::Scalar;

    use crate::dict_encode;

    #[test]
    fn encode_primitive() {
        let arr = PrimitiveArray::from_iter([1, 1, 3, 3, 3]);
        let dict = dict_encode(arr.as_ref()).unwrap();
        assert_eq!(
            dict.codes().into_primitive().unwrap().as_slice::<u64>(),
            &[0, 0, 1, 1, 1]
        );
        assert_eq!(
            dict.values().into_primitive().unwrap().as_slice::<i32>(),
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
        let dict = dict_encode(arr.as_ref()).unwrap();
        assert_eq!(
            dict.codes().into_primitive().unwrap().as_slice::<u64>(),
            &[1, 1, 0, 2, 2, 0, 2, 0]
        );
        let dict_values = dict.values();
        assert_eq!(
            scalar_at(&dict_values, 0).unwrap(),
            Scalar::null(DType::Primitive(PType::I32, Nullable))
        );
        assert_eq!(
            scalar_at(&dict_values, 1).unwrap(),
            Scalar::primitive(1, Nullable)
        );
        assert_eq!(
            scalar_at(&dict_values, 2).unwrap(),
            Scalar::primitive(3, Nullable)
        );
    }

    #[test]
    fn encode_varbin() {
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "again", "world"]);
        let dict = dict_encode(arr.as_ref()).unwrap();
        assert_eq!(
            dict.codes().into_primitive().unwrap().as_slice::<u64>(),
            &[0, 1, 0, 2, 1]
        );
        dict.values()
            .into_varbinview()
            .unwrap()
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
        let dict = dict_encode(arr.as_ref()).unwrap();
        assert_eq!(
            dict.codes().into_primitive().unwrap().as_slice::<u64>(),
            &[1, 0, 2, 1, 0, 3, 2, 0]
        );
        assert_eq!(
            str::from_utf8(&dict.values().into_varbinview().unwrap().bytes_at(0)).unwrap(),
            ""
        );
        dict.values()
            .into_varbinview()
            .unwrap()
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
        let dict = dict_encode(arr.as_ref()).unwrap();
        dict.values()
            .into_varbinview()
            .unwrap()
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
            dict.codes().into_primitive().unwrap().as_slice::<u64>(),
            &[0u64, 0, 1, 1, 0, 1, 0, 1]
        );
    }
}
