use std::hash::BuildHasher;

use arrow_buffer::NullBufferBuilder;
use num_traits::AsPrimitive;
use num_traits::sign::Unsigned;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::aliases::hash_map::{DefaultHashBuilder, HashTable, HashTableEntry, RandomState};
use vortex_array::arrays::{
    BinaryView, PrimitiveArray, VarBinVTable, VarBinViewArray, VarBinViewVTable,
};
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_buffer::{BufferMut, ByteBufferMut};
use vortex_dtype::{DType, NativePType};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_panic};

use super::DictConstraints;
use crate::builders::DictEncoder;

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub struct BytesDictBuilder<Codes> {
    lookup: Option<HashTable<Codes>>,
    views: BufferMut<BinaryView>,
    values: ByteBufferMut,
    hasher: RandomState,
    dtype: DType,
    max_dict_bytes: usize,
    max_dict_len: usize,
}

pub fn bytes_dict_builder(dtype: DType, constraints: &DictConstraints) -> Box<dyn DictEncoder> {
    match constraints.max_len as u64 {
        max if max <= u8::MAX as u64 => Box::new(BytesDictBuilder::<u8>::new(dtype, constraints)),
        max if max <= u16::MAX as u64 => Box::new(BytesDictBuilder::<u16>::new(dtype, constraints)),
        max if max <= u32::MAX as u64 => Box::new(BytesDictBuilder::<u32>::new(dtype, constraints)),
        _ => Box::new(BytesDictBuilder::<u64>::new(dtype, constraints)),
    }
}

impl<Code: Unsigned + AsPrimitive<usize> + NativePType> BytesDictBuilder<Code> {
    pub fn new(dtype: DType, constraints: &DictConstraints) -> Self {
        Self {
            lookup: Some(HashTable::new()),
            views: BufferMut::<BinaryView>::empty(),
            values: BufferMut::empty(),
            hasher: DefaultHashBuilder::default(),
            dtype,
            max_dict_bytes: constraints.max_bytes,
            max_dict_len: constraints.max_len,
        }
    }

    fn dict_bytes(&self) -> usize {
        self.views.len() * size_of::<BinaryView>() + self.values.len()
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
    fn encode_value(&mut self, lookup: &mut HashTable<Code>, val: &[u8]) -> Option<Code> {
        match lookup.entry(
            self.hasher.hash_one(val),
            |idx| val == self.lookup_bytes(idx.as_()),
            |idx| self.hasher.hash_one(self.lookup_bytes(idx.as_())),
        ) {
            HashTableEntry::Occupied(occupied) => Some(*occupied.get()),
            HashTableEntry::Vacant(vacant) => {
                if self.views.len() >= self.max_dict_len {
                    return None;
                }

                let next_code = self.views.len();
                let view =
                    BinaryView::make_view(val, 0, u32::try_from(self.values.len()).vortex_unwrap());
                let additional_bytes = if view.is_inlined() {
                    size_of::<BinaryView>()
                } else {
                    size_of::<BinaryView>() + val.len()
                };

                if self.dict_bytes() + additional_bytes > self.max_dict_bytes {
                    return None;
                }

                self.views.push(view);
                if !view.is_inlined() {
                    self.values.extend_from_slice(val);
                }
                let next_code = Code::from_usize(next_code).unwrap_or_else(|| {
                    vortex_panic!("{next_code} has to fit into {}", Code::PTYPE)
                });
                Some(*vacant.insert(next_code).get())
            }
        }
    }

    fn encode_bytes<A: ArrayAccessor<[u8]>>(
        &mut self,
        accessor: &A,
        len: usize,
    ) -> VortexResult<ArrayRef> {
        let mut local_lookup = self.lookup.take().vortex_expect("Must have a lookup dict");
        let mut codes: BufferMut<Code> = BufferMut::with_capacity(len);

        let (codes, validity) = if self.dtype.is_nullable() {
            let mut null_buf = NullBufferBuilder::new(len);

            accessor.with_iterator(|it| {
                for value in it {
                    let (code, validity) = match value {
                        Some(v) => match self.encode_value(&mut local_lookup, v) {
                            Some(code) => (code, true),
                            None => break,
                        },
                        None => (Code::zero(), false),
                    };
                    null_buf.append(validity);
                    unsafe { codes.push_unchecked(code) }
                }
            })?;
            (
                codes,
                null_buf
                    .finish()
                    .map(Validity::from)
                    .unwrap_or(Validity::AllValid),
            )
        } else {
            accessor.with_iterator(|it| {
                for value in it {
                    let Some(code) = self.encode_value(
                        &mut local_lookup,
                        value.vortex_expect("Dict encode null value in non-nullable array"),
                    ) else {
                        break;
                    };
                    unsafe { codes.push_unchecked(code) }
                }
            })?;
            (codes, Validity::NonNullable)
        };

        // Restore lookup dictionary back into the struct
        self.lookup = Some(local_lookup);
        Ok(PrimitiveArray::new(codes, validity).into_array())
    }
}

impl<Code: Unsigned + AsPrimitive<usize> + NativePType> DictEncoder for BytesDictBuilder<Code> {
    fn encode(&mut self, array: &dyn Array) -> VortexResult<ArrayRef> {
        if &self.dtype != array.dtype() {
            vortex_bail!(
                "Array DType {} does not match builder dtype {}",
                array.dtype(),
                self.dtype
            );
        }

        let len = array.len();
        if let Some(varbinview) = array.as_opt::<VarBinViewVTable>() {
            self.encode_bytes(varbinview, len)
        } else if let Some(varbin) = array.as_opt::<VarBinVTable>() {
            self.encode_bytes(varbin, len)
        } else {
            vortex_bail!("Can only dictionary encode VarBin and VarBinView arrays");
        }
    }

    fn values(&mut self) -> VortexResult<ArrayRef> {
        VarBinViewArray::try_new(
            self.views.clone().freeze(),
            vec![self.values.clone().freeze()],
            self.dtype.clone(),
            self.dtype.nullability().into(),
        )
        .map(|a| a.into_array())
    }
}

#[cfg(test)]
mod test {
    use std::str;

    use vortex_array::ToCanonical;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::VarBinArray;

    use crate::builders::dict_encode;

    #[test]
    fn encode_varbin() {
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "again", "world"]);
        let dict = dict_encode(arr.as_ref()).unwrap();
        assert_eq!(
            dict.codes().to_primitive().unwrap().as_slice::<u8>(),
            &[0, 1, 0, 2, 1]
        );
        dict.values()
            .to_varbinview()
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
            dict.codes().to_primitive().unwrap().as_slice::<u8>(),
            &[0, 0, 1, 0, 0, 2, 1, 0]
        );
        dict.values()
            .to_varbinview()
            .unwrap()
            .with_iterator(|iter| {
                assert_eq!(
                    iter.map(|b| b.map(|v| unsafe { str::from_utf8_unchecked(v) }))
                        .collect::<Vec<_>>(),
                    vec![Some("hello"), Some("world"), Some("again")]
                );
            })
            .unwrap();
    }

    #[test]
    fn repeated_values() {
        let arr = VarBinArray::from(vec!["a", "a", "b", "b", "a", "b", "a", "b"]);
        let dict = dict_encode(arr.as_ref()).unwrap();
        dict.values()
            .to_varbinview()
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
            dict.codes().to_primitive().unwrap().as_slice::<u8>(),
            &[0, 0, 1, 1, 0, 1, 0, 1]
        );
    }
}
