// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::BuildHasher;
use std::mem;
use std::sync::Arc;

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;
use vortex_utils::aliases::hash_map::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;
use vortex_utils::aliases::hash_map::RandomState;

use super::DictConstraints;
use super::DictEncoder;
use crate::ArrayRef;
use crate::IntoArray;
use crate::accessor::ArrayAccessor;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::build_views::BinaryView;
use crate::canonical::ToCanonical;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::validity::Validity;

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub struct BytesDictBuilder<Codes> {
    lookup: Option<HashTable<Codes>>,
    views: BufferMut<BinaryView>,
    values: ByteBufferMut,
    values_nulls: BitBufferMut,
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

impl<Code: UnsignedPType> BytesDictBuilder<Code> {
    pub fn new(dtype: DType, constraints: &DictConstraints) -> Self {
        Self {
            lookup: Some(HashTable::new()),
            views: BufferMut::<BinaryView>::empty(),
            values: BufferMut::empty(),
            values_nulls: BitBufferMut::empty(),
            hasher: DefaultHashBuilder::default(),
            dtype,
            max_dict_bytes: constraints.max_bytes,
            max_dict_len: constraints.max_len,
        }
    }

    fn dict_bytes(&self) -> usize {
        self.views.len() * size_of::<BinaryView>() + self.values.len()
    }

    fn lookup_bytes(&self, idx: usize) -> Option<&[u8]> {
        self.values_nulls.value(idx).then(|| {
            let bin_view = &self.views[idx];
            if bin_view.is_inlined() {
                bin_view.as_inlined().value()
            } else {
                &self.values[bin_view.as_view().as_range()]
            }
        })
    }

    fn encode_value(&mut self, lookup: &mut HashTable<Code>, val: Option<&[u8]>) -> Option<Code> {
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
                match val {
                    None => {
                        // Null value
                        self.views.push(BinaryView::default());
                        self.values_nulls.append_false();
                    }
                    Some(val) => {
                        let view = BinaryView::make_view(
                            val,
                            0,
                            u32::try_from(self.values.len())
                                .vortex_expect("values length must fit in u32"),
                        );
                        let additional_bytes = if view.is_inlined() {
                            size_of::<BinaryView>()
                        } else {
                            size_of::<BinaryView>() + val.len()
                        };

                        if self.dict_bytes() + additional_bytes > self.max_dict_bytes {
                            return None;
                        }

                        self.views.push(view);
                        self.values_nulls.append_true();
                        if !view.is_inlined() {
                            self.values.extend_from_slice(val);
                        }
                    }
                }

                let next_code = Code::from_usize(next_code).unwrap_or_else(|| {
                    vortex_panic!("{next_code} has to fit into {}", Code::PTYPE)
                });
                Some(*vacant.insert(next_code).get())
            }
        }
    }

    fn encode_bytes<A: ArrayAccessor<[u8]>>(&mut self, accessor: &A, len: usize) -> ArrayRef {
        let mut local_lookup = self.lookup.take().vortex_expect("Must have a lookup dict");
        let mut codes: BufferMut<Code> = BufferMut::with_capacity(len);

        accessor.with_iterator(|it| {
            for value in it {
                let Some(code) = self.encode_value(&mut local_lookup, value) else {
                    break;
                };
                // SAFETY: we reserved capacity in the buffer for `len` elements
                unsafe { codes.push_unchecked(code) }
            }
        });

        // Restore lookup dictionary back into the struct
        self.lookup = Some(local_lookup);

        PrimitiveArray::new(codes, Validity::NonNullable).into_array()
    }
}

impl<Code: UnsignedPType> DictEncoder for BytesDictBuilder<Code> {
    fn encode(&mut self, array: &ArrayRef) -> ArrayRef {
        debug_assert_eq!(
            &self.dtype,
            array.dtype(),
            "Array DType {} does not match builder dtype {}",
            array.dtype(),
            self.dtype
        );

        let len = array.len();
        if let Some(varbinview) = array.as_opt::<VarBinView>() {
            self.encode_bytes(&varbinview.into_owned(), len)
        } else if let Some(varbin) = array.as_opt::<VarBin>() {
            self.encode_bytes(&varbin.into_owned(), len)
        } else {
            // NOTE(aduffy): it is very rare that this path would be taken, only e.g.
            //  if we're performing dictionary encoding downstream of some other compression.
            self.encode_bytes(&array.to_varbinview(), len)
        }
    }

    fn reset(&mut self) -> ArrayRef {
        let views = mem::take(&mut self.views).freeze();
        let buffer = mem::take(&mut self.values).freeze();
        let value_nulls = mem::take(&mut self.values_nulls).freeze();

        // SAFETY: we build the views explicitly and the bytes should be checked before feeding
        //  to the encoder.
        unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from([buffer]),
                self.dtype.clone(),
                Validity::from_bit_buffer(value_nulls, self.dtype.nullability()),
            )
            .into_array()
        }
    }

    fn codes_ptype(&self) -> PType {
        Code::PTYPE
    }
}

#[cfg(test)]
mod test {
    use std::str;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::VarBinArray;
    use crate::builders::dict::dict_encode;

    #[test]
    fn encode_varbin() {
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "again", "world"]);
        let dict = dict_encode(&arr.into_array()).unwrap();
        assert_eq!(
            dict.codes().to_primitive().as_slice::<u8>(),
            &[0, 1, 0, 2, 1]
        );
        dict.values().to_varbinview().with_iterator(|iter| {
            assert_eq!(
                iter.flatten()
                    .map(|b| unsafe { str::from_utf8_unchecked(b) })
                    .collect::<Vec<_>>(),
                vec!["hello", "world", "again"]
            );
        });
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
        let dict = dict_encode(&arr.into_array()).unwrap();
        assert_eq!(
            dict.codes().to_primitive().as_slice::<u8>(),
            &[0, 1, 2, 0, 1, 3, 2, 1]
        );
        dict.values().to_varbinview().with_iterator(|iter| {
            assert_eq!(
                iter.map(|b| b.map(|v| unsafe { str::from_utf8_unchecked(v) }))
                    .collect::<Vec<_>>(),
                vec![Some("hello"), None, Some("world"), Some("again")]
            );
        });
    }

    #[test]
    fn repeated_values() {
        let arr = VarBinArray::from(vec!["a", "a", "b", "b", "a", "b", "a", "b"]);
        let dict = dict_encode(&arr.into_array()).unwrap();
        dict.values().to_varbinview().with_iterator(|iter| {
            assert_eq!(
                iter.flatten()
                    .map(|b| unsafe { str::from_utf8_unchecked(b) })
                    .collect::<Vec<_>>(),
                vec!["a", "b"]
            );
        });
        assert_eq!(
            dict.codes().to_primitive().as_slice::<u8>(),
            &[0, 0, 1, 1, 0, 1, 0, 1]
        );
    }
}
