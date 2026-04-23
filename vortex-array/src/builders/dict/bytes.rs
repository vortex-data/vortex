// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::OnceCell;
use std::hash::BuildHasher;
use std::mem;
use std::sync::Arc;

use itertools::Itertools;
use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;
use vortex_utils::aliases::hash_map::RandomState;

use super::DictConstraints;
use super::DictEncoder;
use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbinview::build_views::BinaryView;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// Dictionary encode varbin array. Specializes for primitive byte arrays to avoid double copying
pub struct BytesDictBuilder<Code> {
    lookup: Option<HashTable<Code>>,
    null_code: OnceCell<Code>,
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
            null_code: OnceCell::new(),
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

    fn lookup_bytes(&self, idx: usize) -> &[u8] {
        let bin_view = &self.views[idx];
        if bin_view.is_inlined() {
            bin_view.as_inlined().value()
        } else {
            &self.values[bin_view.as_view().as_range()]
        }
    }

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
                let view = BinaryView::make_view(
                    val,
                    0,
                    u32::try_from(self.values.len()).vortex_expect("values length must fit in u32"),
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

                let next_code = Code::from_usize(next_code).unwrap_or_else(|| {
                    vortex_panic!("{next_code} has to fit into {}", Code::PTYPE)
                });
                Some(*vacant.insert(next_code).get())
            }
        }
    }

    /// Encode a stream of value bytes against the dictionary, honoring the supplied validity mask.
    ///
    /// `values` must yield one slice per logical row in input order; the mask is applied here so
    /// callers do not need to emit anything for null positions.
    fn encode_iter<'a, I>(
        &mut self,
        len: usize,
        validity_mask: Mask,
        values: I,
    ) -> VortexResult<PrimitiveArray>
    where
        I: Iterator<Item = &'a [u8]>,
    {
        let mut local_lookup = self.lookup.take().vortex_expect("Must have a lookup dict");
        let mut codes: BufferMut<Code> = BufferMut::with_capacity(len);

        match validity_mask.bit_buffer() {
            AllOr::All => {
                for value in values {
                    let Some(code) = self.encode_value(&mut local_lookup, value) else {
                        break;
                    };
                    // SAFETY: we reserved capacity in the buffer for `len` elements
                    unsafe { codes.push_unchecked(code) }
                }
            }
            AllOr::None => {
                self.views.push(BinaryView::default());
                self.values_nulls.append_false();
                unsafe {
                    codes.push_n_unchecked(Code::from_usize(0).vortex_expect("must fit 0"), len)
                }
            }
            AllOr::Some(b) => {
                for (value, valid) in values.zip_eq(b.iter()) {
                    if !valid {
                        let code = self.null_code.get_or_init(|| {
                            let code = self.views.len();
                            self.views.push(BinaryView::default());
                            self.values_nulls.append_false();
                            Code::from_usize(code).unwrap_or_else(|| {
                                vortex_panic!("{} has to fit into {}", code, Code::PTYPE)
                            })
                        });
                        // SAFETY: we reserved capacity in the buffer for `len` elements
                        unsafe { codes.push_unchecked(*code) }
                    } else {
                        let Some(code) = self.encode_value(&mut local_lookup, value) else {
                            break;
                        };
                        // SAFETY: we reserved capacity in the buffer for `len` elements
                        unsafe { codes.push_unchecked(code) }
                    }
                }
            }
        }

        // Restore lookup dictionary back into the struct
        self.lookup = Some(local_lookup);

        Ok(PrimitiveArray::new(codes, Validity::NonNullable))
    }

    fn encode_varbin(
        &mut self,
        var_bin: ArrayView<VarBin>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let offsets = var_bin.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let bytes = var_bin.bytes();
        let validity_mask = var_bin.validity()?.execute_mask(var_bin.len(), ctx)?;
        let len = var_bin.len();

        match_each_integer_ptype!(offsets.ptype(), |P| {
            let slice_offsets = offsets.as_slice::<P>();
            let values = slice_offsets.windows(2).map(|w| {
                let start: usize = w[0].as_();
                let end: usize = w[1].as_();
                &bytes[start..end]
            });
            self.encode_iter(len, validity_mask, values)
        })
    }

    fn encode_varbinview(
        &mut self,
        var_bin_view: ArrayView<VarBinView>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        let validity_mask = var_bin_view
            .validity()?
            .execute_mask(var_bin_view.len(), ctx)?;
        let len = var_bin_view.len();
        let views = var_bin_view.views();
        let buffers = var_bin_view
            .data_buffers()
            .iter()
            .map(|b| b.as_host())
            .collect::<Vec<_>>();

        let values = views.iter().map(|view| {
            if view.is_inlined() {
                view.as_inlined().value()
            } else {
                &buffers[view.as_view().buffer_index as usize][view.as_view().as_range()]
            }
        });
        self.encode_iter(len, validity_mask, values)
    }
}

impl<Code: UnsignedPType> DictEncoder for BytesDictBuilder<Code> {
    fn encode(&mut self, array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
        debug_assert_eq!(
            &self.dtype,
            array.dtype(),
            "Array DType {} does not match builder dtype {}",
            array.dtype(),
            self.dtype
        );

        if let Some(varbinview) = array.as_opt::<VarBinView>() {
            self.encode_varbinview(varbinview, ctx)
        } else if let Some(varbin) = array.as_opt::<VarBin>() {
            self.encode_varbin(varbin, ctx)
        } else {
            // NOTE(aduffy): it is very rare that this path would be taken, only e.g.
            //  if we're performing dictionary encoding downstream of some other compression.
            let vbv_array = array.clone().execute::<VarBinViewArray>(ctx)?;
            self.encode_varbinview(vbv_array.as_view(), ctx)
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
    use std::sync::LazyLock;

    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::accessor::ArrayAccessor;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::builders::dict::dict_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    #[test]
    fn encode_varbin() {
        let arr = VarBinViewArray::from_iter_str(vec!["hello", "world", "hello", "again", "world"]);
        let dict = dict_encode(&arr.into_array(), &mut SESSION.create_execution_ctx()).unwrap();
        let codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(codes.as_slice::<u8>(), &[0, 1, 0, 2, 1]);
        let values = dict
            .values()
            .clone()
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        values.with_iterator(|iter| {
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
        let arr: VarBinViewArray = vec![
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
        let dict = dict_encode(&arr.into_array(), &mut SESSION.create_execution_ctx()).unwrap();
        let codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(codes.as_slice::<u8>(), &[0, 1, 2, 0, 1, 3, 2, 1]);
        let values = dict
            .values()
            .clone()
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        values.with_iterator(|iter| {
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
        let dict = dict_encode(&arr.into_array(), &mut SESSION.create_execution_ctx()).unwrap();
        let values = dict
            .values()
            .clone()
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        values.with_iterator(|iter| {
            assert_eq!(
                iter.flatten()
                    .map(|b| unsafe { str::from_utf8_unchecked(b) })
                    .collect::<Vec<_>>(),
                vec!["a", "b"]
            );
        });
        let codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(codes.as_slice::<u8>(), &[0, 0, 1, 1, 0, 1, 0, 1]);
    }
}
