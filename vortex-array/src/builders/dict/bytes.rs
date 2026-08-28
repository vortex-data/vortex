// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::OnceCell;
use std::hash::BuildHasher;
use std::mem;
use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_array::ExecutionCtx;
use vortex_buffer::Alignment;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashTable;
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
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbinview::build_views::BinaryView;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// The value's length, held in the low four bytes of every view.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the low four bytes of a view are its length"
)]
fn view_len(raw: u128) -> usize {
    raw as u32 as usize
}

/// Canonicalize an inlined view by zeroing the padding that follows the value.
///
/// A value short enough to live inside its view is described entirely by the canonical view: the
/// hot loop hashes and compares it as a single 16-byte integer, and the dictionary entry is that
/// integer. Views this encoder builds are already canonical, but incoming arrays may carry
/// arbitrary bytes in the padding, so they are normalized before being hashed or compared.
#[inline]
fn inlined_key(raw: u128, len: usize) -> u128 {
    debug_assert!(len <= BinaryView::MAX_INLINED_SIZE);
    // Keep the four length bytes plus `len` bytes of value.
    raw & (u128::MAX >> (8 * (BinaryView::MAX_INLINED_SIZE - len)))
}

/// Build the canonical inlined key for a value held as a plain byte slice.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "callers only pass values that fit inline, so the length is at most 12"
)]
fn inlined_key_from_bytes(val: &[u8]) -> u128 {
    debug_assert!(val.len() <= BinaryView::MAX_INLINED_SIZE);
    let mut le_bytes = [0u8; 16];
    le_bytes[..4].copy_from_slice(&(val.len() as u32).to_le_bytes());
    le_bytes[4..4 + val.len()].copy_from_slice(val);
    u128::from_le_bytes(le_bytes)
}

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
    allocator: BufferAllocatorRef,
}

pub fn bytes_dict_builder(
    dtype: DType,
    constraints: &DictConstraints,
    allocator: BufferAllocatorRef,
) -> Box<dyn DictEncoder> {
    match constraints.max_len as u64 {
        max if max <= u8::MAX as u64 => Box::new(BytesDictBuilder::<u8>::new_in(
            dtype,
            constraints,
            allocator,
        )),
        max if max <= u16::MAX as u64 => Box::new(BytesDictBuilder::<u16>::new_in(
            dtype,
            constraints,
            allocator,
        )),
        max if max <= u32::MAX as u64 => Box::new(BytesDictBuilder::<u32>::new_in(
            dtype,
            constraints,
            allocator,
        )),
        _ => Box::new(BytesDictBuilder::<u64>::new_in(
            dtype,
            constraints,
            allocator,
        )),
    }
}

impl<Code: UnsignedPType> BytesDictBuilder<Code> {
    pub fn new_in(
        dtype: DType,
        constraints: &DictConstraints,
        allocator: BufferAllocatorRef,
    ) -> Self {
        Self {
            lookup: Some(HashTable::new()),
            views: BufferMut::<BinaryView>::empty_aligned_in(
                Alignment::of::<BinaryView>(),
                allocator.clone(),
            ),
            null_code: OnceCell::new(),
            values: BufferMut::empty_aligned_in(Alignment::of::<u8>(), allocator.clone()),
            values_nulls: BitBufferMut::empty_in(allocator.clone()),
            hasher: DefaultHashBuilder::default(),
            dtype,
            max_dict_bytes: constraints.max_bytes.min(u32::MAX as usize),
            max_dict_len: constraints.max_len,
            allocator,
        }
    }

    fn dict_bytes(&self) -> usize {
        self.views.len() * size_of::<BinaryView>() + self.values.len()
    }

    /// Rehash a dictionary entry, matching the hash the entry was inserted with.
    fn hash_entry(&self, idx: usize) -> u64 {
        let view = self.views[idx];
        if view.is_inlined() {
            self.hasher.hash_one(view.as_u128())
        } else {
            self.hasher
                .hash_one(&self.values[view.as_view().as_range()])
        }
    }

    /// Append `view` to the dictionary and return the code that now addresses it.
    fn push_view(&mut self, view: BinaryView) -> Code {
        let code = self.views.len();
        self.views.push(view);
        self.values_nulls.append_true();
        Code::from_usize(code)
            .unwrap_or_else(|| vortex_panic!("{code} has to fit into {}", Code::PTYPE))
    }

    /// Encode a value short enough to live inside its view, where the key is the whole value and
    /// equality is a single 16-byte comparison against the dictionary's own view.
    ///
    /// Returns `None` when assigning a code would exceed the dictionary constraints, and callers
    /// should stop encoding after the current prefix.
    #[inline]
    fn encode_inlined(&mut self, lookup: &mut HashTable<Code>, key: u128) -> Option<Code> {
        let hash = self.hasher.hash_one(key);
        match lookup.find(hash, |idx| self.views[idx.as_()].as_u128() == key) {
            Some(&code) => Some(code),
            None => self.insert_inlined(lookup, hash, key),
        }
    }

    /// [`Self::encode_inlined`] for a value held inline in an incoming view.
    ///
    /// The per-row loops branch between inlined and longer values. Keeping the inlined lookup
    /// out of line keeps those loops small, which measurably speeds up the longer values at the
    /// cost of a call for the inlined ones.
    #[inline(never)]
    fn encode_inlined_view(
        &mut self,
        lookup: &mut HashTable<Code>,
        raw: u128,
        len: usize,
    ) -> Option<Code> {
        self.encode_inlined(lookup, inlined_key(raw, len))
    }

    /// [`Self::encode_inlined`] for a value held as a plain byte slice; out of line for the same
    /// reason as [`Self::encode_inlined_view`].
    #[inline(never)]
    fn encode_inlined_bytes(&mut self, lookup: &mut HashTable<Code>, val: &[u8]) -> Option<Code> {
        self.encode_inlined(lookup, inlined_key_from_bytes(val))
    }

    #[cold]
    fn insert_inlined(
        &mut self,
        lookup: &mut HashTable<Code>,
        hash: u64,
        key: u128,
    ) -> Option<Code> {
        if self.views.len() >= self.max_dict_len
            || self.dict_bytes() + size_of::<BinaryView>() > self.max_dict_bytes
        {
            return None;
        }
        let code = self.push_view(BinaryView::from(key));
        lookup.insert_unique(hash, code, |idx| self.hash_entry(idx.as_()));
        Some(code)
    }

    /// Encode a value too long to inline, whose bytes live on the dictionary's value heap.
    ///
    /// The length filters out most candidate entries - inlined ones always, since their length
    /// cannot reach `len` - before the value heap is read.
    ///
    /// Returns `None` when assigning a code would exceed the dictionary constraints, and callers
    /// should stop encoding after the current prefix.
    #[inline]
    fn encode_referenced(
        &mut self,
        lookup: &mut HashTable<Code>,
        len: usize,
        val: &[u8],
    ) -> Option<Code> {
        let hash = self.hasher.hash_one(val);
        let found = lookup.find(hash, |idx| {
            let view = self.views[idx.as_()];
            view.len() as usize == len && self.values[view.as_view().as_range()] == *val
        });
        match found {
            Some(&code) => Some(code),
            None => self.insert_referenced(lookup, hash, val),
        }
    }

    #[cold]
    fn insert_referenced(
        &mut self,
        lookup: &mut HashTable<Code>,
        hash: u64,
        val: &[u8],
    ) -> Option<Code> {
        if self.views.len() >= self.max_dict_len
            || self.dict_bytes() + size_of::<BinaryView>() + val.len() > self.max_dict_bytes
        {
            return None;
        }
        let offset =
            u32::try_from(self.values.len()).vortex_expect("values length must fit in u32");
        let view = BinaryView::make_view(val, 0, offset);
        self.values.extend_from_slice(val);
        let code = self.push_view(view);
        lookup.insert_unique(hash, code, |idx| self.hash_entry(idx.as_()));
        Some(code)
    }

    /// Returns `None` when assigning the null code would exceed the dictionary constraints,
    /// and callers should stop encoding after the current prefix.
    fn encode_null(&mut self) -> Option<Code> {
        if let Some(code) = self.null_code.get() {
            return Some(*code);
        }

        if self.views.len() >= self.max_dict_len
            || self.dict_bytes() + size_of::<BinaryView>() > self.max_dict_bytes
        {
            return None;
        }

        let code = self.views.len();
        self.views.push(BinaryView::default());
        self.values_nulls.append_false();
        let code = Code::from_usize(code)
            .unwrap_or_else(|| vortex_panic!("{} has to fit into {}", code, Code::PTYPE));
        self.null_code
            .set(code)
            .ok()
            .vortex_expect("null code is initialized once");
        Some(code)
    }

    /// Encode row values against the dictionary, honoring the supplied validity mask.
    ///
    /// `encode_at` is called only for valid rows. That matters for VarBinView arrays because null
    /// rows can hold arbitrary view metadata. It receives the builder rather than capturing it so
    /// that each caller can dispatch straight to the encoder for its value representation.
    fn encode_validity<F>(
        &mut self,
        len: usize,
        validity_mask: Mask,
        mut encode_at: F,
    ) -> VortexResult<PrimitiveArray>
    where
        F: FnMut(&mut Self, &mut HashTable<Code>, usize) -> Option<Code>,
    {
        let mut local_lookup = self.lookup.take().vortex_expect("Must have a lookup dict");
        let mut codes = BufferMut::<Code>::with_capacity_in(len, self.allocator.clone());

        match validity_mask.bit_buffer() {
            AllOr::All => {
                for idx in 0..len {
                    let Some(code) = encode_at(self, &mut local_lookup, idx) else {
                        break;
                    };
                    // SAFETY: we reserved capacity in the buffer for `len` elements
                    unsafe { codes.push_unchecked(code) }
                }
            }
            AllOr::None => {
                if let Some(code) = self.encode_null() {
                    unsafe { codes.push_n_unchecked(code, len) }
                }
            }
            AllOr::Some(b) => {
                for (idx, valid) in b.iter().enumerate() {
                    if !valid {
                        let Some(code) = self.encode_null() else {
                            break;
                        };
                        // SAFETY: we reserved capacity in the buffer for `len` elements
                        unsafe { codes.push_unchecked(code) }
                    } else {
                        let Some(code) = encode_at(self, &mut local_lookup, idx) else {
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
            self.encode_validity(len, validity_mask, |this, lookup, idx| {
                let start: usize = slice_offsets[idx].as_();
                let end: usize = slice_offsets[idx + 1].as_();
                let val = &bytes[start..end];
                if val.len() <= BinaryView::MAX_INLINED_SIZE {
                    this.encode_inlined_bytes(lookup, val)
                } else {
                    this.encode_referenced(lookup, val.len(), val)
                }
            })
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
            .map(|b| b.as_host().as_slice())
            .collect::<Vec<_>>();

        self.encode_validity(len, validity_mask, |this, lookup, idx| {
            let view = views[idx];
            let raw = view.as_u128();
            let val_len = view_len(raw);
            if val_len <= BinaryView::MAX_INLINED_SIZE {
                this.encode_inlined_view(lookup, raw, val_len)
            } else {
                let reference = view.as_view();
                let buffer = buffers[reference.buffer_index as usize];
                this.encode_referenced(lookup, val_len, &buffer[reference.as_range()])
            }
        })
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
        if let Some(lookup) = self.lookup.as_mut() {
            lookup.clear();
        }
        self.null_code = OnceCell::new();
        let views = mem::replace(
            &mut self.views,
            BufferMut::empty_aligned_in(Alignment::of::<BinaryView>(), self.allocator.clone()),
        )
        .freeze();
        let buffer = mem::replace(
            &mut self.values,
            BufferMut::empty_aligned_in(Alignment::of::<u8>(), self.allocator.clone()),
        )
        .freeze();
        let value_nulls = mem::replace(
            &mut self.values_nulls,
            BitBufferMut::empty_in(self.allocator.clone()),
        )
        .freeze();

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
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::BufferAllocatorRef;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::BytesDictBuilder;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::varbinview::BinaryView;
    use crate::assert_arrays_eq;
    use crate::buffer::BufferHandle;
    use crate::builders::dict::UNCONSTRAINED;
    use crate::builders::dict::dict_encode;
    use crate::builders::dict::dict_encoder_in;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    #[test]
    fn encode_varbin() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter_str(vec!["hello", "world", "hello", "again", "world"]);
        let mut ctx = SESSION.create_execution_ctx();
        let dict = dict_encode(&arr.into_array(), &mut ctx)?;
        let codes = dict.codes().clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(codes.as_slice::<u8>(), &[0, 1, 0, 2, 1]);
        let values = dict.values().clone().execute::<VarBinViewArray>(&mut ctx)?;
        let mask = values.validity()?.execute_mask(values.len(), &mut ctx)?;
        let decoded = (0..values.len())
            .filter(|&i| mask.value(i))
            .map(|i| unsafe { String::from_utf8_unchecked(values.bytes_at(i).to_vec()) })
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec!["hello", "world", "again"]);
        Ok(())
    }

    #[test]
    fn encode_varbin_nulls() -> VortexResult<()> {
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
        let mut ctx = SESSION.create_execution_ctx();
        let dict = dict_encode(&arr.into_array(), &mut ctx)?;
        let codes = dict.codes().clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(codes.as_slice::<u8>(), &[0, 1, 2, 0, 1, 3, 2, 1]);
        let values = dict.values().clone().execute::<VarBinViewArray>(&mut ctx)?;
        let mask = values.validity()?.execute_mask(values.len(), &mut ctx)?;
        let decoded = (0..values.len())
            .map(|i| {
                mask.value(i)
                    .then(|| unsafe { String::from_utf8_unchecked(values.bytes_at(i).to_vec()) })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            decoded,
            vec![
                Some("hello".to_string()),
                None,
                Some("world".to_string()),
                Some("again".to_string())
            ]
        );
        Ok(())
    }

    #[test]
    fn encode_varbinview_ignores_invalid_null_views() {
        let value = b"outlined value";
        let valid_view = BinaryView::make_view(value, 0, 0);
        let invalid_null_view = BinaryView::make_view(b"invalid null view", 99, 0);
        let views = Buffer::copy_from([valid_view, invalid_null_view, valid_view]);
        let buffers = Arc::from([BufferHandle::new_host(ByteBuffer::copy_from(value))]);
        let arr = unsafe {
            VarBinViewArray::new_handle_unchecked(
                BufferHandle::new_host(views.into_byte_buffer()),
                buffers,
                DType::Utf8(Nullability::Nullable),
                Validity::from_iter([true, false, true]),
            )
        }
        .into_array();

        let dict = dict_encode(&arr, &mut SESSION.create_execution_ctx()).unwrap();
        let codes = dict
            .codes()
            .clone()
            .execute::<PrimitiveArray>(&mut SESSION.create_execution_ctx())
            .unwrap();
        assert_eq!(codes.as_slice::<u8>(), &[0, 1, 0]);
    }

    #[test]
    fn repeated_values() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["a", "a", "b", "b", "a", "b", "a", "b"]);
        let mut ctx = SESSION.create_execution_ctx();
        let dict = dict_encode(&arr.into_array(), &mut ctx)?;
        let values = dict.values().clone().execute::<VarBinViewArray>(&mut ctx)?;
        let mask = values.validity()?.execute_mask(values.len(), &mut ctx)?;
        let decoded = (0..values.len())
            .filter(|&i| mask.value(i))
            .map(|i| unsafe { String::from_utf8_unchecked(values.bytes_at(i).to_vec()) })
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec!["a", "b"]);
        let codes = dict.codes().clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(codes.as_slice::<u8>(), &[0, 0, 1, 1, 0, 1, 0, 1]);
        Ok(())
    }

    #[test]
    fn reset_clears_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let first = VarBinViewArray::from_iter_str(["one", "two"]).into_array();
        let mut encoder = dict_encoder_in(&first, &UNCONSTRAINED, ctx.allocator().clone());

        assert_arrays_eq!(
            encoder.encode(&first, &mut ctx)?,
            PrimitiveArray::from_iter([0u64, 1]),
            &mut ctx
        );
        assert_arrays_eq!(encoder.reset(), first, &mut ctx);

        let second = VarBinViewArray::from_iter_str(["one", "three"]).into_array();
        assert_arrays_eq!(
            encoder.encode(&second, &mut ctx)?,
            PrimitiveArray::from_iter([0u64, 1]),
            &mut ctx
        );
        assert_arrays_eq!(encoder.reset(), second, &mut ctx);

        Ok(())
    }

    #[test]
    fn encode_varbinview_dedupes_noncanonical_inline_padding() -> VortexResult<()> {
        // The bytes past an inlined value are padding; producers are expected to zero them, but a
        // view carrying junk there still describes the same value and must share its code.
        let canonical = BinaryView::make_view(b"hello", 0, 0);
        let padded = BinaryView::from(canonical.as_u128() | (u128::MAX << (8 * (4 + 5))));
        let views = Buffer::copy_from([canonical, padded, canonical]);
        let arr = unsafe {
            VarBinViewArray::new_handle_unchecked(
                BufferHandle::new_host(views.into_byte_buffer()),
                Arc::<[BufferHandle]>::from(vec![]),
                DType::Utf8(Nullability::NonNullable),
                Validity::NonNullable,
            )
        }
        .into_array();

        let mut ctx = SESSION.create_execution_ctx();
        let dict = dict_encode(&arr, &mut ctx)?;
        let codes = dict.codes().clone().execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(codes.as_slice::<u8>(), &[0, 0, 0]);
        assert_eq!(dict.values().len(), 1);
        Ok(())
    }

    #[test]
    fn max_dict_bytes_cannot_exceed_the_view_offset_range() {
        let builder = BytesDictBuilder::<u32>::new_in(
            DType::Utf8(Nullability::NonNullable),
            &UNCONSTRAINED,
            BufferAllocatorRef::statically_allocated(),
        );
        assert_eq!(builder.max_dict_bytes, u32::MAX as usize);
    }
}
