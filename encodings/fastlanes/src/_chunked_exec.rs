// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bit-pack–aware chunked decoders that integrate with `vortex_array::_chunked_exec`.
//!
//! Avoids the full upfront bit-unpack the canonical executor performs. For a
//! `Dict<BitPacked<P>, …>` array we unpack one 1024-element code chunk at a time and
//! immediately AVX2-gather it into the output buffer. The working set is the small
//! values dictionary plus the 4 KiB chunk-of-codes, never the materialised codes column.

use std::mem::MaybeUninit;
use std::sync::Arc;

use fastlanes::BitPacking;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::_chunked_exec::CHUNK_LEN;
use vortex_array::_chunked_exec::Scratch;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernel;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernelDispatcher;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkProducer;
use vortex_array::_chunked_exec::take_into_uninit;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnsignedPType;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;

/// Chunked dict decoder where the codes child is bit-packed.
///
/// `dict` is the materialised values buffer (small, expected L1-resident). `packed` is
/// the bit-packed codes buffer. Each chunk: bit-unpack 1024 codes into an internal
/// `[I; CHUNK_LEN]` (one stack-resident buffer reused across chunks), then AVX2-gather
/// them into the output via [`take_into_uninit`].
pub struct BitPackedDictProducer<T: NativePType, I: NativePType + UnsignedPType + BitPacking> {
    dict: Buffer<T>,
    packed: ByteBuffer,
    bit_width: usize,
    elems_per_chunk: usize,
    /// Number of *full* 1024-element chunks in the packed buffer.
    full_chunks: usize,
    /// Length of the trailing partial chunk in elements (0 if perfectly aligned).
    trailer_len: usize,
    /// Total remaining logical elements to produce.
    remaining: usize,
    /// Current full-chunk index.
    chunk_idx: usize,
    /// Scratch for one chunk of unpacked codes (4–8 KiB depending on `I`).
    code_scratch: Box<[MaybeUninit<I>; CHUNK_LEN]>,
}

impl<T, I> BitPackedDictProducer<T, I>
where
    T: NativePType,
    I: NativePType + UnsignedPType + BitPacking,
{
    fn new(
        dict: Buffer<T>,
        packed: ByteBuffer,
        bit_width: usize,
        full_chunks: usize,
        trailer_len: usize,
    ) -> Self {
        let elems_per_chunk = 128 * bit_width / size_of::<I>();
        let total = full_chunks * CHUNK_LEN + trailer_len;
        Self {
            dict,
            packed,
            bit_width,
            elems_per_chunk,
            full_chunks,
            trailer_len,
            remaining: total,
            chunk_idx: 0,
            code_scratch: Box::new([const { MaybeUninit::<I>::uninit() }; CHUNK_LEN]),
        }
    }

    /// Bit-unpack fastlanes chunk `chunk_index` into `self.code_scratch[offset..offset+1024]`.
    ///
    /// # Safety
    ///
    /// `offset + 1024 ≤ code_scratch.len()` (the caller must ensure room for one fastlanes
    /// chunk starting at `offset`).
    unsafe fn unpack_chunk_into(&mut self, chunk_index: usize, offset: usize) {
        let packed_bytes = self.packed.as_ref();
        // SAFETY: same alignment used by fastlanes; bit_width-derived chunk layout.
        let packed_slice: &[I] = unsafe {
            std::slice::from_raw_parts(
                packed_bytes.as_ptr().cast::<I>(),
                packed_bytes.len() / size_of::<I>(),
            )
        };
        let chunk = &packed_slice[chunk_index * self.elems_per_chunk
            ..chunk_index * self.elems_per_chunk + self.elems_per_chunk];
        // SAFETY: caller ensures offset + 1024 ≤ scratch capacity.
        let dst_ptr = unsafe { self.code_scratch.as_mut_ptr().add(offset).cast::<I>() };
        let dst_slice: &mut [I] = unsafe { std::slice::from_raw_parts_mut(dst_ptr, 1024) };
        unsafe {
            BitPacking::unchecked_unpack(self.bit_width, chunk, dst_slice);
        }
    }
}

impl<T, I> PrimitiveChunkProducer<T> for BitPackedDictProducer<T, I>
where
    T: NativePType,
    I: NativePType + UnsignedPType + BitPacking,
{
    fn next_chunk<'a>(
        &mut self,
        scratch: &'a mut Scratch<T>,
    ) -> VortexResult<Option<&'a [T]>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let dst = &mut scratch.as_uninit_mut()[..CHUNK_LEN];
        let dst_ptr = dst.as_ptr().cast::<T>();
        let n = self.write_next_into(&mut dst[..])?;
        match n {
            Some(n) => Ok(Some(unsafe { std::slice::from_raw_parts(dst_ptr, n) })),
            None => Ok(None),
        }
    }

    fn next_chunk_into_uninit(
        &mut self,
        _scratch: &mut Scratch<T>,
        dst: &mut [MaybeUninit<T>],
    ) -> VortexResult<Option<usize>> {
        self.write_next_into(dst)
    }

    fn remaining(&self) -> usize {
        self.remaining
    }
}

impl<T, I> BitPackedDictProducer<T, I>
where
    T: NativePType,
    I: NativePType + UnsignedPType + BitPacking,
{
    fn write_next_into(&mut self, dst: &mut [MaybeUninit<T>]) -> VortexResult<Option<usize>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        // Unpack up to `CHUNK_LEN / FL_CHUNK = 4` fastlanes chunks into the local
        // code_scratch, then issue a single AVX-gather over the full super-chunk. This
        // amortises the per-chunk dispatch + gather-call overhead.
        const FL_CHUNK: usize = 1024;
        let super_chunk_max = CHUNK_LEN / FL_CHUNK;
        debug_assert!(super_chunk_max >= 1);
        let mut produced = 0usize;
        let dst_cap = dst.len();

        // Full fastlanes chunks first.
        while self.chunk_idx < self.full_chunks
            && produced + FL_CHUNK <= dst_cap
            && produced + FL_CHUNK <= CHUNK_LEN
        {
            // SAFETY: code_scratch has CHUNK_LEN cells; produced + FL_CHUNK ≤ CHUNK_LEN.
            unsafe {
                self.unpack_chunk_into(self.chunk_idx, produced);
            }
            self.chunk_idx += 1;
            produced += FL_CHUNK;
        }
        // Trailing partial chunk if there's room.
        if produced < dst_cap && produced < CHUNK_LEN && self.trailer_len > 0 {
            let trailer_take = self.trailer_len.min(dst_cap - produced).min(CHUNK_LEN - produced);
            // Unpack the trailing fastlanes chunk in full into scratch (its prefix is what we want).
            // SAFETY: code_scratch has space for CHUNK_LEN; produced + FL_CHUNK ≤ CHUNK_LEN.
            unsafe {
                self.unpack_chunk_into(self.chunk_idx, produced);
            }
            self.trailer_len -= trailer_take;
            if self.trailer_len == 0 {
                self.chunk_idx += 1;
            }
            produced += trailer_take;
        }

        if produced == 0 {
            return Ok(None);
        }

        // Single AVX gather over the full super-chunk.
        // SAFETY: code_scratch[..produced] is initialised by the unpack calls above.
        let codes = unsafe {
            std::slice::from_raw_parts(self.code_scratch.as_ptr().cast::<I>(), produced)
        };
        take_into_uninit::<T, I>(self.dict.as_slice(), codes, &mut dst[..produced]);
        self.remaining -= produced;
        Ok(Some(produced))
    }
}

/// Kernel that matches `Dict<…>` whose codes child is bit-packed.
///
/// Falls back to the in-crate canonical `DictKernel` for non-bit-packed codes or sliced
/// arrays (those still go through the AVX2 gather in `take_into_uninit`, just with a
/// canonicalised codes buffer).
pub struct BitPackedDictKernel<T: NativePType> {
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: NativePType> BitPackedDictKernel<T> {
    /// Construct a new kernel marker.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: NativePType> Default for BitPackedDictKernel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativePType> PrimitiveChunkKernel<T> for BitPackedDictKernel<T> {
    fn build(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn PrimitiveChunkProducer<T>>>> {
        let Some(dict) = array.as_opt::<Dict>() else {
            return Ok(None);
        };
        if !matches!(array.dtype().nullability(), Nullability::NonNullable) {
            return Ok(None);
        }
        let codes = dict.codes();
        let values = dict.values();
        let Some(bp) = codes.as_opt::<BitPacked>() else {
            return Ok(None);
        };
        // v1 fast path only handles non-sliced bit-packed inputs without patches.
        if bp.offset() != 0 || bp.patches().is_some() {
            return Ok(None);
        }
        if !matches!(codes.dtype().nullability(), Nullability::NonNullable)
            || !matches!(values.dtype().nullability(), Nullability::NonNullable)
        {
            return Ok(None);
        }
        let DType::Primitive(values_ptype, _) = *values.dtype() else {
            return Ok(None);
        };
        if values_ptype != T::PTYPE {
            return Ok(None);
        }
        let DType::Primitive(codes_ptype, _) = *codes.dtype() else {
            return Ok(None);
        };
        if !codes_ptype.is_unsigned_int() {
            return Ok(None);
        }

        let values_canonical = values.clone().execute::<PrimitiveArray>(ctx)?;
        let dict_buf = values_canonical.into_buffer::<T>();
        let len = codes.len();
        let packed = bp.packed().clone().unwrap_host();
        let bit_width = bp.bit_width() as usize;
        let full_chunks = len / CHUNK_LEN;
        let trailer_len = len % CHUNK_LEN;

        Ok(Some(match codes_ptype {
            PType::U8 => Box::new(BitPackedDictProducer::<T, u8>::new(
                dict_buf,
                packed,
                bit_width,
                full_chunks,
                trailer_len,
            )),
            PType::U16 => Box::new(BitPackedDictProducer::<T, u16>::new(
                dict_buf,
                packed,
                bit_width,
                full_chunks,
                trailer_len,
            )),
            PType::U32 => Box::new(BitPackedDictProducer::<T, u32>::new(
                dict_buf,
                packed,
                bit_width,
                full_chunks,
                trailer_len,
            )),
            PType::U64 => Box::new(BitPackedDictProducer::<T, u64>::new(
                dict_buf,
                packed,
                bit_width,
                full_chunks,
                trailer_len,
            )),
            _ => return Ok(None),
        }))
    }
}

/// Register the bit-packed chunked kernels onto `dispatcher` for every supported `T`.
pub fn register_chunk_kernels(dispatcher: &mut PrimitiveChunkKernelDispatcher) {
    macro_rules! register_all_for {
        ($($T:ty),*) => {
            $(
                // BitPackedDictKernel is registered LAST for `Dict.id()` so it tries first
                // (dispatcher iterates in registration order; bit-packed match short-circuits).
                dispatcher.register::<$T>(Dict.id(), Arc::new(BitPackedDictKernel::<$T>::new()));
            )*
        };
    }
    register_all_for!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernelDispatcher;
    use vortex_array::_chunked_exec::primitive::decode_to_buffer;
    use vortex_array::_chunked_exec::primitive::default_dispatcher;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::register_chunk_kernels;
    use crate::BitPackedData;

    fn session_dispatcher() -> (VortexSession, PrimitiveChunkKernelDispatcher) {
        let s = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&s);
        let mut d = default_dispatcher();
        register_chunk_kernels(&mut d);
        (s, d)
    }

    #[test]
    fn dict_bitpacked_codes_chunked() -> VortexResult<()> {
        let (session, dispatcher) = session_dispatcher();
        let mut ctx = session.create_execution_ctx();

        let dict_values = Buffer::<i32>::from_iter((0..256).map(|i| i as i32 * 7 + 11));
        let dict = PrimitiveArray::new(dict_values.clone(), Validity::NonNullable);

        // 4096 codes — 4 full 1024-chunks. Use u16 at 8 bits (bit_width must be < type width).
        let codes_vec: Vec<u16> = (0..4096u32).map(|i| (i % 256) as u16).collect();
        let codes_prim = PrimitiveArray::new(
            Buffer::<u16>::from_iter(codes_vec.iter().copied()),
            Validity::NonNullable,
        );
        let bp_codes = BitPackedData::encode(&codes_prim.into_array(), 8, &mut ctx)?;
        let dict_arr = DictArray::try_new(bp_codes.into_array(), dict.into_array())?;

        let buf = decode_to_buffer::<i32>(dict_arr.into_array(), &dispatcher, &mut ctx)?;
        let expected: Vec<i32> = codes_vec
            .iter()
            .map(|c| dict_values.as_slice()[*c as usize])
            .collect();
        assert_eq!(buf.as_slice(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn dict_bitpacked_codes_trailing_partial_chunk() -> VortexResult<()> {
        let (session, dispatcher) = session_dispatcher();
        let mut ctx = session.create_execution_ctx();
        let dict_values = Buffer::<i32>::from_iter([10, 20, 30, 40, 50]);
        let dict = PrimitiveArray::new(dict_values.clone(), Validity::NonNullable);
        // 1500 codes => 1 full chunk + 476 trailer. u16 at 3 bits fits 0..5.
        let codes_vec: Vec<u16> = (0..1500u32).map(|i| (i % 5) as u16).collect();
        let codes_prim = PrimitiveArray::new(
            Buffer::<u16>::from_iter(codes_vec.iter().copied()),
            Validity::NonNullable,
        );
        let bp_codes = BitPackedData::encode(&codes_prim.into_array(), 3, &mut ctx)?;
        let dict_arr = DictArray::try_new(bp_codes.into_array(), dict.into_array())?;
        let buf = decode_to_buffer::<i32>(dict_arr.into_array(), &dispatcher, &mut ctx)?;
        let expected: Vec<i32> = codes_vec
            .iter()
            .map(|c| dict_values.as_slice()[*c as usize])
            .collect();
        assert_eq!(buf.as_slice(), expected.as_slice());
        Ok(())
    }
}
