// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::buffers::{BufferHandle, ByteBufferHandle};
use crate::experiment::encodings::{BindContext, Encoding, Evaluation, EvaluationContext};
use crate::experiment::mask::{BitMask, BitView};
use crate::experiment::selection::Selection;
use crate::experiment::view_mut::ViewMut;
use fastlanes::{BitPacking, FastLanes};
use std::task::{Poll, ready};
use vortex_dtype::{NativePType, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;

pub struct BitPackedEncoding {
    bit_width: usize,
    buffer: ByteBufferHandle,
}

impl BitPackedEncoding {
    pub fn new(bit_width: usize, buffer: ByteBufferHandle) -> Self {
        Self { bit_width, buffer }
    }
}

impl Encoding for BitPackedEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        let ptype = ctx.dtype.as_ptype().to_unsigned();
        match_each_unsigned_integer_ptype!(ptype, |T| {
            Ok(Box::new(BitPackedEvaluation::<T> {
                width: self.bit_width,
                packed_stride: self.bit_width * <T as FastLanes>::LANES,
                buffer: BufferHandle::from_byte_buffer(self.buffer.clone()),
                packed_offset: 0,
            }))
        })
    }
}

struct BitPackedEvaluation<T> {
    width: usize,
    packed_stride: usize,

    buffer: BufferHandle<T>,
    packed_offset: usize,
}

impl<T: NativePType + BitPacking> Evaluation for BitPackedEvaluation<T> {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: &dyn BitMask,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;

        let mut view = out.as_primitive::<T>();
        let packed = &buffer.as_slice()[self.packed_offset..];

        // We compute the number of FastLanes vectors that we have remaining.
        let nvecs = (N / 1024).min(packed.len() / self.packed_stride);

        // We short-circuit full unpacking logic if the mask is sufficiently sparse.
        if selected.true_count() > 16 {
            for i in 0..nvecs {
                unsafe {
                    BitPacking::unchecked_unpack(
                        self.width,
                        &packed[(i * self.packed_stride)..][..self.packed_stride],
                        &mut view.as_mut()[(i * 1024)..],
                    );
                }
            }

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            out.set_selection_mask(selected);
            // view.flatten_with_mask(selected.borrow());

            Poll::Ready(Ok(()))
        } else {
            let mut offset = 0;
            selected.iter_ones(|idx| {
                let chunk_idx = idx / 1024;
                let bit_idx = idx % 1024;
                // SAFETY: we verify the bounds of the vector during construction.
                unsafe {
                    *view.as_mut().get_unchecked_mut(offset) = BitPacking::unchecked_unpack_single(
                        self.width,
                        &packed[(chunk_idx * self.packed_stride)..][..self.packed_stride],
                        bit_idx,
                    );
                }
                offset += 1;
            });

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            out.set_selection(Selection::Prefix {
                len: selected.true_count(),
            });

            Poll::Ready(Ok(()))
        }
    }
}
