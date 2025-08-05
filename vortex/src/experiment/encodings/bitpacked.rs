// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::bits::BitView;
use crate::experiment::buffers::BufferHandle;
use crate::experiment::encodings::{BindContext, Encoding, Evaluation, EvaluationContext};
use crate::experiment::selection::Selection;
use crate::experiment::view::{Canonical, ViewMut};
use fastlanes::{BitPacking, FastLanes};
use std::task::{Poll, ready};
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_fastlanes::unpack_iter::BitPacked;

pub struct BitPackedEncoding {
    bit_width: usize,
    buffer: BufferHandle<u8>,
}

impl BitPackedEncoding {
    pub fn new(bit_width: usize, buffer: BufferHandle<u8>) -> Self {
        Self { bit_width, buffer }
    }
}

impl Encoding for BitPackedEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        let ptype = ctx.dtype.as_ptype().to_unsigned();
        match_each_integer_ptype!(ptype, |T| {
            Ok(Box::new(BitPackedEvaluation::<T> {
                width: self.bit_width,
                packed_stride: self.bit_width * <<T as BitPacked>::UnsignedT as FastLanes>::LANES,
                buffer: self.buffer.clone().into_typed(),
                packed_offset: 0,
            }))
        })
    }
}

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
struct BitPackedEvaluation<T: BitPacked> {
    width: usize,
    packed_stride: usize,

    buffer: BufferHandle<<T as BitPacked>::UnsignedT>,
    packed_offset: usize,
}

impl<T> Evaluation for BitPackedEvaluation<T>
where
    T: BitPacked,
    T: Canonical<Element = T>,
    <T as BitPacked>::UnsignedT: Canonical<Element = <T as BitPacked>::UnsignedT>,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;

        // We re-interpret the output view as the unsigned bitpacked type.
        out.reinterpret_as::<<T as BitPacked>::UnsignedT>();

        let elements = out.as_mut::<<T as BitPacked>::UnsignedT>();
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
                        &mut elements[(i * 1024)..],
                    );
                }
            }

            self.packed_offset += nvecs * self.packed_stride;

            // Set the selection to the given mask, which is a bit array of length N.
            out.set_selection_mask(selected.into());
        } else {
            let mut offset = 0;
            selected.iter_ones(|idx| {
                let chunk_idx = idx / 1024;
                let bit_idx = idx % 1024;
                // SAFETY: we verify the bounds of the vector during construction.
                unsafe {
                    *elements.get_unchecked_mut(offset) = BitPacking::unchecked_unpack_single(
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
        }

        // Put the output vector back to type `T`!
        out.reinterpret_as::<T>();

        Poll::Ready(Ok(()))
    }
}
