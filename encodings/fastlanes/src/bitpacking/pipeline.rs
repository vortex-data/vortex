// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{BitPackedArray, BitPackedVTable};
use fastlanes::{BitPacking, FastLanes};
use std::hash::{Hash, Hasher};
use std::task::{Poll, ready};
use vortex_array::pipeline::PipelineContext;
use vortex_array::pipeline::bits::BitView;
use vortex_array::pipeline::buffers::BufferHandle;
use vortex_array::pipeline::nodes::plan::source::{SourceNode, SourceNodeAdapter, SourceOperator};
use vortex_array::pipeline::nodes::plan::{BindContext, PlanNode};
use vortex_array::pipeline::selection::Selection;
use vortex_array::pipeline::types::Element;
use vortex_array::pipeline::view::ViewMut;
use vortex_array::pipeline::{N, Operator};
use vortex_array::vtable::PipelineVTable;
use vortex_dtype::{PhysicalPType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_bail};

impl PipelineVTable<BitPackedVTable> for BitPackedVTable {
    fn to_pipeline_plan(array: &BitPackedArray) -> VortexResult<Box<dyn PlanNode>> {
        if array.dtype.is_nullable() {
            vortex_bail!("BitPackedVTable does not support nullable types");
        }
        if array.patches.is_some() {
            vortex_bail!("BitPackedVTable does not support patched arrays");
        }

        let ptype = array.dtype.as_ptype();
        match_each_integer_ptype!(ptype, |T| {
            Ok(Box::new(SourceNodeAdapter::new(BitPackedPlan::<T> {
                width: array.bit_width as usize,
                packed_stride: array.bit_width as usize
                    * <<T as PhysicalPType>::Physical as FastLanes>::LANES,
                buffer: BufferHandle::new(array.packed.clone())
                    .into_typed::<<T as PhysicalPType>::Physical>(),
            })))
        })
    }

    fn to_pipeline(array: &BitPackedArray) -> VortexResult<Box<dyn Operator>> {
        if array.dtype.is_nullable() {
            vortex_bail!("BitPackedVTable does not support nullable types");
        }
        if array.patches.is_some() {
            vortex_bail!("BitPackedVTable does not support patched arrays");
        }

        let ptype = array.dtype.as_ptype();
        match_each_integer_ptype!(ptype, |T| {
            Ok(Box::new(BitPackedPipeline::<T> {
                width: array.bit_width as usize,
                packed_stride: array.bit_width as usize
                    * <<T as PhysicalPType>::Physical as FastLanes>::LANES,
                buffer: BufferHandle::new(array.packed.clone())
                    .into_typed::<<T as PhysicalPType>::Physical>(),
                packed_offset: 0,
            }))
        })
    }
}

#[derive(Debug)]
struct BitPackedPlan<T: PhysicalPType> {
    width: usize,
    packed_stride: usize,
    buffer: BufferHandle<<T as PhysicalPType>::Physical>,
}

impl<T: PhysicalPType> Hash for BitPackedPlan<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.width.hash(state);
        self.packed_stride.hash(state);
        self.buffer.hash(state);
    }
}

impl<T: Element + PhysicalPType<Physical: BitPacking>> SourceNode<T, BitPackedPipeline<T>>
    for BitPackedPlan<T>
{
    fn bind(&self, _ctx: &dyn BindContext) -> VortexResult<BitPackedPipeline<T>> {
        Ok(BitPackedPipeline::<T> {
            width: self.width,
            packed_stride: self.packed_stride,
            buffer: self.buffer.clone(),
            packed_offset: 0,
        })
    }
}

// TODO(ngates): we should try putting the const bit width as a generic here, to avoid
//  a switch in the fastlanes library on every invocation of `unchecked_unpack`.
pub(crate) struct BitPackedPipeline<T: PhysicalPType<Physical: BitPacking>> {
    width: usize,
    packed_stride: usize,

    buffer: BufferHandle<<T as PhysicalPType>::Physical>,
    packed_offset: usize,
}

impl<T> SourceOperator<T> for BitPackedPipeline<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;

        // We re-interpret the output view as the unsigned bitpacked type.
        out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = out.as_mut::<<T as PhysicalPType>::Physical>();
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

impl<T> Operator for BitPackedPipeline<T>
where
    T: PhysicalPType<Physical: BitPacking>,
    T: Element,
    <T as PhysicalPType>::Physical: Element,
{
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        let fls_chunk_idx = chunk_idx * (N / 1024);
        self.packed_offset = fls_chunk_idx * self.packed_stride;
        Ok(())
    }

    fn step(
        &mut self,
        ctx: &dyn PipelineContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>> {
        let buffer = ready!(self.buffer.get_or_load(ctx))?;

        // We re-interpret the output view as the unsigned bitpacked type.
        out.reinterpret_as::<<T as PhysicalPType>::Physical>();

        let elements = out.as_mut::<<T as PhysicalPType>::Physical>();
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
