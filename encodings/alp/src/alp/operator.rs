// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{match_each_alp_float_ptype, ALPArray, ALPFloat, ALPVTable, Exponents};
use std::marker::PhantomData;
use vortex_array::pipeline::{
    BindContext, KernelContext, PipelineTransform, TransformKernel, VectorId,
};
use vortex_array::vtable::{OperatorVTable, PipelineNode};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_integer_ptype, NativePType, PTypeDowncastExt};
use vortex_error::VortexResult;
use vortex_vector::primitive::PVector;
use vortex_vector::VectorMut;

impl OperatorVTable<ALPVTable> for ALPVTable {
    fn pipeline_node(array: &ALPArray) -> Option<PipelineNode<'_>> {
        Some(PipelineNode::Transform(array))
    }
}

impl PipelineTransform for ALPArray {
    fn is_pipelined_child(&self, child_idx: usize) -> bool {
        match child_idx {
            0 => true,  // encoded array
            _ => false, // patch indices + patch values
        }
    }

    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn TransformKernel>> {
        let encoded_vector_id = ctx.pipelined_input(0);
        let exponents = self.exponents();

        match self.patches() {
            None => {
                match_each_alp_float_ptype!(self.ptype(), |A| {
                    Ok(Box::new(ALPKernel::<A> {
                        encoded_vector_id,
                        exponents,
                        _phantom: PhantomData,
                    }))
                })
            }
            Some(patches) => {
                let patch_idxs = ctx.batch_input(0).into_primitive();
                let patch_vals = ctx.batch_input(1).into_primitive();

                match_each_alp_float_ptype!(self.ptype(), |A| {
                    match_each_integer_ptype!(patches.indices_ptype(), |P| {
                        let patch_indices: Buffer<P> = P::downcast(patch_idxs).into_buffer();
                        let patch_values: PVector<A> = A::downcast(patch_vals);
                        Ok(Box::new(PatchedALPKernel {
                            encoded_vector_id,
                            exponents,
                            patch_indices,
                            patch_values,
                        }))
                    })
                })
            }
        }
    }
}

struct ALPKernel<A: ALPFloat> {
    // The encoded vector that returns `A::ALPInt` values
    encoded_vector_id: VectorId,
    // The ALP exponents
    exponents: Exponents,
    _phantom: PhantomData<A>,
}

impl<A: ALPFloat> TransformKernel for ALPKernel<A> {
    fn step(&mut self, ctx: &KernelContext, out: &mut VectorMut) -> VortexResult<()> {
        let encoded = ctx
            .vector(self.encoded_vector_id)
            .into_primitive()
            .downcast::<A::ALPInt>()
            .into_buffer();

        let mut decoded = A::downcast(out.into_primitive());
        decoded.extend(
            encoded
                .iter()
                .map(|encoded_int| A::decode_single(*encoded_int, self.exponents)),
        );
        Ok(())
    }
}

struct PatchedALPKernel<A: ALPFloat, P: NativePType> {
    // The encoded vector that returns `A::ALPInt` values
    encoded_vector_id: VectorId,
    // The ALP exponents
    exponents: Exponents,
    // The patch indices and values
    patch_indices: Buffer<P>,
    patch_values: PVector<A>,
}

impl<A: ALPFloat, P: NativePType> TransformKernel for PatchedALPKernel<A, P> {
    fn step(&mut self, ctx: &KernelContext, out: &mut VectorMut) -> VortexResult<()> {
        let encoded = ctx
            .vector(self.encoded_vector_id)
            .into_primitive()
            .downcast::<A::ALPInt>()
            .into_buffer();

        let mut decoded = out.into_primitive().downcast::<A>();
        decoded.extend(
            encoded
                .iter()
                .map(|encoded| A::decode_single(*encoded, self.exponents)),
        );

        // Errrrrrr what patches do we apply?

        todo!()
    }
}
