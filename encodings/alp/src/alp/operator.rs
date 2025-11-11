// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_array::pipeline::{BindContext, PipelineTransform, TransformKernel};
use vortex_array::vtable::{OperatorVTable, PipelineNode};
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, PTypeDowncastExt, match_each_integer_ptype};
use vortex_error::VortexResult;
use vortex_vector::primitive::PVector;
use vortex_vector::{Vector, VectorMut};

use crate::{ALPArray, ALPFloat, ALPVTable, Exponents, match_each_alp_float_ptype};

impl OperatorVTable<ALPVTable> for ALPVTable {
    fn pipeline_node(array: &ALPArray) -> Option<PipelineNode<'_>> {
        Some(PipelineNode::Transform(array))
    }
}

impl PipelineTransform for ALPArray {
    fn pipelined_child(&self) -> usize {
        0 // The encoded vector is the first child
    }

    fn bind(&self, ctx: &mut dyn BindContext) -> VortexResult<Box<dyn TransformKernel>> {
        let exponents = self.exponents();

        match self.patches() {
            None => {
                match_each_alp_float_ptype!(self.ptype(), |A| {
                    Ok(Box::new(ALPKernel::<A> {
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
                        let patch_indices: Buffer<P> = patch_idxs.downcast::<P>().into_elements();
                        let patch_values: PVector<A> = A::downcast(patch_vals);
                        Ok(Box::new(PatchedALPKernel {
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
    // The ALP exponents
    exponents: Exponents,
    _phantom: PhantomData<A>,
}

impl<A: ALPFloat> TransformKernel for ALPKernel<A> {
    fn step(&mut self, input: &Vector, out: &mut VectorMut) -> VortexResult<()> {
        let encoded = input.as_primitive().downcast::<A::ALPInt>().elements();

        let decoded = out.as_primitive_mut().downcast::<A>();
        decoded.extend(
            encoded
                .iter()
                .map(|encoded_int| A::decode_single(*encoded_int, self.exponents)),
        );
        Ok(())
    }
}

#[allow(dead_code)] // TODO(ngates): implement patching
struct PatchedALPKernel<A: ALPFloat, P: NativePType> {
    // The ALP exponents
    exponents: Exponents,
    // The patch indices and values
    patch_indices: Buffer<P>,
    patch_values: PVector<A>,
}

impl<A: ALPFloat, P: NativePType> TransformKernel for PatchedALPKernel<A, P> {
    fn step(&mut self, input: &Vector, out: &mut VectorMut) -> VortexResult<()> {
        let encoded = input.as_primitive().downcast::<A::ALPInt>().elements();

        let decoded = out.as_primitive_mut().downcast::<A>();
        decoded.extend(
            encoded
                .iter()
                .map(|encoded| A::decode_single(*encoded, self.exponents)),
        );

        // Errrrrrr what patches do we apply?

        todo!()
    }
}
