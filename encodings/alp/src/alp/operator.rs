// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{match_each_alp_float_ptype, ALPArray, ALPFloat, ALPVTable, Exponents};
use std::marker::PhantomData;
use vortex_array::pipeline::bit_view::BitView;
use vortex_array::pipeline::{
    BindContext, Kernel, KernelCtx, PipelineInputs, PipelinedNode, Position, VectorId, N,
};
use vortex_array::vtable::OperatorVTable;
use vortex_dtype::PTypeDowncastExt;
use vortex_error::{vortex_bail, VortexResult};
use vortex_vector::VectorMut;

impl OperatorVTable<ALPVTable> for ALPVTable {
    fn pipeline_node(array: &ALPArray) -> Option<&dyn PipelinedNode> {
        Some(array)
    }
}

impl PipelinedNode for ALPArray {
    fn inputs(&self) -> PipelineInputs {
        PipelineInputs::Transform {
            pipelined_inputs: vec![0],
        }
    }

    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        let encoded_vector_id = ctx.pipelined_input(0);
        match self.patches() {
            Some(_) => vortex_bail!("patched ALP kernel not implemented",),
            None => {
                if !self.all_valid() {
                    vortex_bail!("ALP kernel does not yet handle nulls",);
                }
                match_each_alp_float_ptype!(self.ptype(), |A| {
                    Ok(Box::new(UnpatchedALPKernel {
                        encoded_vector_id,
                        exponents: self.exponents(),
                        _phantom: PhantomData::<A>::default(),
                    }))
                })
            }
        }
    }
}

struct UnpatchedALPKernel<A> {
    encoded_vector_id: VectorId,
    exponents: Exponents,
    _phantom: PhantomData<A>,
}

impl<A: ALPFloat> Kernel for UnpatchedALPKernel<A> {
    fn step(
        &mut self,
        ctx: &KernelCtx,
        selection: &BitView,
        out: &mut VectorMut,
    ) -> VortexResult<()> {
        if selection.true_count() == 0 {
            // Nothing to do, and no kernel state to update
            return Ok(());
        }

        let encoded = ctx.input(self.encoded_vector_id);
        let encoded_vec = encoded.as_primitive().downcast::<A::ALPInt>();
        let encoded_buf = encoded_vec.elements().as_slice();

        let decoded = out.as_primitive_mut().downcast::<A>();

        match encoded.position() {
            Position::InPlace => {
                // TODO(ngates): tune the threshold
                if selection.true_count() < (N / 8) {
                    // Operate only over the selected elements, appending `true_count` elements
                    unsafe {
                        decoded
                            .validity_mut()
                            .append_n(true, selection.true_count())
                    };
                    unsafe { decoded.elements_mut().set_len(selection.true_count()) };
                    let decoded_buf = unsafe { decoded.elements_mut() };

                    let mut out_pos = 0;
                    selection.iter_ones(|idx| {
                        let encoded = unsafe { encoded_buf.get_unchecked(idx) };
                        let decoded = A::decode_single(*encoded, self.exponents);
                        *unsafe { decoded_buf.get_unchecked_mut(out_pos) } = decoded;
                        out_pos += 1;
                    })
                } else {
                    // Operate over all N elements, appending N elements
                    assert_eq!(encoded_buf.len(), N);
                    decoded.extend(
                        encoded_buf
                            .iter()
                            .map(|e| A::decode_single(*e, self.exponents)),
                    );
                }
            }
            Position::Compact => {
                // Loop over the compacted input elements
                decoded.extend(
                    encoded
                        .as_primitive()
                        .downcast::<A::ALPInt>()
                        .elements()
                        .iter()
                        .map(|e| A::decode_single(*e, self.exponents)),
                )
            }
        }

        Ok(())
    }
}
