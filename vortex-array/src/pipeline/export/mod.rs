// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::bits::BitView;
use crate::pipeline::types::Element;
use crate::pipeline::vector::TypedVector;
use crate::pipeline::{N, Pipeline, PipelineContext};
use bitvec::order::Msb0;
use bitvec::vec::BitVec;
use std::iter;
use std::task::Poll;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};

struct PrimitiveExporter<'a, T> {
    ctx: &'a dyn PipelineContext,
    pipeline: Box<dyn Pipeline>,
    remaining: usize,
    vector: TypedVector<T>,
    tail_mask: BitVec<u64, Msb0>,
}

impl<'a, T: Element + NativePType> PrimitiveExporter<'a, T> {
    pub fn new(ctx: &'a dyn PipelineContext, pipeline: Box<dyn Pipeline>, len: usize) -> Self {
        Self {
            ctx,
            pipeline,
            remaining: len,
            vector: TypedVector::default(),
            tail_mask: Default::default(),
        }
    }

    pub fn export_all<F>(&mut self, mut f: F)
    where
        F: FnMut(&TypedVector<T>),
    {
        self.try_export_all(|v| {
            f(v);
            Ok(())
        })
        .vortex_expect("infallible");
    }

    pub fn try_export_all<F>(&mut self, mut f: F) -> VortexResult<()>
    where
        F: FnMut(&TypedVector<T>) -> VortexResult<()>,
    {
        while self.remaining > 0 {
            let mask = if self.remaining >= N {
                BitView::all_true()
            } else {
                self.tail_mask.clear();
                self.tail_mask
                    .extend(iter::repeat(true).take(self.remaining));
                self.tail_mask.resize(N, false);
                unsafe {
                    BitView::new_unchecked(
                        self.tail_mask
                            .as_bitslice()
                            .try_into()
                            .map_err(|e| vortex_err!("infallible: {e}"))?,
                        self.remaining,
                    )
                }
            };

            match self
                .pipeline
                .step(self.ctx, mask, &mut self.vector.as_view_mut())
            {
                Poll::Ready(_) => {}
                Poll::Pending => {
                    vortex_bail!("Pipeline step is pending, cannot proceed with iteration.");
                }
            }
            self.remaining -= mask.true_count();

            f(&self.vector)?;
        }

        Ok(())
    }
}
