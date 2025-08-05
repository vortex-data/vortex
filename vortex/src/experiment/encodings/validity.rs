// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::encodings::{BindContext, Encoding, Evaluation, EvaluationContext};
use crate::experiment::mask::{BitMask, BitMaskView, BitVector};
use crate::experiment::view_mut::{Selection, Vector};
use bitvec::array::BitArray;
use std::ops::BitAnd;
use std::task::{Poll, ready};
use vortex_error::VortexResult;
use vortex_mask::Mask;

/// An operator that applies validity to the underlying.
pub struct ValidityEncoding {
    elements: Box<dyn Encoding>,
    validity: Box<dyn Encoding>,
}

impl Encoding for ValidityEncoding {
    fn bind(&self, ctx: &BindContext) -> VortexResult<Box<dyn Evaluation>> {
        todo!()
    }
}

struct FusedValidityEvaluation<E: Evaluation> {
    // Generic over the specific evaluation type for elements. Therefore, enables compiler
    // inlining and optimization opportunities.
    elements: E,
    validity: Box<dyn Evaluation>,
}

struct ValidityEvaluation {
    elements: Box<dyn Evaluation>,
    validity: Box<dyn Evaluation>,

    /// A reusable buffer for exporting validity from the child.
    validity_buffer: BitVector,
}

impl Evaluation for ValidityEvaluation {
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()> {
        self.elements.seek(chunk_idx)?;
        self.validity.seek(chunk_idx)
    }

    fn step(
        &mut self,
        ctx: &dyn EvaluationContext,
        selected: BitMask,
        defined: BitMask,
        out: &mut Vector,
    ) -> Poll<VortexResult<()>> {
        // First, we export a validity array from the validity child.
        let mut validity_vector = Vector::new_bool(&mut self.validity_buffer, None);
        ready!(
            self.validity
                .step(ctx, selected, defined, &mut validity_vector)
        )?;
        validity_vector.flatten();
        let validity = validity_vector.as_bool();

        // If the validity vector is all-true or all-false, we can skip further processing.
        match validity.as_mask() {
            BitMaskView::All => {
                // All values are valid, so we can just pass through the elements.
                self.elements.step(ctx, selected, defined, out)
            }
            BitMaskView::None => {
                // All values are invalid, therefore we can just return a constant invalid vector
                // without ever calling into the elements.
                // FIXME(ngates): we must seek forwards though
                out.validity().fill(false);
                out.set_selection(Selection::Constant {
                    element: 0,
                    len: selected.true_count(),
                });
                Poll::Ready(Ok(()))
            }
            BitMaskView::Some(validity) => {
                // Otherwise, we set the invalid values to be undefined when calling into the
                // child.
                todo!()
                //let defined = defined.bitand(validity);
                //self.elements.step(ctx, selected, defined.borrow(), out)
            }
        }
    }
}
