// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains experiments into pipelined data processing within Vortex.
//!
//! Arrays (and eventually Layouts) will be convertible into a [`Kernel`] that can then be
//! exported into a [`ViewMut`] one chunk of [`N`] elements at a time. This allows us to keep
//! compute largely within the L1 cache, as well as to write out canonical data into externally
//! provided buffers.
//!
//! Each chunk is represented in a canonical physical form, as determined by the logical
//! [`vortex_dtype::DType`] of the array. This provides a predicate base on which to perform
//! compute. Unlike DuckDB and other vectorized systems, we force a single canonical representation
//! instead of supporting multiple encodings because compute push-down is applied a priori to the
//! logical representation.
//!
//! It is a work-in-progress and is not yet used in production.

pub mod bits;
pub mod buffers;
pub mod canonical;
pub mod operators;
pub mod query;
pub mod selection;
pub mod types;
pub mod vector;
pub mod view;

/// The number of elements in each step of a Vortex evaluation pipeline.
pub const N: usize = 1024;

use std::ops::Range;
use std::task::Poll;

use vector::{VectorId, VectorRef};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err, vortex_panic};

use crate::pipeline::bits::BitView;
use crate::pipeline::buffers::BufferId;
use crate::pipeline::view::ViewMut;

/// A pipeline provides a push-based way to emit a stream of canonical data.
///
/// By passing multiple vector computations through the same pipeline, we can amortize
/// the setup costs (such as DType validation, stats short-circuiting, etc.), and to make better
/// use of CPU caches by performing all operations while the data is hot.
///
/// By passing a mask into the `step` function, we give encodings visibility into the data that
/// will be read by their parents. Some encodings may choose to decode all `N` elements, and then
/// set the given selection mask on the output vector. Other encodings may choose to only unpack
/// the selected elements.
///
/// We are considering further adding a `defined` parameter that indicates which elements are
/// defined and will be interpreted by the parent. This differs from masking, in that undefined
/// elements should still live in the correct location, it just doesn't matter what their value
/// is. This will allow, e.g. a validity encoding to tell its children that the values in certain
/// positions are going to be masked out anyway, so don't bother doing any expensive compute.
pub trait Kernel {
    /// Seek the pipeline to a specific chunk offset.
    ///
    /// i.e. the resulting row offset is `idx * N`, where `N` is the number of elements in a chunk.
    ///
    /// The reason for a separate seek function (vs passing an offset directly to `step`) is that
    /// it allows the pipeline to optimize for sequential access patterns, which is common in
    /// many encodings. For example, a run-length encoding can efficiently seek to the start of a
    /// chunk without needing to perform a full binary search of the ends in each step.
    // TODO(ngates): should this be `skip(n)` instead? Depends if we want to support going
    //  backwards?
    fn seek(&mut self, chunk_idx: usize) -> VortexResult<()>;

    /// Attempts to perform a single step of the pipeline, writing data to the output vector.
    /// Returns `Poll::Done` if the pipeline is complete, or `Poll::Pending` if buffers are
    /// required to continue.
    ///
    /// The `selected` parameter defines which elements of the chunk should be exported, where
    /// `None` indicates that all elements are selected.
    ///
    // TODO(ngates): we could introduce a `defined` parameter to indicate which elements are
    //  defined and will be interpreted by the parent. This would allow us to skip writing
    //  elements that are not defined, for example if the parent is a dense null validity encoding.
    fn step(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> Poll<VortexResult<()>>;
}

pub trait KernelExt: Kernel {
    /// Perform a single step of the pipeline, panics if the step returns [`Poll::Pending`].
    fn step_now(
        &mut self,
        ctx: &dyn KernelContext,
        selected: BitView,
        out: &mut ViewMut,
    ) -> VortexResult<()> {
        match self.step(ctx, selected, out) {
            Poll::Ready(r) => r,
            Poll::Pending => {
                vortex_panic!("Pipeline step is pending, but expected it to be ready.")
            }
        }
    }
}

impl<K: Kernel + ?Sized> KernelExt for K {}

pub trait KernelContext {
    /// Get a vector by its ID.
    fn vector(&self, vector_id: VectorId) -> VectorRef<'_>;

    /// Get a buffer by its ID.
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>>;

    /// Pre-fetch buffers for future use (non-blocking hint).
    fn prefetch(&self, buffer_ids: &[BufferId]) {
        for &buffer_id in buffer_ids {
            let _ = self.buffer(buffer_id);
        }
    }

    /// Request a range of data from a buffer (for partial reads).
    fn buffer_range(
        &self,
        buffer_id: BufferId,
        range: Range<usize>,
    ) -> Poll<VortexResult<ByteBuffer>> {
        match self.buffer(buffer_id) {
            Poll::Ready(Ok(buffer)) => {
                let start = range.start;
                let end = range.end;
                if start < end && end <= buffer.len() {
                    Poll::Ready(Ok(buffer.slice(start..end)))
                } else {
                    Poll::Ready(Err(vortex_err!(
                        "Invalid range for buffer: {}..{}",
                        start,
                        end
                    )))
                }
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl KernelContext for () {
    fn vector(&self, vector_id: VectorId) -> VectorRef<'_> {
        todo!()
    }

    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use vortex_buffer::BufferMut;
    use vortex_dtype::{DType, Nullability};
    use vortex_mask::Mask;

    use crate::canonical::ToCanonical;
    use crate::compute::Operator;
    use crate::pipeline::canonical::export_canonical_pipeline_expr;
    use crate::pipeline::operators::compare::CompareOperator;
    use crate::{Array, IntoArray};

    #[test]
    fn test_pipeline_with_comparison() {
        // Create test data
        let mut rng = StdRng::seed_from_u64(42);
        let values = (0..1000)
            .map(|_| rng.random_range(0i32..100))
            .collect::<BufferMut<i32>>()
            .into_array()
            .to_primitive()
            .unwrap();

        // Create a mask that selects ~50% of elements
        let mask_bools: Vec<bool> = (0..1000).map(|_| rng.random_bool(0.5)).collect();
        let mask = Mask::from_buffer(BooleanBuffer::from_iter(mask_bools));

        // Create a pipeline with comparison: array > array (self-comparison)
        let expr1 = values.to_pipeline_plan().unwrap();
        let expr2 = values.to_pipeline_plan().unwrap();
        let compare_expr = CompareOperator::new(expr1, expr2, Operator::Gt);

        // Execute the pipeline
        let result = export_canonical_pipeline_expr(
            &DType::Bool(Nullability::NonNullable),
            values.len(),
            &compare_expr,
            &mask,
        )
        .unwrap();

        // Verify the result
        assert!(matches!(result, crate::Canonical::Bool(_)));
        if let crate::Canonical::Bool(bool_array) = result {
            // Since we're comparing array > array (same values), all results should be false
            let expected_len = mask.true_count();
            assert_eq!(bool_array.len(), expected_len);

            // All values should be false since we're comparing identical values
            let bool_buffer = bool_array.boolean_buffer();
            assert_eq!(
                bool_buffer.count_set_bits(),
                0,
                "All comparisons should be false"
            );
        }
    }

    #[test]
    fn test_pipeline_with_different_arrays_comparison() {
        // Create test data with known pattern
        let values1 = (0..1000)
            .map(|i| (i % 100))
            .collect::<BufferMut<i32>>()
            .into_array()
            .to_primitive()
            .unwrap();
        let values2 = (0..1000)
            .map(|i| ((i + 1) % 100))
            .collect::<BufferMut<i32>>()
            .into_array()
            .to_primitive()
            .unwrap();

        // Select all elements
        let mask = Mask::from_buffer(BooleanBuffer::new_set(1000));

        // Create pipeline: array1 < array2
        let expr1 = values1.to_pipeline_plan().unwrap();
        let expr2 = values2.to_pipeline_plan().unwrap();
        let compare_expr = CompareOperator::new(expr1, expr2, Operator::Lt);

        // Execute the pipeline
        let result = export_canonical_pipeline_expr(
            &DType::Bool(Nullability::NonNullable),
            1000,
            &compare_expr,
            &mask,
        )
        .unwrap();

        // Verify the result
        assert!(matches!(result, crate::Canonical::Bool(_)));
        if let crate::Canonical::Bool(bool_array) = result {
            assert_eq!(bool_array.len(), 1000);

            // Most comparisons should be true (except when values wrap around)
            let bool_buffer = bool_array.boolean_buffer();
            let true_count = bool_buffer.count_set_bits();

            // Should be approximately 990 true values (10 false when wrapping from 99 to 0)
            assert!(
                true_count > 980,
                "Expected most comparisons to be true, got {}",
                true_count
            );
            assert!(
                true_count < 1000,
                "Expected some comparisons to be false due to wraparound"
            );
        }
    }
}
