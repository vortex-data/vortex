// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! I'm calling these vectors for two reasons: first, so I don't confuse myself with what we
//! currently call arrays (we're probably on Arrays 5.0 at this point), and second, because
//! as first writing this, I'm not entirely sure if vectors are distinct from arrays. Anyway,
//! you're here for the ride now!
//!
//! Goals:
//! - Bring Vortex performance up to state-of-the-art.
//! - Support zero-copy decoding of data from disk into externally provided buffers.
//! - All without a wire break (I'm quite confident in this).
//!
//! How I plan to achieve this:
//! - Lean heavily on SIMD compute and CPU cache locality.
//!
//! Therefore, some meta-goals that fall out of this:
//! - Thread-locality and core affinity is important. Keep data within the L1 cache as much as
//!   possible. This has the additional benefit of avoiding overhead of concurrency and
//!   synchronization primitives.
//! - Data to be processed in much smaller chunks, fitting in the L1 cache, rather than now where
//!   data is largely processed in the chunks as they appear in the file.
//! - Outputs need to be passed in to the scan / compute functions in order to support externally
//!   provided buffers, such as Arrow, Numpy, etc.
//!
//! Evaluation:
//! - Our primary focus is on DuckDB performance, largely because the execution model aligns so
//!   well. If we can return DuckDB's 2k vectors efficiently, then we can hopefully keep the entire
//!   pipeline from disk through to the DuckDB result within the L1 or L2 caches.
//! - We care more about the performance of scan-heavy queries, less about join-heavy queries.
//!   We do care about the performance of highly selective queries to explore how masking interacts
//!   with pipelined compute.
//!
//! ## Pipelined Compute
//!
//! The core component if this change is to introduce a new compute model that allows for better
//! pipelining of operations over smaller chunks of data.
//!
//! In this world, an Array becomes actually _more_ like a Layout, in that it can be converted into
//! a compute pipeline (evaluation) to be executed piecemeal. An array holds onto zero-copy data
//! from disk, where the data is only accessed at the time of evaluation. A pipeline is then driven
//! with small chunks of data at a time.
//!
//! Arrays still support compute functions that take and return arrays, but internally, these are
//! implemented using pipelined evaluation. The array on which the compute function was invoked is
//! known as the "entry point" array, and it is responsible for constructing an evaluation, driving
//! it, and collecting the result. For example, a DictArray can drive separate evaluations for its
//! values and codes, and then re-assemble the results into a dictionary. Note that this dict
//! push-down will therefore only occur if the top-level entry point is a DictArray.
//!
//! So each array has one function to get a compute kernel, and one function to get a compute
//! evaluation. If either fails to return, a default canonical implementation is used, as now.
//!

#![allow(dead_code)]
#![allow(unused_variables)]

mod array;
mod evaluation;
mod exporter;
mod vector;

use evaluation::Evaluation;
use exporter::Exporter;
use fastlanes::BitPacking;
use std::sync::Arc;
use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::ScalarValue;

/// What's the relationship between a Layout, a Vector, and an Array?
///
/// Ideally, compute is performed by passing streams of vectors through an evaluation task.
///
/// The caller of an evaluation knows the expected length, so it will keep invoking the evaluation
/// repeatedly until it has returned enough data. An evaluation takes a context, so we could use
/// this to pass out required segment IDs and use this to drive async layout evaluation.
///
/// How static are evaluation trees? For example, chunking may result in different
/// evaluations for each chunk (due to compression/layout differences). We don't want to eagerly
/// construct all of these... do we? We currently do for layouts. This would be similar to passing
/// in a row selection + field mask, and allowing the evaluation to do pruning on construction. But
/// it doesn't allow for short-circuiting operations. So yes, maybe evaluations should be allowed
/// to be mutable at runtime.
///
/// Evaluations should be `Send`, but `Sync` would add too much overhead. This means for execution,
/// we are able to work-steal evaluations across threads, but we cannot invoke them on a
/// work-stealing runtime, only on a single thread at a time. This is fine.
///
/// So it sounds like evaluations replace compute function kernels. Instead of taking kernel inputs
/// as arrays, they are taken as evaluations.
///
/// What are vectors then? Do they replace arrays? Do they replace canonical arrays? Maybe they do
/// replace arrays ultimately?

/// The number of values in a vector. This is a compile-time constant.
pub const N: usize = 1024;

enum Selection {
    All,
    /// A selection that is a mask.
    Mask(Mask),
    /// A selection that is a list of indices.
    Indices(Vec<usize>),
}

/// Let's define a dummy expression language.
enum Expression {
    /// References the root scope.
    Root,
    /// Holds a scalar value.
    Literal(ScalarValue),
    /// Less than comparison.
    Lt(Box<Expression>, Box<Expression>),
    /// Logical AND operation.
    And(Box<Expression>, Box<Expression>),
}

trait Array {
    /// Create a new evaluation for the given expression.
    fn evaluation(&self, expr: &Expression) -> Box<dyn Evaluation + '_>;
}

/// Let's design a FastLanes BitPacked array.
struct FLBitPacked<T> {
    packed_width: usize, // The packed width in bits.
    packed: Buffer<T>,
}

impl<T: NativePType + BitPacking> Array for FLBitPacked<T> {
    fn evaluation(&self, expr: &Expression) -> Box<dyn Evaluation + '_> {
        match &expr {
            Expression::Root => Box::new(FLBitPackedExport {
                packed_width: self.packed_width,
                packed_chunk_len: 1024 * self.packed_width / T::PTYPE.bit_width(),
                packed: &self.packed,
            }),
            _ => unreachable!("Only root expressions are supported for now."),
        }
    }
}

/// Export a BitPacked array into a stream of vectors.
struct FLBitPackedExport<'a, T> {
    packed_width: usize,     // The width of the packed data in bits.
    packed_chunk_len: usize, // The number of elements of type T form a packed chunk.
    packed: &'a [T],
}

impl<'a, T: NativePType + BitPacking> Evaluation for FLBitPackedExport<'a, T> {
    fn next(&mut self, mask: &Mask, out: &mut dyn Exporter) -> VortexResult<()> {
        // We know that the vector has a fixed capacity of N. The mask also covers the same range.
        assert_eq!(mask.len(), N);

        // So now we can produce a vector from the packed data.
        let packed_len = 1024 * self.packed_width / T::PTYPE.bit_width();

        // Unpack the values.
        unsafe {
            BitPacking::unchecked_unpack(
                self.packed_width,
                &self.packed[0..packed_len],
                out.as_mut_primitive::<T>(),
            )
        }

        // Set the selection mask
        out.set_selection(Selection::Mask(mask.clone()));

        // Advance the packed data offset.
        self.packed = &self.packed[self.packed_chunk_len..];

        Ok(())
    }
}

/// And a FoR array. To make it interesting, we don't fuse it with bitpacking.
struct FoR {
    child: Arc<dyn Array>,
    reference: ScalarValue,
}

impl Array for FoR {
    fn evaluation(&self, expr: &Expression) -> Box<dyn Evaluation + '_> {
        todo!()
    }
}

fn export_to_arrow<T>(len: usize) {
    // We create the un-initialized buffers for the Arrow array.
    //
    // We then call export on the evaluation repeatedly, increasing the slice a little bit each
    // time until we reach the end of the array. TODO(ngates): this means the fastlanes output
    // buffer will not have the correct alignment if we ever get a returned vector that has a
    // selection mask.
    //
    // It's like, our exporter needs a way to return the buffer if correctly aligned, or a new
    // buffer from the pool if not. And if not, it must then later copy into the correct buffer.

    // Vector needs a way to "compact" itself into a FlatVector (i.e. no selection mask), with a
    // specified length. Compact vectors can be operated on together.

    // What do we do about the expression tree though?
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::vector::Expression;
    use vortex_array::IntoArray;
    use vortex_buffer::buffer;
    use vortex_fastlanes::BitPackedArray;

    #[test]
    fn test_expr_export() {
        // Pack 1 million u32s into 3 bits each.
        let packed = BitPackedArray::encode(
            &buffer![6u32; N * 1000].into_array(),
            3, // Packed width in bits.
        )
        .unwrap();

        let array = Arc::new(FLBitPacked {
            packed_width: 3,
            packed: packed.packed().clone(),
        });

        // To perform a simple identitiy evaluation, we can use the root expression.
        let eval = array.evaluation(&Expression::Root);
    }
}
