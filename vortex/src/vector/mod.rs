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
//! - Move performance decisions into the array logic. For example, currently the caller has to
//!   decide which order to run compute vs filter.
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

mod export;
mod expression;
mod impls;
mod pipeline;
mod primitive;

use bitvec::access::BitSafeU64;
use bitvec::slice::BitSlice;
use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_dtype::PType;
use vortex_mask::Mask;

pub const N: usize = 2048;

/// A vector is the atomic unit of canonical data in Vortex.
///
/// Like with our canonical arrays, it is physically typed, not logically typed. So each DType
/// has a well-defined physical representation in vector form.
///
/// I'm not quite sure on sizing yet. We could follow DuckDB and make vectors 2k elements. We
/// could follow FastLanes and make them 1k elements. Or we could do something interesting where
/// we pick the largest power of two that lets us fit ~3-4 vectors into the L1 cache. For example,
/// strings may be 1k elements, but u8 integers may be 8k elements. Many compute functions operate
/// over two inputs of the same type, so this could be interesting.
///
/// Interestingly, Vectors don't own their own data. In fact, I'm considering calling them 'views'.
/// This also solves our problem of zero-copy export by allowing us to pass down an output buffer
/// from an external source. This tends to work well since these external sources typically agree
/// with us on the physical representation of data, e.g. DuckDB, Arrow, Numpy, etc.
///
/// ## Why is this a single type-erased struct?
///
/// If we used generics at this level, we would taint all functions that use this type with a
/// generic type. To remove the generic, we'd need to introduce a trait, at which point we're
/// forced into both dynamic dispatch, and boxed return types. We also cannot down-cast a dynamic
/// trait that holds borrowed data since `Any` requires a static lifetime.
///
/// ## How do we handle custom encodings, e.g. FSST, RoaringBitMap, ZStd, etc.?
///
/// I could imagine a VarBinView vector (i.e. it has 16-byte views in its elements buffer), but
/// is able to delay decompression of the data buffers. These could be stored as child arrays and
/// decompressed on-demand, since this is now an opaque operation (and then call export on the
/// child data arrays using a slices mask? We'd be masking binary data... that sounds slow).
/// In conclusion... I'm not really sure yet.
///
/// What about dictionary arrays? Are they even important at this level?
/// I have done a "medium amount" of thinking on this, and I think we can actually get away with
/// not supporting dictionary encoding at the vector level. Vortex arrays actually define the
/// encoding tree, and therefore can decide how to implement a compute function. So where it's
/// possible to return a dictionary encoded thing, e.g. to DuckDB, we would have some sort of
/// Vortex Array -> DuckDB function that would check for top-level dictionaries and handle the
/// conversion at that layer.

/// ## Can we re-use parts of the pipeline to avoid common-subexpression elimination?
///
/// This gets tricky... Suppose we start with a Vortex expression. We can then pass that naively
/// through pipeline construction. This now represents a physical execution plan. At this point,
/// we could in theory optimize the pipeline by removing common sub-expressions, such as
/// canonicalizing the same field multiple times to pass into two comparison operators.
///
/// We'd then need some way to buffer the intermediate results as both expressions are driven.
/// Maybe this works? Not sure yet.
pub struct Vector<'a> {
    /// The physical type of the vector, which defines how the elements are stored.
    vtype: VType,
    /// A pointer to the allocated elements buffer.
    /// Alignment is at least the size of the element type.
    /// The capacity of the elements buffer is N * size_of::<T>() where T is the element type.
    // TODO(ngates): it would be nice to guarantee _wider_ alignment, ideally 128 bytes, so that
    //  we can use aligned load/store instructions for wide SIMD lanes.
    elements: *mut u8,
    /// The validity mask for the vector, indicating which elements in the buffer are valid.
    validity: &'a mut VectorMask,
    // A selection mask over the elements and validity of the vector.
    // FIXME(ngates): using a selection mask means rank-based operations are expensive, vs
    //  selection indices which are always constant time.
    selection: Selection,

    /// Additional buffers of data used by the vector, such as string data.
    // TODO(ngates): ideally these buffers are compressed somehow? E.g. using FSST?
    data: Vec<ByteBuffer>,
    // Additional arrays used by the vector, such as...?
    children: Vec<ArrayRef>,

    /// Marker defining the lifetime of the contents of the vector.
    _marker: std::marker::PhantomData<&'a mut ()>,
}

/// A vector mask is a bit array containing N boolean values.
pub type VectorMask = BitSlice<BitSafeU64, bitvec::order::Msb0>;

/// Defines the "vector type", a physical type describing the data that's held in the vector.
///
/// See the specific vector view types, e.g. [`PrimitiveVector`], for more details.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VType {
    Primitive(PType),
    VarBin,
}

/// Provides a fast way to select a subset of elements from a vector.
pub enum Selection {
    /// Select no elements in the vector.
    Empty,
    /// Select all elements in the vector from zero up to the given length.
    Prefix { len: usize },
    /// The element in the vector to be considered the constant value.
    Constant { element: usize, len: usize },
    /// Select from the vector using a list of indices, up to a maximum of `N` indices.
    Indices(Box<[usize]>),
    /// Select from the vector using a mask, which is a bit array of length `N`.
    Mask(Mask),
}

impl<'a> Vector<'a> {
    /// Return the logical length of the vector, which is the number of selected elements.
    pub fn len(&self) -> usize {
        match self.selection {
            Selection::Empty => 0,
            Selection::Prefix { len } => len,
            Selection::Constant { len, .. } => len,
            Selection::Indices(ref indices) => indices.len(),
            Selection::Mask(ref mask) => mask.true_count(),
        }
    }

    pub fn set_selection(&mut self, selection: Selection) {
        #[cfg(debug_assertions)]
        {
            match &selection {
                Selection::Empty => {}
                Selection::Prefix { len } => {
                    assert!(
                        *len <= N,
                        "Selection prefix length must be less than or equal to N"
                    );
                }
                Selection::Constant { len, element } => {
                    assert!(
                        *len <= N,
                        "Selection constant length must be less than or equal to N"
                    );
                    assert!(
                        *element < N,
                        "Selection constant element must be less than N"
                    );
                }
                Selection::Indices(indices) => {
                    assert!(
                        indices.len() <= N,
                        "Selection indices length must be less than or equal to N"
                    );
                    for &index in indices.iter() {
                        assert!(index < N, "Selection index {} must be less than N", index);
                    }
                }
                Selection::Mask(mask) => {
                    assert_eq!(
                        mask.len(),
                        N,
                        "Selection mask length must be less than or equal to N"
                    );
                }
            }
        }
        self.selection = selection;
    }
}
