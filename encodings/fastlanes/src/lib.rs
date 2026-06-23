// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

//! FastLanes integer encodings for Vortex arrays.
//!
//! This crate provides SIMD-friendly integer encodings:
//!
//! - [`BitPacked`] stores fixed-width integer values using the minimum bit width plus optional
//!   patches.
//! - [`FoR`] stores frame-of-reference deltas from a base value.
//! - [`Delta`] stores adjacent deltas in chunked form.
//! - [`RLE`] stores repeated runs.
//!
//! Call [`initialize`] to register the encodings and encoding-specific aggregate kernels in a
//! session before deserializing or executing arrays that may contain these encodings.
//!
//! ```rust
//! let session = vortex_array::array_session();
//! vortex_fastlanes::initialize(&session);
//! ```
//!
//! ## Paper
//!
//! The original encodings are described in the paper [The FastLanes Compression Layout](https://15721.courses.cs.cmu.edu/spring2024/papers/03-data2/p2132-afroozeh.pdf),
//! but are not fully binary compatible. See the underlying [fastlanes](https://github.com/spiraldb/fastlanes) crate for more details.

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;
pub use rle::*;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

pub mod bit_transpose;
mod bitpacking;
mod delta;
mod r#for;
mod rle;

pub(crate) const FL_CHUNK_SIZE: usize = 1024;

use bitpacking::compute::is_constant::BitPackedIsConstantKernel;
use r#for::compute::is_constant::FoRIsConstantKernel;
use r#for::compute::is_sorted::FoRIsSortedKernel;
use vortex_array::ArrayVTable;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize fastlanes encodings in the given session.
pub fn initialize(session: &VortexSession) {
    // If we're using the experimental Patched encoding, register a shim
    // for BitPacked with interior patches decode as Patched array.
    if use_experimental_patches() {
        session.arrays().register(BitPackedPatchedPlugin);
    } else {
        session.arrays().register(BitPacked);
    }
    session.arrays().register(Delta);
    session.arrays().register(FoR);
    session.arrays().register(RLE);
    bitpacking::initialize(session);
    r#for::initialize(session);
    rle::initialize(session);

    // Register the encoding-specific aggregate kernels.
    session.aggregate_fns().register_aggregate_kernel(
        BitPacked.id(),
        Some(IsConstant.id()),
        &BitPackedIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        FoR.id(),
        Some(IsConstant.id()),
        &FoRIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        FoR.id(),
        Some(IsSorted.id()),
        &FoRIsSortedKernel,
    );
}

/// Fill-forward null values in a buffer, replacing each null with the last valid value seen.
///
/// The fill-forward state resets to `T::default()` at every [`FL_CHUNK_SIZE`] boundary
/// so that values from one chunk never leak into the next. This is important because
/// both RLE and Delta encodings treat each chunk independently: a fill-forwarded value
/// that crosses a chunk boundary can become an invalid chunk-local index (for RLE) or
/// an incorrect delta base (for Delta).
///
/// Returns the original buffer if there are no nulls (i.e. the validity is
/// `NonNullable` or `AllValid`), avoiding any allocation or copy.
pub(crate) fn fill_forward_nulls<T: Copy + Default>(
    values: Buffer<T>,
    validity: &Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<T>> {
    match validity {
        Validity::NonNullable | Validity::AllValid => Ok(values),
        Validity::AllInvalid => Ok(Buffer::zeroed(values.len())),
        Validity::Array(validity_array) => {
            let bit_buffer = validity_array
                .clone()
                .execute::<BoolArray>(ctx)?
                .to_bit_buffer();
            let mut last_valid = T::default();
            match values.try_into_mut() {
                Ok(mut to_fill_mut) => {
                    for (i, (v, is_valid)) in
                        to_fill_mut.iter_mut().zip(bit_buffer.iter()).enumerate()
                    {
                        if is_valid {
                            last_valid = *v;
                        } else if i.is_multiple_of(FL_CHUNK_SIZE) {
                            last_valid = T::default();
                        } else {
                            *v = last_valid;
                        }
                    }
                    Ok(to_fill_mut.freeze())
                }
                Err(to_fill) => {
                    let mut to_fill_mut = BufferMut::<T>::with_capacity(to_fill.len());
                    for (i, (v, (out, is_valid))) in to_fill
                        .iter()
                        .zip(
                            to_fill_mut
                                .spare_capacity_mut()
                                .iter_mut()
                                .zip(bit_buffer.iter()),
                        )
                        .enumerate()
                    {
                        if is_valid {
                            last_valid = *v;
                        } else if i.is_multiple_of(FL_CHUNK_SIZE) {
                            last_valid = T::default();
                        }
                        out.write(last_valid);
                    }
                    unsafe { to_fill_mut.set_len(to_fill.len()) };
                    Ok(to_fill_mut.freeze())
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::VortexSessionExecute;
    use vortex_buffer::BitBufferMut;
    use vortex_session::VortexSession;

    use super::*;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        initialize(&session);
        session
    });

    #[test]
    fn fill_forward_nulls_resets_at_chunk_boundary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Build a buffer spanning two chunks where the last valid value in chunk 0
        // is non-zero. Null positions at the start of chunk 1 must get T::default()
        // (0), not the carry-over from chunk 0.
        let mut values = BufferMut::zeroed(2 * FL_CHUNK_SIZE);
        // Place a non-zero valid value near the end of chunk 0.
        values[FL_CHUNK_SIZE - 1] = 42;

        let mut validity_bits = BitBufferMut::new_unset(2 * FL_CHUNK_SIZE);
        validity_bits.set(FL_CHUNK_SIZE - 1); // only this position is valid

        let validity = Validity::from(validity_bits.freeze());
        let result = fill_forward_nulls(values.freeze(), &validity, &mut ctx)?;

        // Within chunk 0, nulls before the valid element get 0 (default), and the
        // valid element itself is 42.
        assert_eq!(result[FL_CHUNK_SIZE - 1], 42);

        // Chunk 1 has no valid elements. Every position must be T::default() (0),
        // NOT 42 carried over from chunk 0.
        for i in FL_CHUNK_SIZE..2 * FL_CHUNK_SIZE {
            assert_eq!(
                result[i], 0,
                "position {i} should be 0, not carried from chunk 0"
            );
        }
        Ok(())
    }
}
