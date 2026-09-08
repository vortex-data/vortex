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

mod bitpacking;
mod delta;
mod r#for;
mod rle;

pub const FL_CHUNK_SIZE: usize = 1024;

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

/// What the fill-forward carry does when it enters a new [`FL_CHUNK_SIZE`] chunk.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ChunkBoundary {
    /// Restart from `T::default()`. Encodings whose values are chunk-local, such as RLE
    /// indices, must not see a value from the previous chunk.
    Reset,
    /// Keep the last valid value. Delta encodes each chunk against its own bases, so a
    /// carried value only keeps the first residual small.
    Carry,
}

/// Fill-forward null values in a buffer, replacing each null with the last valid value seen.
///
/// `boundary` decides whether the carried value survives a [`FL_CHUNK_SIZE`] boundary.
///
/// Returns the original buffer if there are no nulls (i.e. the validity is
/// `NonNullable` or `AllValid`), avoiding any allocation or copy.
pub(crate) fn fill_forward_nulls<T: Copy + Default>(
    values: Buffer<T>,
    validity: &Validity,
    boundary: ChunkBoundary,
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
            let resets = boundary == ChunkBoundary::Reset;
            match values.try_into_mut() {
                Ok(mut to_fill_mut) => {
                    for (i, (v, is_valid)) in
                        to_fill_mut.iter_mut().zip(bit_buffer.iter()).enumerate()
                    {
                        if is_valid {
                            last_valid = *v;
                        } else {
                            if resets && i.is_multiple_of(FL_CHUNK_SIZE) {
                                last_valid = T::default();
                            }
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
                        } else if resets && i.is_multiple_of(FL_CHUNK_SIZE) {
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

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_buffer::BitBufferMut;
    use vortex_session::VortexSession;

    use super::*;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        initialize(&session);
        session
    });

    /// Only one value is valid, the last of chunk 0. Chunk 1 is all null and must either
    /// restart from zero or repeat that value, and the shared-buffer copy path must agree.
    #[rstest]
    #[case::reset_owned(ChunkBoundary::Reset, false, 0)]
    #[case::reset_shared(ChunkBoundary::Reset, true, 0)]
    #[case::carry_owned(ChunkBoundary::Carry, false, 42)]
    #[case::carry_shared(ChunkBoundary::Carry, true, 42)]
    fn fill_forward_nulls_at_chunk_boundary(
        #[case] boundary: ChunkBoundary,
        #[case] shared: bool,
        #[case] next_chunk_fill: u32,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = BufferMut::from_iter(std::iter::repeat_n(99u32, 2 * FL_CHUNK_SIZE));
        values[FL_CHUNK_SIZE - 1] = 42;

        let mut validity_bits = BitBufferMut::new_unset(2 * FL_CHUNK_SIZE);
        validity_bits.set(FL_CHUNK_SIZE - 1);

        let validity = Validity::from(validity_bits.freeze());
        let values = values.freeze();
        let _shared = shared.then(|| values.clone());
        let result = fill_forward_nulls(values, &validity, boundary, &mut ctx)?;

        let mut expected = BufferMut::zeroed(2 * FL_CHUNK_SIZE);
        expected[FL_CHUNK_SIZE - 1] = 42;
        expected[FL_CHUNK_SIZE..].fill(next_chunk_fill);
        assert_eq!(result, expected.freeze());
        Ok(())
    }
}
