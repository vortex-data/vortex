// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::cast_possible_truncation)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;
pub use rle::*;
use vortex_array::ToCanonical;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

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
use vortex_array::arrays::patched::USE_EXPERIMENTAL_PATCHES;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize fastlanes encodings in the given session.
pub fn initialize(session: &VortexSession) {
    // If we're using the experimental Patched encoding, register a shim
    // for BitPacked with interior patches decode as Patched array.
    if *USE_EXPERIMENTAL_PATCHES {
        session.arrays().register(BitPackedPatchedPlugin);
    } else {
        session.arrays().register(BitPacked);
    }
    session.arrays().register(Delta);
    session.arrays().register(FoR);
    session.arrays().register(RLE);

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
) -> Buffer<T> {
    match validity {
        Validity::NonNullable | Validity::AllValid => values,
        Validity::AllInvalid => Buffer::zeroed(values.len()),
        Validity::Array(validity_array) => {
            let bit_buffer = validity_array.to_bool().to_bit_buffer();
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
                    to_fill_mut.freeze()
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
                    to_fill_mut.freeze()
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::session::ArraySessionExt;
    use vortex_buffer::BitBufferMut;
    use vortex_session::VortexSession;

    use super::*;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        session.arrays().register(BitPacked);
        session.arrays().register(Delta);
        session.arrays().register(FoR);
        session.arrays().register(RLE);
        session
    });

    #[test]
    fn fill_forward_nulls_resets_at_chunk_boundary() {
        // Build a buffer spanning two chunks where the last valid value in chunk 0
        // is non-zero. Null positions at the start of chunk 1 must get T::default()
        // (0), not the carry-over from chunk 0.
        let mut values = BufferMut::zeroed(2 * FL_CHUNK_SIZE);
        // Place a non-zero valid value near the end of chunk 0.
        values[FL_CHUNK_SIZE - 1] = 42;

        let mut validity_bits = BitBufferMut::new_unset(2 * FL_CHUNK_SIZE);
        validity_bits.set(FL_CHUNK_SIZE - 1); // only this position is valid

        let validity = Validity::from(validity_bits.freeze());
        let result = fill_forward_nulls(values.freeze(), &validity);

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
    }
}
