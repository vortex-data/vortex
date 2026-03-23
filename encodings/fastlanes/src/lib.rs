// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;
pub use rle::*;
use vortex_array::ToCanonical;
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
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::fns::is_constant::IsConstant;
use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize fastlanes encodings in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(BitPacked);
    session.arrays().register(Delta);
    session.arrays().register(FoR);
    session.arrays().register(RLE);

    // Register the encoding-specific aggregate kernels.
    session.aggregate_fns().register_aggregate_kernel(
        BitPacked::ID,
        Some(IsConstant.id()),
        &BitPackedIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        FoR::ID,
        Some(IsConstant.id()),
        &FoRIsConstantKernel,
    );
    session.aggregate_fns().register_aggregate_kernel(
        FoR::ID,
        Some(IsSorted.id()),
        &FoRIsSortedKernel,
    );
}

/// Fill-forward null values in a buffer, replacing each null with the last valid value seen.
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
                    for (v, is_valid) in to_fill_mut.iter_mut().zip(bit_buffer.iter()) {
                        if is_valid {
                            last_valid = *v;
                        } else {
                            *v = last_valid;
                        }
                    }
                    to_fill_mut.freeze()
                }
                Err(to_fill) => {
                    let mut to_fill_mut = BufferMut::<T>::with_capacity(to_fill.len());
                    for (v, (out, is_valid)) in to_fill.iter().zip(
                        to_fill_mut
                            .spare_capacity_mut()
                            .iter_mut()
                            .zip(bit_buffer.iter()),
                    ) {
                        if is_valid {
                            last_valid = *v;
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
}
