// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;
pub use rle::*;

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
    session.arrays().register(BitPacked::ID, BitPacked);
    session.arrays().register(Delta::ID, Delta);
    session.arrays().register(FoR::ID, FoR);
    session.arrays().register(RLE::ID, RLE);

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

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::session::ArraySessionExt;
    use vortex_session::VortexSession;

    use super::*;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        session.arrays().register(BitPacked::ID, BitPacked);
        session.arrays().register(Delta::ID, Delta);
        session.arrays().register(FoR::ID, FoR);
        session.arrays().register(RLE::ID, RLE);
        session
    });
}
