// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

pub use bitpacking::*;
pub use delta::*;
pub use r#for::*;
pub use rle::*;

mod bitpacking;
mod delta;
mod r#for;
mod rle;

pub(crate) const FL_CHUNK_SIZE: usize = 1024;

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::session::ArraySessionExt;
    use vortex_session::VortexSession;

    use super::*;

    pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        session
            .arrays()
            .register(BitPackedVTable::ID, BitPackedVTable);
        session.arrays().register(DeltaVTable::ID, DeltaVTable);
        session.arrays().register(FoRVTable::ID, FoRVTable);
        session.arrays().register(RLEVTable::ID, RLEVTable);
        session
    });
}
