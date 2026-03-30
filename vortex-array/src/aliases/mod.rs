// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Re-exports of third-party crates we use in macros exported from vortex-array.
//!

pub mod paste {
    //! Re-export of [`paste`](https://docs.rs/paste/latest/paste/).
    pub use paste::paste;
}

// Re-export of [`inventory`](https://docs.rs/inventory/latest/inventory/).
pub use inventory;

pub mod vortex_error {
    //! Re-export of [`vortex_error`](https://docs.rs/vortex-error/latest/vortex_error/).
    pub use vortex_error::VortexExpect;
}
