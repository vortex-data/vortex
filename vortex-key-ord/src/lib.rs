// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stream-kernel architecture for key-ordered operations.
//!
//! Adds a third class of operator alongside Vortex's scalar and aggregate
//! functions for ops that carry state across batches and exploit
//! encoding-aware fast paths (OVC, sort-merge join).
//!
//! See module docs for [`stream_kernel`] (the core trait), [`ovc_scalarfn`]
//! (production-shape `Ovc: ScalarFnVTable`), [`binary_kernel`] (two-VTable
//! dispatch for SMJ), [`smj`] (sort-merge join), and [`progression`]
//! (arithmetic-progression encoding used as SMJ Cartesian output).
//!
//! Research-stage; `publish = false`. The production path is to promote
//! `OvcKernel` to `vortex-array/src/scalar_fn/fns/ovc/` mirroring
//! `CastKernel`, and wire kernels into encodings' `PARENT_KERNELS`.

pub mod binary_kernel;
pub mod ovc_scalarfn;
pub mod progression;
pub mod smj;
pub mod stream_kernel;

pub mod prelude {
    pub use crate::binary_kernel::BinaryKernel;
    pub use crate::binary_kernel::BinaryKernelSet;
    pub use crate::ovc_scalarfn::Ovc;
    pub use crate::ovc_scalarfn::ovc;
    pub use crate::ovc_scalarfn::register_ovc;
    pub use crate::progression::Progression;
    pub use crate::progression::ProgressionArray;
    pub use crate::stream_kernel::CHUNKED_OVC_KERNELS;
    pub use crate::stream_kernel::CONSTANT_OVC_KERNELS;
    pub use crate::stream_kernel::DICT_OVC_KERNELS;
    pub use crate::stream_kernel::OvcKernel;
    pub use crate::stream_kernel::PRIMITIVE_OVC_KERNELS;
}
