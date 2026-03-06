// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tensor extension type.

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod proto;
mod vtable;

pub mod scalar_fns;

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;
