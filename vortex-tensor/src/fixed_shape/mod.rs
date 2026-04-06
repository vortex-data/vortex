// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fixed-shape Tensor extension type.

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod proto;
mod vtable;

// TODO(connor): Add a dedicated `AnyFixedShapeTensor` that also contains the element ptype and
// the storage fixed size list size (which is just the product of all logical shapes).
