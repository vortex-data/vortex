// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tensor extension type.

use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_error::VortexResult;

mod metadata;
pub use metadata::FixedShapeTensorMetadata;

mod vtable;

/// The VTable for the Tensor extension type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FixedShapeTensor;

impl FixedShapeTensor {
    /// Creates a new [`Tensor`] extension type.
    ///
    /// TODO docs.
    pub fn new(
        metadata: FixedShapeTensorMetadata,
        dtype: DType,
    ) -> VortexResult<ExtDType<FixedShapeTensor>> {
        // TODO verify that the dtype matches the metadata.

        ExtDType::try_new(metadata, dtype)
    }
}
