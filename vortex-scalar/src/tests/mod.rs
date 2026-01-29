// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test modules for the vortex-scalar crate.

mod casting;
mod consistency;
mod nested;
mod nullability;
mod primitives;
mod round_trip;

use std::sync::LazyLock;

use vortex_dtype::DType;
use vortex_dtype::ExtID;
use vortex_dtype::extension::EmptyMetadata;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_dtype::session::DTypeSession;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

pub(crate) static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());

/// We define a dummy extension type here for testing purposes.
#[derive(Debug, Clone, Default)]
struct Even;

impl ExtDTypeVTable for Even {
    type Metadata = EmptyMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref("test.even")
    }

    fn validate(&self, _options: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(storage_dtype.is_primitive());
        Ok(())
    }
}
