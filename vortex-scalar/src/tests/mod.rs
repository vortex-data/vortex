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

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;

// TODO(v2): will be used when tests are re-enabled
#[allow(dead_code)]
pub(crate) static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());

/// We define a dummy extension type here for testing purposes.
// TODO(v2): will be used when tests are re-enabled
#[allow(dead_code)]
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct Even;

impl ExtDTypeVTable for Even {
    type Metadata = EmptyMetadata;

    fn id(&self) -> ExtID {
        ExtID::new_ref("test.even")
    }

    fn validate_dtype(&self, _options: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(storage_dtype.is_primitive());
        Ok(())
    }
}

impl ExtScalarVTable for Even {
    type Value<'a> = Option<i64>;

    fn unpack<'a>(
        &self,
        _metadata: &'a EmptyMetadata,
        _storage_dtype: &'a DType,
        storage_value: Option<&'a ScalarValue>,
    ) -> Self::Value<'a> {
        let storage_value = storage_value?;
        let pvalue = storage_value.as_primitive();
        Some(pvalue.cast::<i64>())
    }

    fn fmt_scalar(
        &self,
        _metadata: &EmptyMetadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let pvalue = storage_value.as_primitive();
        write!(f, "{}", pvalue.cast::<i64>())
    }

    fn validate_scalar(
        &self,
        _metadata: &EmptyMetadata,
        _storage_dtype: &DType,
        _storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        Ok(())
    }
}
