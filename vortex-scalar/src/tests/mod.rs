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
use vortex_dtype::ExtDType;
use vortex_dtype::ExtID;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::extension::EmptyMetadata;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_dtype::session::DTypeSession;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::Scalar;
use crate::ScalarValue;
use crate::extension::ExtScalarVTable;

pub(crate) static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<DTypeSession>());

/// We define a dummy extension type here for testing purposes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct Even;

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

impl ExtScalarVTable for Even {
    type Value = i64;

    fn unpack(&self, _dtype: &ExtDType<Self>, storage: &ScalarValue) -> VortexResult<Self::Value> {
        let ScalarValue::Primitive(pvalue) = storage else {
            vortex_bail!("storage is not a primitive value");
        };
        Ok(pvalue.cast::<i64>())
    }

    fn pack(
        &self,
        _metadata: &Self::Metadata,
        value: Option<&Self::Value>,
        nullability: Nullability,
    ) -> VortexResult<Scalar> {
        let Some(value) = value else {
            return Ok(Scalar::null(DType::Primitive(
                PType::I64,
                Nullability::Nullable,
            )));
        };

        vortex_ensure!(
            value % 2 == 0,
            "value {} is not even, cannot pack into Even scalar",
            value
        );

        Ok(Scalar::primitive(*value, nullability))
    }
}
