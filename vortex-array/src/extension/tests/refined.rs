// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A positive integer fixture with a batch validator based on the storage minimum.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::DeserializeMetadata;
use crate::EmptyMetadata;
use crate::ExecutionCtx;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min::Min;
use crate::dtype::Claim;
use crate::dtype::Claims;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::scalar::ScalarValue;

/// An extension dtype that restricts `i64` storage to positive values.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Refined;

impl Refined {
    fn validate_positive(value: i64) -> VortexResult<i64> {
        vortex_ensure!(value > 0, "refined integer must be positive, got {value}");
        Ok(value)
    }
}

impl ExtVTable for Refined {
    type Metadata = EmptyMetadata;
    type StoredMetadata = Self::Metadata;
    type NativeValue<'a> = i64;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.refined");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(Vec::new())
    }

    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::StoredMetadata> {
        EmptyMetadata::deserialize(bytes)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        _storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(stored_metadata)
    }

    fn storage_dtype(_metadata: &Self::Metadata) -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn claims(_metadata: &Self::Metadata) -> Claims {
        Claims::none().with(Claim::Ord)
    }

    fn validate_array(
        ext_dtype: &ExtDType<Self>,
        storage: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure_eq!(ext_dtype.storage_dtype(), storage.dtype());

        let mut min = Accumulator::try_new(
            Min,
            NumericalAggregateOpts::default(),
            storage.dtype().clone(),
        )?;
        min.accumulate(storage, ctx)?;
        if let Some(value) = min.finish()?.as_primitive().typed_value::<i64>() {
            Self::validate_positive(value)?;
        }

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        Self::validate_positive(storage_value.as_primitive().cast::<i64>()?)
    }
}
