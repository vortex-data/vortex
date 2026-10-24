// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extrema over an extension type whose storage values can fail validation.

use std::fmt::Debug;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFn;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::max::Max;
use crate::min::Min;
use crate::min_max::MinMax;
use crate::min_max::min_max;
use crate::min_max::nan_scalar;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[track_caller]
fn assert_non_zero_error(result: VortexResult<impl Debug>, value: &str) {
    let error = result.unwrap_err().to_string();
    let expected = format!("NonZero requires a finite nonzero value, got {value}");
    assert!(error.contains(&expected), "{error}");
}

/// A test extension that rejects zero and non-finite storage values.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct NonZero;

impl NonZero {
    fn dtype(ptype: PType, nullability: Nullability) -> VortexResult<ExtDTypeRef> {
        Ok(ExtDType::<Self>::try_new(ptype, nullability)?.erased())
    }

    fn array(storage: ArrayRef) -> VortexResult<ArrayRef> {
        let dtype = Self::dtype(storage.dtype().as_ptype(), storage.dtype().nullability())?;

        Ok(ExtensionArray::try_new(dtype, storage)?.into_array())
    }
}

impl ExtVTable for NonZero {
    type Metadata = PType;
    type StoredMetadata = ();
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.non_zero");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        Ok(())
    }

    fn metadata_from_storage(
        &self,
        _stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        PType::try_from(storage_dtype)
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        DType::Primitive(*metadata, Nullability::NonNullable)
    }

    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        vortex_ensure!(
            matches!(metadata, PType::I64 | PType::F64),
            "NonZero requires i64 or f64 storage, got {metadata}"
        );

        Ok(())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let value = storage_value.as_primitive().cast::<f64>()?;
        vortex_ensure!(
            value.is_finite() && value != 0.0,
            "NonZero requires a finite nonzero value, got {value}"
        );

        Ok(storage_value)
    }
}

#[rstest]
#[case::min(AggregateFn::new(Min, NumericalAggregateOpts::default()).erased())]
#[case::max(AggregateFn::new(Max, NumericalAggregateOpts::default()).erased())]
#[case::min_max(AggregateFn::new(MinMax, NumericalAggregateOpts::default()).erased())]
fn test_aggregates_return_extension_validation_errors(
    #[case] aggregate: AggregateFnRef,
) -> VortexResult<()> {
    let array = NonZero::array(buffer![0i64, 1].into_array())?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut accumulator = aggregate.accumulator(array.dtype())?;

    assert_non_zero_error(accumulator.accumulate(&array, &mut ctx), "0");

    Ok(())
}

#[test]
fn test_nan_extrema_return_extension_validation_errors() -> VortexResult<()> {
    let array = NonZero::array(buffer![f64::NAN, 1.0].into_array())?;
    let mut ctx = SESSION.create_execution_ctx();

    assert_non_zero_error(
        min_max(&array, &mut ctx, NumericalAggregateOpts::include_nans()),
        "NaN",
    );

    Ok(())
}

#[rstest]
fn test_nan_scalar_returns_extension_validation_errors(
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) -> VortexResult<()> {
    let dtype = DType::Extension(NonZero::dtype(PType::F64, nullability)?);

    assert_non_zero_error(nan_scalar(&dtype), "NaN");

    Ok(())
}
