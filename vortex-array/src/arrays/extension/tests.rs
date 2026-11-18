// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression tests for extension storage validation and fallible scalar reads.

use std::fmt::Debug;
use std::sync::LazyLock;

use rstest::rstest;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::aggregate_fn::AggregateFn;
use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::max::Max;
use crate::aggregate_fn::fns::min::Min;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::min_max::nan_scalar;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::extension::ExtensionSlots;
use crate::assert_arrays_eq;
use crate::builders::ExtensionBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::extension::datetime::Time;
use crate::extension::datetime::TimeUnit;
use crate::extension::datetime::Timestamp;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[rstest]
#[case::primitive(Ok(buffer![86_401i32].into_array()), false)]
#[case::constant(Ok(ConstantArray::new(86_401i32, 2).into_array()), false)]
#[case::chunked(ChunkedArray::try_new(
    [buffer![86_399i32].into_array(), buffer![86_401i32].into_array()],
    DType::Primitive(PType::I32, Nullability::NonNullable),
).map(IntoArray::into_array), false)]
#[case::masked(Ok(PrimitiveArray::new(
    buffer![86_399i32, 86_401],
    Validity::from_iter([true, false]),
).into_array()), true)]
fn try_new_rejects_illegal_values(
    #[case] storage: VortexResult<ArrayRef>,
    #[case] expected: bool,
) -> VortexResult<()> {
    let storage = storage?;
    let dtype = Time::try_new(TimeUnit::Seconds, storage.dtype().nullability())?.erased();
    let result = ExtensionArray::try_new(dtype, storage, &mut SESSION.create_execution_ctx());
    assert_eq!(result.is_ok(), expected, "{result:?}");
    Ok(())
}

#[test]
fn try_new_structural_checks_dtype_only() -> VortexResult<()> {
    let dtype = Time::try_new(TimeUnit::Seconds, Nullability::NonNullable)?.erased();
    let storage = buffer![86_401i32].into_array();
    let array = ExtensionArray::try_new_structural(dtype.clone(), storage.clone())?;
    assert!(ArrayRef::ptr_eq(array.storage_array(), &storage));
    assert!(ExtensionArray::try_new_structural(dtype, buffer![1i64].into_array()).is_err());

    Ok(())
}

#[test]
fn builder_rejects_scalar_of_other_extension_type() -> VortexResult<()> {
    let dtype = NonZero::dtype(PType::I64, Nullability::NonNullable)?;
    let timestamp = Scalar::extension::<Timestamp>(
        Timestamp::try_new_with_tz(TimeUnit::Seconds, None, Nullability::NonNullable)?
            .metadata()
            .clone(),
        Scalar::from(0i64),
    );
    let mut builder = ExtensionBuilder::new_in(dtype.clone(), BufferAllocatorRef::static_ref());

    let error = builder
        .append_value(timestamp.as_extension())
        .unwrap_err()
        .to_string();
    assert!(error.contains(&dtype.to_string()), "{error}");
    assert!(error.contains(&timestamp.dtype().to_string()), "{error}");
    Ok(())
}

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

    /// Builds structurally valid storage as the file reader would.
    fn decoded_array(storage: ArrayRef) -> VortexResult<ArrayRef> {
        let dtype = Self::dtype(storage.dtype().as_ptype(), storage.dtype().nullability())?;

        Ok(ExtensionArray::try_from_parts(ArrayParts::new(
            Extension,
            DType::Extension(dtype),
            storage.len(),
            EmptyArrayData,
            ExtensionSlots { storage }.into_slots(),
        ))?
        .into_array())
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
#[case::primitive(buffer![0i64, 1].into_array())]
#[case::constant(ConstantArray::new(0i64, 2).into_array())]
fn test_invalid_values_return_read_errors(#[case] storage: ArrayRef) -> VortexResult<()> {
    let array = NonZero::decoded_array(storage.clone())?;
    let mut ctx = SESSION.create_execution_ctx();

    assert_non_zero_error(array.execute_scalar(0, &mut ctx), "0");
    assert_non_zero_error(array.repeated_probe().execute_scalar(0, &mut ctx), "0");

    let optimized = array.optimize_ctx(&SESSION)?;
    assert!(optimized.is::<Extension>());
    assert_arrays_eq!(
        optimized.as_::<Extension>().storage_array().clone(),
        storage,
        &mut ctx
    );
    let sliced = array.slice(0..1)?.optimize_ctx(&SESSION)?;
    assert_arrays_eq!(
        sliced.as_::<Extension>().storage_array().clone(),
        buffer![0i64].into_array(),
        &mut ctx
    );

    Ok(())
}

#[rstest]
#[case::valid(Scalar::primitive(1i64, Nullability::Nullable))]
#[case::null(Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)))]
fn test_valid_values_keep_the_extension_dtype(#[case] storage: Scalar) -> VortexResult<()> {
    let ext_dtype = NonZero::dtype(PType::I64, Nullability::Nullable)?;
    let scalar = Scalar::try_extension_ref(ext_dtype.clone(), storage.clone())?;
    let mut ctx = SESSION.create_execution_ctx();
    let array = ExtensionArray::try_new(
        ext_dtype.clone(),
        ConstantArray::new(storage, 2).into_array(),
        &mut ctx,
    )?
    .into_array()
    .optimize_ctx(&SESSION)?;

    assert_eq!(scalar.dtype(), &DType::Extension(ext_dtype));
    assert_eq!(array.execute_scalar(0, &mut ctx)?, scalar);
    assert_arrays_eq!(array, ConstantArray::new(scalar, 2), &mut ctx);

    Ok(())
}

#[rstest]
#[case::wrong_kind(ScalarValue::Bool(true))]
#[case::wrong_primitive(ScalarValue::Primitive(PValue::F64(1.0)))]
fn test_scalar_values_require_the_storage_dtype(#[case] value: ScalarValue) -> VortexResult<()> {
    let dtype = DType::Extension(NonZero::dtype(PType::I64, Nullability::NonNullable)?);

    assert!(Scalar::try_new(dtype, Some(value)).is_err());

    Ok(())
}

#[rstest]
#[case::wrong_dtype(
    Scalar::from(1.0f64),
    "test.non_zero storage must match its declaration\n  left: i64\n right: f64"
)]
#[case::invalid_value(Scalar::from(0i64), "NonZero requires a finite nonzero value, got 0")]
fn test_extension_constructor_rejects_invalid_storage(
    #[case] storage: Scalar,
    #[case] expected: &str,
) -> VortexResult<()> {
    let dtype = NonZero::dtype(PType::I64, Nullability::NonNullable)?;

    let result = Scalar::try_extension_ref(dtype, storage);
    assert!(result.is_err_and(|error| error.to_string().contains(expected)));

    Ok(())
}

/// Building an extension array over storage of another dtype fails with an error that names the
/// type and both storage dtypes.
#[test]
fn test_array_constructor_rejects_other_storage_dtype() {
    let result = ExtensionArray::try_new_from_vtable(
        NonZero,
        PType::I64,
        buffer![1f64].into_array(),
        &mut SESSION.create_execution_ctx(),
    );
    assert!(result.is_err_and(|error| {
        error
            .to_string()
            .contains("test.non_zero storage must match its declaration\n  left: i64\n right: f64")
    }));
}

#[rstest]
#[case::wrong_kind(ConstantArray::new("value", 1).into_array())]
#[case::wrong_primitive(buffer![1f64].into_array())]
#[case::wrong_nullability(PrimitiveArray::from_option_iter([Some(1i64)]).into_array())]
#[case::empty(PrimitiveArray::from_iter(std::iter::empty::<f64>()).into_array())]
fn test_array_parts_require_the_storage_dtype(#[case] storage: ArrayRef) -> VortexResult<()> {
    let dtype = NonZero::dtype(PType::I64, Nullability::NonNullable)?;
    let parts = ArrayParts::new(
        Extension,
        DType::Extension(dtype),
        storage.len(),
        EmptyArrayData,
        ExtensionSlots { storage }.into_slots(),
    );

    assert!(ExtensionArray::try_from_parts(parts).is_err());

    Ok(())
}

#[rstest]
#[case::min(AggregateFn::new(Min, NumericalAggregateOpts::default()).erased())]
#[case::max(AggregateFn::new(Max, NumericalAggregateOpts::default()).erased())]
#[case::min_max(AggregateFn::new(MinMax, NumericalAggregateOpts::default()).erased())]
fn test_aggregates_return_extension_validation_errors(
    #[case] aggregate: AggregateFnRef,
) -> VortexResult<()> {
    let array = NonZero::decoded_array(buffer![0i64, 1].into_array())?;
    let mut ctx = SESSION.create_execution_ctx();
    let mut accumulator = aggregate.accumulator(array.dtype())?;

    assert_non_zero_error(accumulator.accumulate(&array, &mut ctx), "0");

    Ok(())
}

#[test]
fn test_nan_extrema_return_extension_validation_errors() -> VortexResult<()> {
    let array = NonZero::decoded_array(buffer![f64::NAN, 1.0].into_array())?;
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

#[rstest]
#[case::extension(false)]
#[case::nested(true)]
#[should_panic(expected = "has no non-null zero value")]
fn test_rejected_extension_zero_cannot_construct_a_scalar(
    #[case] nested: bool,
    #[values(Nullability::NonNullable, Nullability::Nullable)] nullability: Nullability,
) {
    let dtype = NonZero::dtype(PType::I64, nullability).expect("NonZero accepts i64 storage");
    let mut dtype = DType::Extension(dtype);
    if nested {
        dtype = DType::struct_([("value", dtype)], Nullability::NonNullable);
    }

    Scalar::zero_value(&dtype);
}

/// Storage-based construction preserves outer nullability and scalar validation.
#[rstest]
#[case::non_nullable(Nullability::NonNullable)]
#[case::nullable(Nullability::Nullable)]
fn storage_constructors(#[case] nullability: Nullability) -> VortexResult<()> {
    let storage = Scalar::primitive(1i64, nullability);
    let scalar = Scalar::try_extension_from_storage::<NonZero>((), storage.clone())?;
    assert_eq!(scalar.dtype().nullability(), nullability);

    let array = ConstantArray::new(storage, 2).into_array();
    let result = ExtensionArray::try_from_storage::<NonZero>(
        (),
        array.clone(),
        &mut SESSION.create_execution_ctx(),
    )?;
    assert_eq!(result.ext_dtype().storage_dtype(), array.dtype());

    assert!(
        Scalar::try_extension_from_storage::<NonZero>((), Scalar::primitive(0i64, nullability))
            .is_err()
    );

    Ok(())
}

#[test]
fn storage_constructor_accepts_null_scalar() -> VortexResult<()> {
    let null = Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable));
    assert!(Scalar::try_extension_from_storage::<NonZero>((), null)?.is_null());

    Ok(())
}
