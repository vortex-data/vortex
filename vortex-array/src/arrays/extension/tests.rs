// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Regression tests for extension storage validation and fallible scalar reads.

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayParts;
use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::EmptyMetadata;
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
use crate::arrays::ConstantArray;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::extension::ExtensionSlots;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::expr::ext_storage;
use crate::expr::root;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A test extension that rejects zero and non-finite storage values.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct NonZero;

impl NonZero {
    fn dtype(ptype: PType, nullability: Nullability) -> VortexResult<ExtDTypeRef> {
        Ok(
            ExtDType::<Self>::try_new(EmptyMetadata, DType::Primitive(ptype, nullability))?
                .erased(),
        )
    }

    fn array(storage: ArrayRef) -> VortexResult<ArrayRef> {
        let dtype = Self::dtype(storage.dtype().as_ptype(), storage.dtype().nullability())?;

        Ok(ExtensionArray::try_new(dtype, storage)?.into_array())
    }
}

impl ExtVTable for NonZero {
    type Metadata = EmptyMetadata;
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("test.non_zero");
        *ID
    }

    fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(vec![])
    }

    fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(EmptyMetadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        vortex_ensure!(
            matches!(
                ext_dtype.storage_dtype(),
                DType::Primitive(PType::I64 | PType::F64, _)
            ),
            "NonZero requires i64 or f64 storage, got {}",
            ext_dtype.storage_dtype()
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
    let array = NonZero::array(storage.clone())?;
    let mut ctx = VortexSession::empty().create_execution_ctx();

    let error = array.execute_scalar(0, &mut ctx).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("NonZero requires a finite nonzero value, got 0")
    );
    let error = array
        .repeated_probe()
        .execute_scalar(0, &mut ctx)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("NonZero requires a finite nonzero value, got 0")
    );

    let optimized = array.optimize()?;
    assert!(optimized.is::<Extension>());
    assert_arrays_eq!(optimized.apply(&ext_storage(root()))?, storage, &mut ctx);
    assert_arrays_eq!(
        array.slice(0..1)?.apply(&ext_storage(root()))?,
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
    let array = NonZero::array(ConstantArray::new(storage, 2).into_array())?.optimize()?;
    let mut ctx = VortexSession::empty().create_execution_ctx();

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

    assert!(Scalar::validate(&dtype, Some(&value)).is_err());
    assert!(Scalar::try_new(dtype, Some(value)).is_err());

    Ok(())
}

#[test]
fn test_extension_constructor_rejects_wrong_storage_dtype() -> VortexResult<()> {
    let dtype = NonZero::dtype(PType::I64, Nullability::NonNullable)?;

    assert!(Scalar::try_extension_ref(dtype.clone(), Scalar::from(1.0f64)).is_err());
    assert!(Scalar::try_extension_ref(dtype, Scalar::from(0i64)).is_err());

    Ok(())
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
    let array = NonZero::array(buffer![0i64, 1].into_array())?;
    let mut ctx = VortexSession::empty().create_execution_ctx();
    let mut accumulator = aggregate.accumulator(array.dtype())?;

    let error = accumulator.accumulate(&array, &mut ctx).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("NonZero requires a finite nonzero value, got 0")
    );

    Ok(())
}

#[test]
fn test_nan_extrema_return_extension_validation_errors() -> VortexResult<()> {
    let array = NonZero::array(buffer![f64::NAN, 1.0].into_array())?;
    let mut ctx = VortexSession::empty().create_execution_ctx();

    let error = min_max(&array, &mut ctx, NumericalAggregateOpts::include_nans()).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("NonZero requires a finite nonzero value, got NaN")
    );

    Ok(())
}

#[test]
fn test_nan_scalar_returns_extension_validation_errors() -> VortexResult<()> {
    let dtype = DType::Extension(NonZero::dtype(PType::F64, Nullability::Nullable)?);

    let error = nan_scalar(&dtype).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("NonZero requires a finite nonzero value, got NaN")
    );

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
