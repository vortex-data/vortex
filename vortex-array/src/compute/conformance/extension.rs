// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conformance checks for extension dtype implementations.
//!
//! The checks use caller-provided storage to test metadata reconstruction, value validation, and
//! storage claims independently of extension array construction.

use std::cmp::Ordering;

use vortex_error::VortexExpect;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::Claim;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtDTypeRef;
use crate::dtype::extension::ExtVTable;
use crate::dtype::extension::has_non_nullable_form;
use crate::dtype::extension::validate_each_value;

type ReferenceOrder<'a, V> = &'a dyn for<'value> Fn(
    &<V as ExtVTable>::NativeValue<'value>,
    &<V as ExtVTable>::NativeValue<'value>,
) -> Ordering;

/// Checks metadata reconstruction, value validation, and the claims of an extension dtype.
///
/// Pass storage that matches the extension dtype's declaration. Use both legal and illegal
/// values to exercise an override's acceptance and rejection paths.
///
/// When `reference_order` is supplied, the harness compares every pair of legal non-null values
/// against that order. `Ordering::Equal` must mean native equality. The harness uses the direction
/// of the ordering only for dtypes that claim [`Claim::Ord`]. Use small fixtures because the
/// number of comparisons grows quadratically with their length.
#[track_caller]
pub fn test_ext_vtable_conformance<V: ExtVTable>(
    ext_dtype: &ExtDType<V>,
    storage: &ArrayRef,
    reference_order: Option<ReferenceOrder<'_, V>>,
    ctx: &mut ExecutionCtx,
) {
    let vtable = ext_dtype.vtable();
    let bytes = ext_dtype
        .serialize_metadata()
        .vortex_expect("extension metadata must serialize");
    let stored_metadata = vtable
        .deserialize_metadata(&bytes)
        .vortex_expect("serialized extension metadata must deserialize");
    let roundtrip = ExtDType::try_from_storage_with_vtable(
        vtable.clone(),
        stored_metadata,
        storage.dtype().clone(),
    )
    .vortex_expect("extension dtype must reconstruct from stored metadata and storage");
    assert_eq!(
        &roundtrip, ext_dtype,
        "extension dtype changed on round trip"
    );

    let ext = roundtrip.erased();
    let expected = validate_each_value(ext_dtype, storage, ctx);
    let actual = ext.validate_storage_array(storage, ctx);
    assert_eq!(
        actual.is_ok(),
        expected.is_ok(),
        "{} array validation returned {actual:?}, scalar validation returned {expected:?}",
        ext_dtype.id(),
    );

    assert_claims(&ext);

    let dtype = DType::Extension(ext);
    if let Some(reference_order) = reference_order
        && dtype.holds(Claim::Eq)
    {
        assert_storage_matches_reference_order(ext_dtype, &dtype, storage, reference_order, ctx);
    }
}

#[track_caller]
fn assert_claims(ext: &ExtDTypeRef) {
    let nullable = ext.with_nullability(Nullability::Nullable);
    assert_storage_holds_claims(&nullable);

    // Null storage, including an extension over null storage, has only a nullable form.
    if has_non_nullable_form(ext.storage_dtype()) {
        let non_nullable = ext.with_nullability(Nullability::NonNullable);
        assert_storage_holds_claims(&non_nullable);

        let nullable = DType::Extension(nullable);
        let non_nullable = DType::Extension(non_nullable);
        for claim in Claim::ALL {
            assert_eq!(
                nullable.holds(claim),
                non_nullable.holds(claim),
                "{} must hold {claim} independently of top-level nullability",
                ext.id(),
            );
        }
    }
}

#[track_caller]
fn assert_storage_holds_claims(ext: &ExtDTypeRef) {
    let dtype = DType::Extension(ext.clone());
    assert!(
        !dtype.holds(Claim::Ord) || dtype.holds(Claim::Eq),
        "{dtype} must hold Eq whenever it holds Ord",
    );

    for claim in ext.claims().iter() {
        let storage_dtype = ext.storage_dtype();
        assert!(
            storage_dtype.holds(claim),
            "{dtype} claims {claim}, but its storage dtype {storage_dtype} does not hold it",
        );
    }
}

#[track_caller]
fn assert_storage_matches_reference_order<V: ExtVTable>(
    ext_dtype: &ExtDType<V>,
    dtype: &DType,
    storage: &ArrayRef,
    reference_order: ReferenceOrder<'_, V>,
    ctx: &mut ExecutionCtx,
) {
    let mut probe = storage.repeated_probe();
    let mut legal = Vec::new();
    for index in 0..storage.len() {
        let scalar = probe
            .execute_scalar(index, ctx)
            .vortex_expect("extension storage sample must be readable");
        if let Some(value) = scalar.value()
            && ext_dtype.validate_scalar_value(value).is_ok()
        {
            legal.push((index, scalar));
        }
    }

    let samples: Vec<_> = legal
        .iter()
        .map(|(index, scalar)| {
            let value = scalar.value().vortex_expect("legal sample is non-null");
            let native = V::unpack_native(ext_dtype, value)
                .vortex_expect("legal storage must unpack as a native value");
            (index, scalar, native)
        })
        .collect();
    let ordered = dtype.holds(Claim::Ord);
    for (lhs_index, lhs, lhs_native) in &samples {
        for (rhs_index, rhs, rhs_native) in &samples {
            let order = reference_order(lhs_native, rhs_native);
            assert_eq!(
                lhs == rhs,
                order == Ordering::Equal,
                "{dtype} storage equality disagrees with native equality at rows {lhs_index} and {rhs_index}",
            );
            if ordered {
                assert_eq!(
                    lhs.partial_cmp(rhs),
                    Some(order),
                    "{dtype} storage order disagrees with native order at rows {lhs_index} and {rhs_index}",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use super::test_ext_vtable_conformance;
    use crate::EmptyMetadata;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::extension::ForeignExtDType;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::DateValue;
    use crate::extension::datetime::Time;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::TimeValue;
    use crate::extension::datetime::Timestamp;
    use crate::extension::datetime::TimestampValue;
    use crate::extension::tests::DecimalLike;
    use crate::extension::tests::DivisibleInt;
    use crate::extension::tests::Divisor;
    use crate::extension::tests::Opaque;
    use crate::extension::tests::Refined;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn date_units_since_epoch(value: &DateValue) -> i64 {
        match *value {
            DateValue::Days(days) => i64::from(days),
            DateValue::Milliseconds(milliseconds) => milliseconds,
        }
    }

    fn time_units_since_midnight(value: &TimeValue) -> i64 {
        match *value {
            TimeValue::Seconds(value) | TimeValue::Milliseconds(value) => i64::from(value),
            TimeValue::Microseconds(value) | TimeValue::Nanoseconds(value) => value,
        }
    }

    fn timestamp_units_since_epoch(value: &TimestampValue<'_>) -> i64 {
        match *value {
            TimestampValue::Seconds(value, _)
            | TimestampValue::Milliseconds(value, _)
            | TimestampValue::Microseconds(value, _)
            | TimestampValue::Nanoseconds(value, _) => value,
        }
    }

    #[rstest]
    #[should_panic(expected = "storage equality disagrees with native equality")]
    #[case::equality(|_: &DateValue, _: &DateValue| Ordering::Equal)]
    #[should_panic(expected = "storage order disagrees with native order")]
    #[case::order(|lhs: &DateValue, rhs: &DateValue| {
        date_units_since_epoch(rhs).cmp(&date_units_since_epoch(lhs))
    })]
    fn claim_conformance_rejects_storage_that_disagrees_with_native_values(
        #[case] reference_order: fn(&DateValue, &DateValue) -> Ordering,
    ) {
        let dtype = Date::new(TimeUnit::Days, NonNullable);
        let storage = buffer![1i32, 2].into_array();

        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&reference_order),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[rstest]
    #[case::masked_invalid([true, false, true], true)]
    #[case::unmasked_invalid([true, true, false], false)]
    fn validate_array_default_checks_each_value(
        #[case] validity: [bool; 3],
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let dtype = Time::try_new(TimeUnit::Seconds, Nullable)?;
        let storage =
            PrimitiveArray::new(buffer![0i32, 86_401, 86_399], Validity::from_iter(validity))
                .into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = Time::validate_array(&dtype, &storage, &mut ctx);
        assert_eq!(result.is_ok(), expected, "{result:?}");
        if let Err(error) = result {
            assert!(error.to_string().contains("Invalid time scalar"), "{error}");
        }
        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&|lhs, rhs| time_units_since_midnight(lhs).cmp(&time_units_since_midnight(rhs))),
            &mut ctx,
        );
        Ok(())
    }

    #[rstest]
    #[case::valid(Some(86_399), 32, true)]
    #[case::invalid(Some(86_401), 32, false)]
    #[case::null(None, 32, true)]
    #[case::empty(Some(86_401), 0, true)]
    fn constant_time_conformance(
        #[case] value: Option<i32>,
        #[case] len: usize,
        #[case] expected: bool,
    ) -> VortexResult<()> {
        let dtype = Time::try_new(TimeUnit::Seconds, Nullable)?;
        let storage = ConstantArray::new(value, len).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = Time::validate_array(&dtype, &storage, &mut ctx);
        assert_eq!(result.is_ok(), expected, "{result:?}");
        test_ext_vtable_conformance(&dtype, &storage, None, &mut ctx);
        Ok(())
    }

    #[test]
    fn date_conformance() -> VortexResult<()> {
        let dtype = Date::try_new(TimeUnit::Days, NonNullable)?;
        let storage = buffer![-1i32, 0, 1, -1].into_array();
        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&|lhs, rhs| date_units_since_epoch(lhs).cmp(&date_units_since_epoch(rhs))),
            &mut SESSION.create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn timestamp_conformance() -> VortexResult<()> {
        let dtype =
            Timestamp::try_new_with_tz(TimeUnit::Nanoseconds, Some("UTC".into()), NonNullable)?;

        // The slice exercises validation with non-canonical storage.
        let storage = buffer![0, i64::MIN, i64::MAX].into_array().slice(1..3)?;
        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&|lhs, rhs| {
                timestamp_units_since_epoch(lhs).cmp(&timestamp_units_since_epoch(rhs))
            }),
            &mut SESSION.create_execution_ctx(),
        );
        Ok(())
    }

    #[rstest]
    #[case::nullable_values(PrimitiveArray::from_option_iter([Some(1u32), None]).into_array())]
    #[case::null_storage(NullArray::new(2).into_array())]
    fn foreign_extension_conformance(#[case] storage: crate::ArrayRef) -> VortexResult<()> {
        static ID: CachedId = CachedId::new("test.foreign_extension");
        let dtype = ExtDType::try_from_storage_with_vtable(
            ForeignExtDType::new(*ID),
            vec![0, 0xff],
            storage.dtype().clone(),
        )?;
        test_ext_vtable_conformance(&dtype, &storage, None, &mut SESSION.create_execution_ctx());
        Ok(())
    }

    #[test]
    fn divisible_int_claim_conformance() -> VortexResult<()> {
        let dtype = ExtDType::<DivisibleInt>::try_new(Divisor(3), NonNullable)?;
        let storage = buffer![9u64, 0, 6, 9].into_array();
        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&u64::cmp),
            &mut SESSION.create_execution_ctx(),
        );
        Ok(())
    }

    #[test]
    fn opaque_claim_conformance() -> VortexResult<()> {
        let dtype = ExtDType::<Opaque>::try_new(EmptyMetadata, NonNullable)?;
        let storage = buffer![4i64, -2].into_array();
        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&i64::cmp),
            &mut SESSION.create_execution_ctx(),
        );
        Ok(())
    }

    #[rstest]
    #[case::positive_scale(3)]
    #[case::negative_scale(-2)]
    fn decimal_like_claim_conformance(#[case] scale: i8) -> VortexResult<()> {
        let dtype = ExtDType::<DecimalLike>::try_new(scale, NonNullable)?;
        let storage = buffer![i64::MIN, i64::MAX, -1, 0, -1].into_array();
        test_ext_vtable_conformance(
            &dtype,
            &storage,
            Some(&i64::cmp),
            &mut SESSION.create_execution_ctx(),
        );
        Ok(())
    }

    #[rstest]
    #[case::valid(&[Some(3), None, Some(1), Some(3)], true)]
    #[case::zero(&[Some(3), Some(0), None, Some(1)], false)]
    #[case::negative(&[Some(-1), Some(1)], false)]
    #[case::all_null(&[None, None], true)]
    #[case::empty(&[], true)]
    fn refined_claim_conformance(
        #[case] values: &[Option<i64>],
        #[case] should_accept: bool,
    ) -> VortexResult<()> {
        let dtype = ExtDType::<Refined>::try_new(EmptyMetadata, Nullable)?;
        let storage = PrimitiveArray::from_option_iter(values.iter().copied()).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let result = Refined::validate_array(&dtype, &storage, &mut ctx);
        assert_eq!(result.is_ok(), should_accept, "{result:?}");
        test_ext_vtable_conformance(&dtype, &storage, Some(&i64::cmp), &mut ctx);
        Ok(())
    }
}
