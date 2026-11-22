// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated UUID values and metadata for Arrow transport checks.
//!
//! Versioned fixtures set the required version and variant bits before checking transport.
//! Independent Arrow expectations and native-value checks share the generated source bytes.

mod tests {
    use std::sync::Arc;

    use arrow_array::builder::FixedSizeBinaryBuilder;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::Uuid as ArrowUuid;
    use base64::Engine;
    use base64::prelude::BASE64_STANDARD;
    use hegel::TestCase;
    use hegel::generators as gs;
    use uuid::Version;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::aggregate_fn::ExtensionStoragePushdown;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSorted;
    use vortex_array::aggregate_fn::fns::is_sorted::IsSortedOptions;
    use vortex_array::aggregate_fn::fns::min_max::MinMax;
    use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::validity::Validity;
    use vortex_arrow::ExecuteArrowExt;
    use vortex_arrow::test_harness::DefaultExport;
    use vortex_arrow::test_harness::check_arrow;
    use vortex_arrow::test_harness::check_canonical_field;
    use vortex_arrow::test_harness::check_generated;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Uuid;
    use crate::UuidMetadata;

    struct UuidCase {
        array: ArrayRef,
        field: Field,
        arrow: arrow_array::ArrayRef,
        source: Vec<Option<[u8; 16]>>,
        dtype: ExtDType<Uuid>,
        /// Wire bytes built directly from the source version.
        metadata: Vec<u8>,
    }

    fn draw_case(tc: &TestCase, ctx: &mut ExecutionCtx) -> VortexResult<UuidCase> {
        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable = tc.draw(gs::booleans());
        let version = tc.draw(gs::sampled_from(&[
            None,
            Some(Version::Nil),
            Some(Version::Mac),
            Some(Version::Dce),
            Some(Version::Md5),
            Some(Version::Random),
            Some(Version::Sha1),
            Some(Version::SortMac),
            Some(Version::SortRand),
            Some(Version::Custom),
            Some(Version::Max),
        ]));
        let mut bytes = tc.draw(
            gs::vecs(gs::integers::<u8>())
                .min_size(len * 16)
                .max_size(len * 16),
        );

        if let Some(version) = version {
            set_uuid_version(&mut bytes, version);
        }

        let validity = if nullable {
            Validity::from_iter(tc.draw(gs::vecs(gs::booleans()).min_size(len).max_size(len)))
        } else {
            Validity::NonNullable
        };
        let mask = validity.execute_mask(len, ctx)?;
        let mut expected = FixedSizeBinaryBuilder::with_capacity(len, 16);
        let dtype = ExtDType::<Uuid>::try_new(UuidMetadata { version }, nullable.into())?;
        let metadata = version.map(|v| vec![v as u8]).unwrap_or_default();
        let mut source = Vec::with_capacity(len);
        for (i, bytes) in bytes.as_chunks::<16>().0.iter().enumerate() {
            source.push(mask.value(i).then_some(*bytes));
            if mask.value(i) {
                expected.append_value(bytes)?;
            } else {
                expected.append_null();
            }
        }
        let mut field = Field::new("value", DataType::FixedSizeBinary(16), nullable);
        field.try_with_extension_type(ArrowUuid)?;
        if version.is_some() {
            field.metadata_mut().insert(
                "vortex:extension:metadata".into(),
                BASE64_STANDARD.encode(&metadata),
            );
            field
                .metadata_mut()
                .insert("vortex:extension:metadata:encoding".into(), "base64".into());
        }
        let storage = FixedSizeListArray::try_new(
            PrimitiveArray::from_iter(bytes).into_array(),
            16,
            validity,
            len,
        )?
        .into_array();

        let array = ExtensionArray::try_new_from_vtable(Uuid, UuidMetadata { version }, storage)?
            .into_array();

        Ok(UuidCase {
            array,
            field,
            arrow: Arc::new(expected.finish()),
            source,
            dtype,
            metadata,
        })
    }

    #[track_caller]
    fn check_uuid_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let case = draw_case(tc, &mut ctx)?;
        assert_eq!(case.dtype.serialize_metadata()?, case.metadata);
        assert_eq!(
            Uuid.deserialize(&case.metadata, case.dtype.storage_dtype().clone())?,
            case.dtype.clone().erased()
        );

        check_arrow(
            case.array.clone(),
            &case.field,
            Arc::clone(&case.arrow),
            &mut ctx,
        )?;
        let default = case.array.clone().execute_arrow(None, &mut ctx)?;
        assert_eq!(default.to_data(), case.arrow.to_data());
        check_canonical_field(&case.array, &case.field, &mut ctx)?;
        check_generated(tc, case.array, DefaultExport::Supported, &mut ctx)
    }

    #[track_caller]
    fn check_native_and_order(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let case = draw_case(tc, &mut ctx)?;
        for (i, expected) in case.source.iter().enumerate() {
            let scalar = case.array.execute_scalar(i, &mut ctx)?;
            let actual = scalar
                .value()
                .map(|v| Uuid::unpack_native(&case.dtype, v))
                .transpose()?;
            assert_eq!(actual, expected.map(uuid::Uuid::from_bytes));
        }

        for strict in [false, true] {
            let expected = case
                .source
                .windows(2)
                .all(|w| if strict { w[0] < w[1] } else { w[0] <= w[1] });
            let mut acc = Accumulator::try_new(
                ExtensionStoragePushdown::new(IsSorted, Uuid),
                IsSortedOptions { strict },
                case.array.dtype().clone(),
            )?;
            acc.accumulate(&case.array, &mut ctx)?;
            assert_eq!(bool::try_from(&acc.finish()?)?, expected);
        }

        let mut acc = Accumulator::try_new(
            ExtensionStoragePushdown::new(MinMax, Uuid),
            NumericalAggregateOpts::default(),
            case.array.dtype().clone(),
        )?;
        acc.accumulate(&case.array, &mut ctx)?;
        let result = MinMaxResult::from_scalar(&acc.finish()?)?;
        let values: Vec<_> = case.source.iter().flatten().collect();
        match result {
            None => assert!(values.is_empty()),
            Some(result) => {
                for (actual, expected) in [
                    (&result.min, values.iter().min()),
                    (&result.max, values.iter().max()),
                ] {
                    let actual = Uuid::unpack_native(
                        &case.dtype,
                        actual.value().vortex_expect("nonempty min/max has a value"),
                    )?;
                    assert_eq!(
                        actual.as_bytes(),
                        *expected.vortex_expect("nonempty source has an extremum")
                    );
                }
            }
        }

        Ok(())
    }

    /// Sets version and variant bits, including the special nil and max UUID encodings.
    #[track_caller]
    fn set_uuid_version(bytes: &mut [u8], version: Version) {
        let (uuids, remainder) = bytes.as_chunks_mut::<16>();
        assert!(
            remainder.is_empty(),
            "UUID storage must contain complete UUIDs"
        );

        for uuid in uuids {
            match version {
                Version::Nil => uuid.fill(0),
                Version::Max => uuid.fill(255),
                _ => {
                    *uuid = uuid::Builder::from_bytes(*uuid)
                        .with_version(version)
                        .with_variant(uuid::Variant::RFC4122)
                        .into_uuid()
                        .into_bytes();
                }
            }
        }
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn native_and_order(tc: TestCase) {
        check_native_and_order(&tc)
            .vortex_expect("UUID native values and aggregates must agree with source bytes");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn arrow_transport(tc: TestCase) {
        check_uuid_round_trip(&tc)
            .vortex_expect("Arrow transport must preserve dtype and stored values");
    }
}
