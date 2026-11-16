// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generated JSON values for Arrow transport checks.
//!
//! Source text checks byte preservation, including whitespace and escapes. Independent semantic
//! values also check native decoding, including the distinction between JSON null and a null row.

mod tests {
    use std::sync::Arc;

    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::LargeStringArray;
    use arrow_array::StringArray;
    use arrow_array::StringViewArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::Json as ArrowJson;
    use hegel::TestCase;
    use hegel::generators as gs;
    use vortex_array::ArrayRef;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::builders::builder_with_capacity_in;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::memory::BufferAllocatorRef;
    use vortex_array::scalar::Scalar;
    use vortex_arrow::ExecuteArrowExt;
    use vortex_arrow::test_harness::DefaultExport;
    use vortex_arrow::test_harness::check_arrow;
    use vortex_arrow::test_harness::check_canonical_field;
    use vortex_arrow::test_harness::check_generated;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Json;

    /// Exact source text and a semantic value constructed without parsing that text.
    struct JsonValue {
        text: String,
        value: serde_json::Value,
    }

    struct JsonCase {
        array: ArrayRef,
        source: Vec<Option<JsonValue>>,
        nullable: bool,
    }

    fn draw_case(tc: &TestCase) -> VortexResult<JsonCase> {
        let len = tc.draw(gs::integers::<usize>().max_value(32));
        let nullable = tc.draw(gs::booleans());
        let mut builder = builder_with_capacity_in(
            &DType::Utf8(nullable.into()),
            len,
            BufferAllocatorRef::static_ref(),
        );

        let mut source = Vec::with_capacity(len);
        for _ in 0..len {
            if nullable && tc.draw(gs::booleans()) {
                builder.append_nulls(1);
                source.push(None);
            } else {
                let value = tc.draw(gs::integers::<i64>());
                let source_value = match tc.draw(gs::integers::<u8>().max_value(4)) {
                    0 => JsonValue {
                        text: format!(" {{ \"n\": {value} }} "),
                        value: serde_json::json!({"n": value}),
                    },
                    1 => JsonValue {
                        text: value.to_string(),
                        value: value.into(),
                    },
                    2 => JsonValue {
                        text: r#"[null,true,"雪", "\n\"\\"]"#.to_string(),
                        value: serde_json::json!([null, true, "雪", "\n\"\\"]),
                    },
                    3 => JsonValue {
                        text: r#" { "escaped": "\u0061", "array": [1,2] } "#.to_string(),
                        value: serde_json::json!({"escaped": "a", "array": [1, 2]}),
                    },
                    _ => JsonValue {
                        text: "null".to_string(),
                        value: serde_json::Value::Null,
                    },
                };
                builder.append_scalar(&Scalar::utf8(source_value.text.clone(), nullable.into()))?;
                source.push(Some(source_value));
            }
        }

        let array = ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, builder.finish())?
            .into_array();

        Ok(JsonCase {
            array,
            source,
            nullable,
        })
    }

    #[track_caller]
    fn check_json_round_trip(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let JsonCase {
            array,
            source,
            nullable,
        } = draw_case(tc)?;

        for expected in [
            Arc::new(StringArray::from_iter(
                source
                    .iter()
                    .map(|s| s.as_ref().map(|value| value.text.as_str())),
            )) as ArrowArrayRef,
            Arc::new(LargeStringArray::from_iter(
                source
                    .iter()
                    .map(|s| s.as_ref().map(|value| value.text.as_str())),
            )),
            Arc::new(StringViewArray::from_iter(
                source
                    .iter()
                    .map(|s| s.as_ref().map(|value| value.text.as_str())),
            )),
        ] {
            let mut field = Field::new("value", expected.data_type().clone(), nullable);
            field.try_with_extension_type(ArrowJson::default())?;
            check_arrow(array.clone(), &field, expected, &mut ctx)?;
        }

        let mut canonical = Field::new("value", DataType::Utf8View, nullable);
        canonical.try_with_extension_type(ArrowJson::default())?;
        check_canonical_field(&array, &canonical, &mut ctx)?;
        let default = array.clone().execute_arrow(None, &mut ctx)?;
        let expected: ArrowArrayRef = Arc::new(StringViewArray::from_iter(
            source
                .iter()
                .map(|s| s.as_ref().map(|value| value.text.as_str())),
        ));
        assert_eq!(default.to_data(), expected.to_data());

        check_generated(tc, array, DefaultExport::Supported, &mut ctx)
    }

    #[track_caller]
    fn check_native(tc: &TestCase) -> VortexResult<()> {
        let session = VortexSession::empty();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let case = draw_case(tc)?;
        let dtype = ExtDType::<Json>::try_new(EmptyMetadata, case.nullable.into())?;
        for (index, source) in case.source.iter().enumerate() {
            let scalar = case.array.execute_scalar(index, &mut ctx)?;
            let actual = scalar
                .value()
                .map(|v| Json::unpack_native(&dtype, v))
                .transpose()?;
            assert_eq!(actual, source.as_ref().map(|value| value.text.as_str()));
            if let (Some(actual), Some(source)) = (actual, source) {
                let actual: serde_json::Value = serde_json::from_str(actual)
                    .map_err(|e| vortex_error::vortex_err!("generated JSON must be valid: {e}"))?;
                assert_eq!(actual, source.value);
            }
        }

        Ok(())
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn native_values(tc: TestCase) {
        check_native(&tc)
            .vortex_expect("JSON native decoding must preserve source text and values");
    }

    #[hegel::test(test_cases = 100, print_blob = true)]
    fn arrow_transport(tc: TestCase) {
        check_json_round_trip(&tc)
            .vortex_expect("Arrow transport must preserve dtype and stored values");
    }
}
