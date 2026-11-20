// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Validated conversions between UTF-8 text and the [`Json`] type.
//!
//! Both directions run the same validation, so one scalar function implements them and its
//! options select the direction. The validation keeps the original text instead of serializing a
//! decoded value again, so number spelling, key order, and whitespace survive a round trip.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use serde::de::IgnoredAny;
use vortex_array::AnyColumnar;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::expr::Expression;
use vortex_array::expr::display::ExprDisplay;
use vortex_array::require_input;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ScalarFnExecutionResult;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::ext_storage::ExtStorage;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Json;
use crate::proto::json as pb;

/// The number of bytes of an invalid value that its error message quotes.
///
/// A short prefix lets users recognize the value without copying a large document into every
/// error. A larger prefix makes long values easier to find, but it makes every error longer.
const QUOTED_PREFIX_BYTES: usize = 64;

/// Labels each UTF-8 value as [`Json`] after validating that it is one complete JSON document.
///
/// The result keeps the original text and the input nullability. Null rows stay null, and a
/// non-null row that is not valid JSON fails the whole expression. Empty and whitespace-only text
/// is not a JSON document, so it fails too.
///
/// The validation covers only the JSON grammar of [RFC 8259]. It accepts any nesting depth and
/// any number magnitude, such as `1e999`, because [`Json`] stores text, not decoded values.
/// Conversions that decode values, such as [`json_to_variant`], can still reject a value that
/// passes this validation.
///
/// In Spark and Snowflake, `parse_json` returns a Variant. This function returns [`Json`] text
/// instead, and [`json_to_variant`] converts its result to a Variant.
///
/// [RFC 8259]: https://www.rfc-editor.org/rfc/rfc8259
/// [`json_to_variant`]: crate::json_to_variant()
pub fn parse_json(child: Expression) -> Expression {
    JsonText.new_expr(JsonTextDirection::Parse, [child])
}

/// Returns the original text of each [`Json`] value as UTF-8 after validating it.
///
/// [`Json`] arrays do not validate their values when they are built, so this function runs the
/// same validation as [`parse_json`], with the same null handling and errors.
pub fn format_json(child: Expression) -> Expression {
    JsonText.new_expr(JsonTextDirection::Format, [child])
}

/// The direction of a JSON text conversion, which selects the logical dtype of the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum JsonTextDirection {
    /// Converts UTF-8 input to JSON.
    Parse,
    /// Converts JSON input to UTF-8.
    Format,
}

impl Display for JsonTextDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Parse => "parse_json",
            Self::Format => "format_json",
        })
    }
}

/// The scalar function behind [`parse_json`] and [`format_json`].
#[derive(Clone)]
pub(crate) struct JsonText;

impl ScalarFnVTable for JsonText {
    type Options = JsonTextDirection;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.json_text");
        *ID
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _: &Self::Options, _: usize) -> ChildName {
        ChildName::from("input")
    }

    fn fmt_sql(
        &self,
        direction: &Self::Options,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "{direction}(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, ")")
    }

    fn serialize(&self, direction: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let direction = match direction {
            JsonTextDirection::Parse => pb::json_text_opts::Direction::Parse,
            JsonTextDirection::Format => pb::json_text_opts::Direction::Format,
        };
        Ok(Some(
            pb::JsonTextOpts {
                direction: direction.into(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, bytes: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        let opts = pb::JsonTextOpts::decode(bytes)?;
        match pb::json_text_opts::Direction::try_from(opts.direction) {
            Ok(pb::json_text_opts::Direction::Parse) => Ok(JsonTextDirection::Parse),
            Ok(pb::json_text_opts::Direction::Format) => Ok(JsonTextDirection::Format),
            _ => vortex_bail!(
                "Expected a parse or format JSON text direction, got {}",
                opts.direction
            ),
        }
    }

    fn return_dtype(&self, direction: &Self::Options, inputs: &[DType]) -> VortexResult<DType> {
        match direction {
            JsonTextDirection::Parse => parse_json_dtype(&inputs[0]).map(DType::Extension),
            JsonTextDirection::Format => format_json_dtype(&inputs[0]),
        }
    }

    fn execute(
        &self,
        direction: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        match direction {
            JsonTextDirection::Parse => execute_parse_json(input, ctx),
            JsonTextDirection::Format => execute_format_json(input, ctx),
        }
    }

    fn execute_step(
        &self,
        direction: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ScalarFnExecutionResult> {
        require_input!(args, 0 => AnyColumnar);
        self.execute(direction, args, ctx)
            .map(ScalarFnExecutionResult::done)
    }

    fn reduce<T: ReduceNode>(
        &self,
        direction: &Self::Options,
        node: &T,
    ) -> VortexResult<Option<T>> {
        let child = node.child(0);
        let Some(&child_direction) = child
            .scalar_fn()
            .and_then(|scalar_fn| scalar_fn.as_opt::<JsonText>())
        else {
            return Ok(None);
        };

        // Both conversions of a round trip validate the same text. The format becomes a plain
        // storage read, so the parse runs the only validation.
        let storage = ExtStorage.bind(EmptyOptions);
        match (direction, child_direction) {
            (JsonTextDirection::Format, JsonTextDirection::Parse) => {
                Ok(Some(node.new_node(storage, &[child])?))
            }
            (JsonTextDirection::Parse, JsonTextDirection::Format) => {
                let text = node.new_node(storage, &[child.child(0)])?;
                Ok(Some(node.new_node(
                    JsonText.bind(JsonTextDirection::Parse),
                    &[text],
                )?))
            }
            _ => Ok(None),
        }
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }
}

/// Resolves the JSON dtype for a UTF-8 input.
fn parse_json_dtype(input: &DType) -> VortexResult<ExtDTypeRef> {
    vortex_ensure!(
        matches!(input, DType::Utf8(_)),
        "Expected UTF-8 text, got {input}"
    );
    Ok(ExtDType::try_with_vtable(Json, EmptyMetadata, input.nullability())?.erased())
}

/// Resolves the UTF-8 dtype for a JSON input.
fn format_json_dtype(input: &DType) -> VortexResult<DType> {
    vortex_ensure!(
        input.as_extension_opt().is_some_and(|ext| ext.is::<Json>()),
        "Expected JSON, got {input}"
    );
    Ok(DType::Utf8(input.nullability()))
}

/// Validates UTF-8 text and wraps it as JSON.
fn execute_parse_json(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let target = parse_json_dtype(input.dtype())?;
    if input.is_empty() {
        return Ok(Canonical::empty(&DType::Extension(target)).into_array());
    }

    if let Some(text) = input.as_constant() {
        validate_json_scalar(text)?;
        let json = Scalar::try_extension_ref(target, text.clone())?;
        return Ok(ConstantArray::new(json, input.len()).into_array());
    }

    let text = validate_json_array(input, ctx)?;
    Ok(ExtensionArray::try_new(target, text)?.into_array())
}

/// Unwraps JSON and validates its UTF-8 storage.
fn execute_format_json(input: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let target = format_json_dtype(input.dtype())?;
    if input.is_empty() {
        return Ok(Canonical::empty(&target).into_array());
    }

    if let Some(json) = input.as_constant() {
        let text = json.as_extension().to_storage_scalar();
        validate_json_scalar(&text)?;
        return Ok(ConstantArray::new(text, input.len()).into_array());
    }

    let json = input.execute::<ExtensionArray>(ctx)?;
    validate_json_array(json.storage_array().clone(), ctx)
}

/// Validates a UTF-8 constant once for all rows.
fn validate_json_scalar(text: &Scalar) -> VortexResult<()> {
    if let Some(value) = text.as_utf8().value() {
        validate_json(value.as_bytes())?;
    }
    Ok(())
}

/// Validates the non-null rows and returns the canonical UTF-8 array.
fn validate_json_array(text: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let text = text.execute::<VarBinViewArray>(ctx)?;
    vortex_ensure!(
        text.as_array().is_host(),
        "JSON text validation requires host-resident buffers"
    );

    let validity = text.validity()?.execute_mask(text.len(), ctx)?;
    let views = text.resolved_views();
    for (view, valid) in views.views().iter().zip(validity.iter()) {
        if valid {
            validate_json(views.view_bytes(view))?;
        }
    }

    Ok(text.into_array())
}

/// Validates that `value` holds one complete JSON document.
///
/// [`IgnoredAny`] walks the JSON grammar without building a value. It does not validate UTF-8
/// again, because the Utf8 dtype already guarantees it.
///
/// The error cannot name the row, because the input can be one chunk of a column or the distinct
/// values of a dictionary. It quotes the start of the value and gives its length instead, so that
/// users can tell apart values that share a prefix.
fn validate_json(value: &[u8]) -> VortexResult<()> {
    let Err(error) = serde_json::from_slice::<IgnoredAny>(value) else {
        return Ok(());
    };

    let text = String::from_utf8_lossy(value);
    let prefix = &text[..text.floor_char_boundary(QUOTED_PREFIX_BYTES)];
    let ellipsis = if prefix.len() < text.len() { "..." } else { "" };
    let length = value.len();
    Err(VortexError::external(error).with_context(format!(
        "Expected one JSON document, got {prefix:?}{ellipsis} ({length} bytes)"
    )))
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::ext_storage;
    use vortex_array::expr::proto::ExprSerializeProtoExt;
    use vortex_array::expr::root;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::test_harness::ExpectedOutcome;
    use vortex_array::test_harness::assert_execution_paths;

    use super::*;
    use crate::SESSION;

    /// Returns the public expression for `direction`, applied to the root.
    fn expression(direction: JsonTextDirection) -> Expression {
        match direction {
            JsonTextDirection::Parse => parse_json(root()),
            JsonTextDirection::Format => format_json(root()),
        }
    }

    /// Builds an input of the dtype that `direction` accepts, with [`unchecked_json`] for format.
    fn input(direction: JsonTextDirection, values: &[Option<&str>]) -> VortexResult<ArrayRef> {
        match direction {
            JsonTextDirection::Parse => Ok(text(values)),
            JsonTextDirection::Format => unchecked_json(values),
        }
    }

    fn text(values: &[Option<&str>]) -> ArrayRef {
        VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()
    }

    /// Builds JSON values without [`parse_json`], so they can be invalid.
    fn unchecked_json(values: &[Option<&str>]) -> VortexResult<ArrayRef> {
        Ok(ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, text(values))?.into_array())
    }

    #[rstest]
    #[case::spelling(&[Some("  {\"b\": 1e999, \"a\": 1.0} ")])]
    #[case::nulls(&[Some("null"), None, Some("[]")])]
    #[case::all_null(&[None, None])]
    #[case::empty(&[])]
    fn test_json_text_preserves_text(#[case] values: &[Option<&str>]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let utf8 = text(values);
        let json = unchecked_json(values)?;

        for (direction, input, expected) in [
            (JsonTextDirection::Parse, &utf8, &json),
            (JsonTextDirection::Format, &json, &utf8),
        ] {
            let proto = expression(direction).serialize_proto()?;
            let expression = Expression::from_proto(&proto, &SESSION)?;
            assert_execution_paths(
                std::slice::from_ref(input),
                |inputs| inputs[0].clone().apply(&expression),
                ExpectedOutcome::Array(expected.clone()),
                &mut ctx,
            );
        }

        Ok(())
    }

    /// Validation has no depth limit, unlike the 128 levels that [`json_to_variant`] decodes.
    ///
    /// [`json_to_variant`]: crate::json_to_variant()
    #[test]
    fn test_parse_json_accepts_deep_nesting() -> VortexResult<()> {
        let nested = format!("{}{}", "[".repeat(200), "]".repeat(200));

        assert_arrays_eq!(
            text(&[Some(&nested)]).apply(&parse_json(root()))?,
            unchecked_json(&[Some(&nested)])?,
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    #[rstest]
    #[case::parse(JsonTextDirection::Parse)]
    #[case::format(JsonTextDirection::Format)]
    fn test_json_text_rejects_invalid_json(
        #[case] direction: JsonTextDirection,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = input(
            direction,
            &[Some("true false"), Some("{"), Some(""), Some(" ")],
        )?;
        let expression = expression(direction);

        assert_execution_paths(
            std::slice::from_ref(&input),
            |inputs| inputs[0].clone().apply(&expression),
            ExpectedOutcome::Error,
            &mut ctx,
        );

        let error = input
            .apply(&expression)?
            .execute::<Canonical>(&mut ctx)
            .unwrap_err();
        assert!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<serde_json::Error>())
                .is_some_and(serde_json::Error::is_syntax),
            "{error}"
        );

        Ok(())
    }

    #[rstest]
    #[case::short("true false", r#"got "true false" (10 bytes):"#)]
    // The 81-byte value is cut to its first 64 bytes, which end with 31 `1,` pairs and a `1`.
    #[case::long(
        &format!("[{}", "1,".repeat(40)),
        &format!(r#"got "[{}1"... (81 bytes):"#, "1,".repeat(31)),
    )]
    fn test_parse_json_error_quotes_invalid_value(
        #[case] value: &str,
        #[case] quote: &str,
    ) -> VortexResult<()> {
        let error = text(&[Some(value)])
            .apply(&parse_json(root()))?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())
            .unwrap_err();

        assert!(error.to_string().contains(quote), "{error}");

        Ok(())
    }

    #[rstest]
    #[case::parse_json_input(JsonTextDirection::Parse, "Expected UTF-8 text")]
    #[case::format_utf8_input(JsonTextDirection::Format, "Expected JSON")]
    fn test_json_text_rejects_wrong_input_dtype(
        #[case] direction: JsonTextDirection,
        #[case] message: &str,
    ) -> VortexResult<()> {
        let input = match direction {
            JsonTextDirection::Parse => unchecked_json(&[Some("1")])?,
            JsonTextDirection::Format => text(&[Some("1")]),
        };

        let error = input.apply(&expression(direction)).unwrap_err();

        assert!(error.to_string().contains(message), "{error}");

        Ok(())
    }

    #[rstest]
    #[case::unspecified(pb::json_text_opts::Direction::Unspecified as i32)]
    #[case::unknown(3)]
    fn test_json_text_deserialize_rejects_unknown_direction(#[case] direction: i32) {
        let bytes = pb::JsonTextOpts { direction }.encode_to_vec();

        let error = JsonText.deserialize(&bytes, &SESSION).unwrap_err();

        assert!(error.to_string().contains("JSON text direction"), "{error}");
    }

    #[rstest]
    #[case::parse(JsonTextDirection::Parse, "parse_json($)")]
    #[case::format(JsonTextDirection::Format, "format_json($)")]
    fn test_json_text_fmt_sql_names_public_function(
        #[case] direction: JsonTextDirection,
        #[case] sql: &str,
    ) {
        assert_eq!(expression(direction).to_string(), sql);
    }

    /// A round trip validates its text once, because its format becomes a plain storage read.
    #[rstest]
    #[case::format_of_parse(
        JsonTextDirection::Parse,
        format_json(parse_json(root())),
        ext_storage(parse_json(root()))
    )]
    #[case::parse_of_format(
        JsonTextDirection::Format,
        parse_json(format_json(root())),
        parse_json(ext_storage(root()))
    )]
    fn test_json_text_round_trip_validates_once(
        #[case] inner: JsonTextDirection,
        #[case] round_trip: Expression,
        #[case] expected: Expression,
    ) -> VortexResult<()> {
        let invalid = input(inner, &[Some("true false")])?;

        let optimized = round_trip.bind(invalid.dtype())?.optimize()?;
        assert_eq!(optimized, expected.bind(invalid.dtype())?);

        let result = invalid
            .apply(&round_trip)?
            .execute::<Canonical>(&mut SESSION.create_execution_ctx());
        assert!(result.is_err(), "the parse must still validate the text");

        Ok(())
    }

    /// A strict function pushes down to the values of a dictionary with null codes, so only the
    /// distinct values are validated. The function can fail, so the push-down also needs every
    /// value to be referenced.
    #[test]
    fn test_parse_json_pushes_down_through_dictionary_with_null_codes() -> VortexResult<()> {
        let codes = PrimitiveArray::from_option_iter([Some(0u8), None, Some(1), Some(0)]);
        let values = text(&[Some("{\"a\": 1}"), Some("[2]")]);
        // SAFETY: The valid codes `0` and `1` reference both dictionary values.
        let dict = unsafe {
            DictArray::try_new(codes.into_array(), values)?.set_all_values_referenced(true)
        };

        let parsed = dict
            .into_array()
            .apply(&parse_json(root()))?
            .optimize_ctx(&SESSION)?;

        assert!(parsed.is::<Dict>(), "{parsed}");
        assert_arrays_eq!(
            parsed,
            unchecked_json(&[Some("{\"a\": 1}"), None, Some("[2]"), Some("{\"a\": 1}")])?,
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }

    /// An invalid dictionary value that no row references does not fail the expression.
    #[test]
    fn test_parse_json_skips_unreferenced_dictionary_values() -> VortexResult<()> {
        let codes = PrimitiveArray::from_option_iter([Some(0u8), Some(0)]);
        let values = text(&[Some("[1]"), Some("true false")]);
        let dict = DictArray::try_new(codes.into_array(), values)?.into_array();

        assert_arrays_eq!(
            dict.apply(&parse_json(root()))?,
            unchecked_json(&[Some("[1]"), Some("[1]")])?,
            &mut SESSION.create_execution_ctx()
        );

        Ok(())
    }
}
