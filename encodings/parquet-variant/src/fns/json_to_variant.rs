// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.json_to_variant` scalar function.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::FieldRef;
use parquet_variant_compute::ShreddedSchemaBuilder;
use parquet_variant_compute::shred_variant;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_json::Json;
use vortex_proto::expr as pb;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ParquetVariant;
use crate::kernel::to_parquet_variant_path;

/// Parses JSON strings into Parquet Variant values, optionally shredding fields.
///
/// Accepts `Utf8` inputs or [`Json`] extension inputs and returns `Variant` values with the
/// input's nullability. Null rows stay null, the JSON literal `null` becomes a variant-null
/// value, and any row that fails to parse as JSON fails the whole expression.
///
/// A non-empty [`ShreddingSpec`] additionally shreds the selected paths into a typed
/// `typed_value` child, following the [Parquet Variant shredding] rules: rows whose value does
/// not match the requested type stay readable through the residual variant value.
///
/// # Not an inverse of `variant_to_json`
///
/// `json_to_variant` and [`VariantToJson`](crate::VariantToJson) are lossy, normalizing
/// conversions and are NOT inverses of each other:
/// - JSON whitespace is not preserved.
/// - Object keys may be reordered: Variant metadata stores keys in sorted order, so
///   `variant_to_json` emits fields in a canonical order, not source order.
/// - Number formatting and precision change: e.g. `1.0` may render as `1`, exponent forms and
///   very large numbers are re-rendered, and floating-point values are re-encoded.
/// - Duplicate object keys are collapsed to a single entry.
/// - Unicode escape sequences are normalized (e.g. `\u0041` becomes `A`).
/// - `variant_to_json` stringifies Variant-only types — date, timestamp, UUID, binary,
///   decimal — so `json_to_variant(variant_to_json(v))` yields plain strings/numbers and loses
///   the original type information.
/// - Shredding structure is not recoverable from JSON: `variant_to_json` unshreds its input
///   first, and re-parsing produces an unshredded Variant unless a new [`ShreddingSpec`] is
///   supplied.
///
/// [Parquet Variant shredding]: https://github.com/apache/parquet-format/blob/master/VariantShredding.md
#[derive(Clone)]
pub struct JsonToVariant;

impl ScalarFnVTable for JsonToVariant {
    type Options = JsonToVariantOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.json_to_variant");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let shredding = options
            .shredding()
            .fields()
            .iter()
            .map(|(path, dtype)| {
                Ok(pb::ShreddingSpecField {
                    path: path
                        .elements()
                        .iter()
                        .map(VariantPathElement::to_proto)
                        .collect(),
                    dtype: Some(dtype.try_into()?),
                })
            })
            .collect::<VortexResult<_>>()?;

        Ok(Some(pb::JsonToVariantOpts { shredding }.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let opts = pb::JsonToVariantOpts::decode(metadata)?;
        let fields = opts
            .shredding
            .into_iter()
            .map(|field| {
                let path = field
                    .path
                    .into_iter()
                    .map(VariantPathElement::from_proto)
                    .collect::<VortexResult<VariantPath>>()?;
                let dtype = field
                    .dtype
                    .as_ref()
                    .ok_or_else(|| vortex_err!("ShreddingSpecField missing dtype"))
                    .and_then(|dtype| DType::from_proto(dtype, session))?;
                Ok((path, dtype))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(JsonToVariantOptions::new(ShreddingSpec::try_new(fields)?))
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for JsonToVariant expression"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];
        vortex_ensure!(
            input_dtype.is_utf8()
                || input_dtype
                    .as_extension_opt()
                    .is_some_and(|ext_dtype| ext_dtype.is::<Json>()),
            "JsonToVariant input must be Utf8 or a Json extension, found {input_dtype}"
        );

        Ok(DType::Variant(input_dtype.nullability()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let input_nullable = input.dtype().is_nullable();

        let storage = if input.dtype().is_utf8() {
            input
        } else if input
            .dtype()
            .as_extension_opt()
            .is_some_and(|ext_dtype| ext_dtype.is::<Json>())
        {
            input
                .execute::<ExtensionArray>(ctx)?
                .storage_array()
                .clone()
        } else {
            vortex_bail!(
                "JsonToVariant input must be Utf8 or a Json extension, found {}",
                input.dtype()
            );
        };

        let session = ctx.session().clone();
        let arrow_strings = session.arrow().execute_arrow(storage, None, ctx)?;
        // Any row that fails to parse as JSON fails the whole conversion.
        let arrow_variant = parquet_variant_compute::json_to_variant(&arrow_strings)?;

        let arrow_variant = if options.shredding().is_empty() {
            arrow_variant
        } else {
            let mut builder = ShreddedSchemaBuilder::new();
            for (path, dtype) in options.shredding().fields() {
                let field: FieldRef = Arc::new(session.arrow().to_arrow_field("shredded", dtype)?);
                builder = builder.with_path(to_parquet_variant_path(path)?, field)?;
            }
            shred_variant(&arrow_variant, &builder.build())?
        };

        if input_nullable {
            ParquetVariant::from_arrow_variant_nullable(&arrow_variant)
        } else {
            ParquetVariant::from_arrow_variant(&arrow_variant)
        }
    }
}

/// A list of `(path, dtype)` directives describing which Variant paths to shred and as what
/// type.
///
/// Paths must contain only object-field elements; list index elements are rejected because
/// Parquet Variant shredding schemas cannot express list element shredding yet. The root path
/// (`$`) shreds the top-level value itself. When entries overlap (e.g. `$.a` and `$.a.b`),
/// later entries overwrite earlier ones.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShreddingSpec(Vec<(VariantPath, DType)>);

impl ShreddingSpec {
    /// Creates an empty spec, meaning no shredding is performed.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Creates a spec from `(path, dtype)` directives.
    ///
    /// # Errors
    ///
    /// Returns an error if any path contains a list index element.
    pub fn try_new(fields: impl IntoIterator<Item = (VariantPath, DType)>) -> VortexResult<Self> {
        let fields = fields.into_iter().collect::<Vec<_>>();
        for (path, _) in &fields {
            vortex_ensure!(
                path.elements()
                    .iter()
                    .all(|element| matches!(element, VariantPathElement::Field(_))),
                "ShreddingSpec paths must only contain object fields, found list index in {path}"
            );
        }
        Ok(Self(fields))
    }

    /// Returns the `(path, dtype)` directives.
    pub fn fields(&self) -> &[(VariantPath, DType)] {
        &self.0
    }

    /// Returns whether this spec contains no directives.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Display for ShreddingSpec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for (idx, (path, dtype)) in self.0.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{path} as {dtype}")?;
        }
        Ok(())
    }
}

/// Options for [`JsonToVariant`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JsonToVariantOptions {
    /// The paths to shred into typed storage, if any.
    shredding: ShreddingSpec,
}

impl JsonToVariantOptions {
    /// Creates options that shred the paths selected by `shredding`.
    pub fn new(shredding: ShreddingSpec) -> Self {
        Self { shredding }
    }

    /// Creates options that perform no shredding.
    pub fn unshredded() -> Self {
        Self::new(ShreddingSpec::empty())
    }

    /// Returns the shredding spec.
    pub fn shredding(&self) -> &ShreddingSpec {
        &self.shredding
    }
}

impl Display for JsonToVariantOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.shredding.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::Canonical;
    use vortex_array::EmptyMetadata;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::arrow::ArrowSession;
    use vortex_array::assert_arrays_eq;
    use vortex_array::assert_nth_scalar_is_null;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::session::DTypeSession;
    use vortex_array::expr::Expression;
    use vortex_array::expr::proto::ExprSerializeProtoExt;
    use vortex_array::expr::root;
    use vortex_array::expr::variant_get;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::session::ArraySession;
    use vortex_error::vortex_bail;

    use super::*;
    use crate::ParquetVariantArrayExt;
    use crate::fns::json_to_variant;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty()
            .with::<ArraySession>()
            .with::<ArrowSession>()
            .with::<DTypeSession>()
            .with::<ScalarFnSession>();
        crate::initialize(&session);
        session
    });

    fn i64_dtype() -> DType {
        DType::Primitive(PType::I64, Nullability::Nullable)
    }

    fn shred_field_as_i64(field: &str) -> VortexResult<ShreddingSpec> {
        ShreddingSpec::try_new([(VariantPath::field(field), i64_dtype())])
    }

    fn execute_json_to_variant(
        input: ArrayRef,
        shredding: ShreddingSpec,
    ) -> VortexResult<ArrayRef> {
        let expr = json_to_variant(root(), shredding);
        input
            .apply(&expr)?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    fn assert_variant_i64_rows(array: &ArrayRef, expected: &[Option<i64>]) -> VortexResult<()> {
        assert_eq!(array.len(), expected.len());
        let mut ctx = SESSION.create_execution_ctx();
        for (idx, expected) in expected.iter().enumerate() {
            let scalar = array.execute_scalar(idx, &mut ctx)?;
            let variant = scalar.as_variant();
            match expected {
                Some(expected) => {
                    let value = variant
                        .value()
                        .ok_or_else(|| vortex_err!("expected non-null variant at row {idx}"))?
                        .cast(&i64_dtype())?;
                    assert_eq!(value.as_primitive().typed_value::<i64>(), Some(*expected));
                }
                None => assert!(scalar.is_null(), "expected null row {idx}"),
            }
        }
        Ok(())
    }

    #[test]
    fn expression_roundtrip_serialization() -> VortexResult<()> {
        let expr: Expression = json_to_variant(root(), shred_field_as_i64("a")?);
        let proto = expr.serialize_proto()?;
        let actual = Expression::from_proto(&proto, &SESSION)?;

        assert_eq!(actual, expr);
        Ok(())
    }

    #[test]
    fn converts_utf8_json_rows() -> VortexResult<()> {
        let input =
            VarBinViewArray::from_iter_str([r#"{"a": 1}"#, "2", r#"{"a": 3}"#]).into_array();

        let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

        assert_eq!(result.dtype(), &DType::Variant(Nullability::NonNullable));
        let mut ctx = SESSION.create_execution_ctx();
        let row0 = result.execute_scalar(0, &mut ctx)?;
        let object = row0
            .as_variant()
            .value()
            .ok_or_else(|| vortex_err!("expected non-null variant"))?;
        let field = object
            .as_struct()
            .field("a")
            .ok_or_else(|| vortex_err!("expected field a"))?
            .as_variant()
            .value()
            .ok_or_else(|| vortex_err!("expected non-null field a"))?
            .cast(&i64_dtype())?;
        assert_eq!(field.as_primitive().typed_value::<i64>(), Some(1));

        let row1 = result.execute_scalar(1, &mut ctx)?;
        let value = row1
            .as_variant()
            .value()
            .ok_or_else(|| vortex_err!("expected non-null variant"))?
            .cast(&i64_dtype())?;
        assert_eq!(value.as_primitive().typed_value::<i64>(), Some(2));
        Ok(())
    }

    #[test]
    fn converts_json_extension_input() -> VortexResult<()> {
        let storage = VarBinViewArray::from_iter_str(["1", "2"]).into_array();
        let input = ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage)?.into_array();

        let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

        assert_eq!(result.dtype(), &DType::Variant(Nullability::NonNullable));
        assert_variant_i64_rows(&result, &[Some(1), Some(2)])
    }

    #[test]
    fn null_rows_stay_null_and_json_null_becomes_variant_null() -> VortexResult<()> {
        let input =
            VarBinViewArray::from_iter_nullable_str([Some("1"), None, Some("null")]).into_array();

        let result = execute_json_to_variant(input, ShreddingSpec::empty())?;

        assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
        let mut ctx = SESSION.create_execution_ctx();
        assert!(!result.execute_scalar(0, &mut ctx)?.is_null());
        assert_nth_scalar_is_null!(result, 1);
        let row2 = result.execute_scalar(2, &mut ctx)?;
        assert!(!row2.is_null(), "JSON null must not be a row null");
        assert_eq!(row2.as_variant().is_variant_null(), Some(true));
        Ok(())
    }

    #[test]
    fn invalid_json_errors() {
        let input = VarBinViewArray::from_iter_str([r#"{"a": 1}"#, r#"{"a":"#]).into_array();

        let err = execute_json_to_variant(input, ShreddingSpec::empty()).unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn shredding_spec_rejects_index_paths() {
        let err = ShreddingSpec::try_new([(
            VariantPath::new([VariantPathElement::field("a"), VariantPathElement::index(0)]),
            i64_dtype(),
        )])
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("ShreddingSpec paths must only contain object fields"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn shredding_produces_typed_value_child() -> VortexResult<()> {
        let input = VarBinViewArray::from_iter_str([
            r#"{"a": 1, "b": "x"}"#,
            r#"{"a": 2, "b": "y"}"#,
            r#"{"a": "not-a-number", "b": "z"}"#,
            r#"{"b": "missing-a"}"#,
        ])
        .into_array();

        let result = execute_json_to_variant(input, shred_field_as_i64("a")?)?;

        assert!(
            result.as_::<ParquetVariant>().typed_value_array().is_some(),
            "expected shredded typed_value child"
        );

        // The canonical form must expose field `a` through the shredded tree.
        let mut ctx = SESSION.create_execution_ctx();
        let Canonical::Variant(canonical) = result.clone().execute::<Canonical>(&mut ctx)? else {
            vortex_bail!("expected canonical variant array");
        };
        let shredded = canonical
            .shredded()
            .ok_or_else(|| vortex_err!("expected canonical shredded child"))?
            .clone()
            .execute::<StructArray>(&mut ctx)?;
        assert!(shredded.unmasked_field_by_name_opt("a").is_some());

        // Typed extraction must serve shredded rows and fall back for mismatched rows.
        let typed = result
            .clone()
            .apply(&variant_get(
                root(),
                VariantPath::field("a"),
                Some(i64_dtype()),
            ))?
            .execute::<ArrayRef>(&mut ctx)?;
        assert_arrays_eq!(
            typed,
            PrimitiveArray::from_option_iter([Some(1i64), Some(2), None, None])
        );

        // Mismatched rows keep their original value through the variant fallback.
        let untyped = result
            .apply(&variant_get(
                root(),
                VariantPath::field("a"),
                Some(DType::Utf8(Nullability::Nullable)),
            ))?
            .execute::<ArrayRef>(&mut ctx)?;
        let row2 = untyped.execute_scalar(2, &mut ctx)?;
        assert_eq!(
            row2.as_utf8().value().map(|value| value.to_string()),
            Some("not-a-number".to_string())
        );
        Ok(())
    }

    #[test]
    fn shredding_preserves_null_rows() -> VortexResult<()> {
        let input = VarBinViewArray::from_iter_nullable_str([
            Some(r#"{"a": 1}"#),
            None,
            Some(r#"{"a": 3}"#),
        ])
        .into_array();

        let result = execute_json_to_variant(input, shred_field_as_i64("a")?)?;

        assert_eq!(result.dtype(), &DType::Variant(Nullability::Nullable));
        assert_nth_scalar_is_null!(result, 1);
        let mut ctx = SESSION.create_execution_ctx();
        let typed = result
            .apply(&variant_get(
                root(),
                VariantPath::field("a"),
                Some(i64_dtype()),
            ))?
            .execute::<ArrayRef>(&mut ctx)?;
        assert_arrays_eq!(
            typed,
            PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)])
        );
        Ok(())
    }

    #[test]
    fn shredding_root_path_shreds_top_level_values() -> VortexResult<()> {
        let input = VarBinViewArray::from_iter_str(["1", "2", r#""not-a-number""#]).into_array();
        let spec = ShreddingSpec::try_new([(VariantPath::root(), i64_dtype())])?;

        let result = execute_json_to_variant(input, spec)?;

        assert!(
            result.as_::<ParquetVariant>().typed_value_array().is_some(),
            "expected shredded typed_value child"
        );
        assert_variant_i64_rows(&result.slice(0..2)?, &[Some(1), Some(2)])?;
        let mut ctx = SESSION.create_execution_ctx();
        let row2 = result.execute_scalar(2, &mut ctx)?;
        let value = row2
            .as_variant()
            .value()
            .ok_or_else(|| vortex_err!("expected non-null variant"))?;
        assert_eq!(
            value.as_utf8().value().map(|value| value.to_string()),
            Some("not-a-number".to_string())
        );
        Ok(())
    }
}
