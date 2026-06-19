// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.variant_to_json` scalar function.

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Fields;
use vortex_array::ArrayRef;
use vortex_array::EmptyMetadata;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_json::Json;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::arrow::export_storage_to_target;
use crate::arrow::export_unshredded_storage_to_target;
use crate::arrow::parquet_variant_for_export;

/// Renders Variant values as JSON strings with the [`Json`] extension dtype.
///
/// Accepts `Variant` inputs backed by Parquet Variant storage, including shredded storage
/// (top-level, object, and nested fields), which is unshredded before rendering. Null rows stay
/// null, while variant-null values render as the JSON literal `null`. The output nullability
/// matches the input's.
///
/// Inputs are exported through their Parquet Variant storage, so a `Variant` whose core storage
/// is not Parquet Variant-backed is not supported and fails.
///
/// # Not an inverse of `json_to_variant`
///
/// [`JsonToVariant`](crate::JsonToVariant) and `variant_to_json` are lossy, normalizing
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
///   first, and re-parsing produces an unshredded Variant unless a new
///   [`ShreddingSpec`](crate::ShreddingSpec) is supplied.
#[derive(Clone)]
pub struct VariantToJson;

impl ScalarFnVTable for VariantToJson {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_to_json");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for VariantToJson expression"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "variant_to_json(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input_dtype = &arg_dtypes[0];
        vortex_ensure!(
            input_dtype.is_variant(),
            "VariantToJson input must be Variant, found {input_dtype}"
        );

        let storage_dtype = DType::Utf8(input_dtype.nullability());
        Ok(DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, storage_dtype)?.erased(),
        ))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let input_nullable = input.dtype().is_nullable();

        let parquet_array = parquet_variant_for_export(input, ctx)?;
        let parquet_array = parquet_array.as_::<ParquetVariant>();

        // `parquet_variant_compute::variant_to_json` only accepts unshredded
        // `STRUCT<metadata: Binary, value: Binary>` storage, so request exactly that shape and
        // unshred any typed values first.
        let target_fields: Fields = vec![
            Arc::new(Field::new("metadata", DataType::Binary, false)),
            Arc::new(Field::new("value", DataType::Binary, true)),
        ]
        .into();
        let arrow_storage = if parquet_array.typed_value_array().is_some() {
            export_unshredded_storage_to_target(&parquet_array, &target_fields, ctx)?
        } else {
            export_storage_to_target(&parquet_array, &target_fields, ctx)?
        };

        let arrow_json = parquet_variant_compute::variant_to_json(&arrow_storage)?;
        let storage = ArrayRef::from_arrow(&arrow_json, input_nullable)?;

        ExtensionArray::try_new_from_vtable(Json, EmptyMetadata, storage).map(IntoArray::into_array)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use chrono::NaiveDate;
    use parquet_variant::Variant as PqVariant;
    use parquet_variant::VariantBuilder;
    use parquet_variant_compute::VariantArrayBuilder;
    use vortex_array::Canonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::accessor::ArrayAccessor;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::arrays::Variant;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::arrays::variant::VariantArrayExt;
    use vortex_array::arrow::ArrowSession;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::session::DTypeSession;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::fns::variant_get::VariantPath;
    use vortex_array::scalar_fn::fns::variant_get::VariantPathElement;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_error::vortex_bail;
    use vortex_error::vortex_err;

    use super::*;
    use crate::ShreddingSpec;
    use crate::fns::json_to_variant;
    use crate::fns::variant_to_json;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty()
            .with::<ArraySession>()
            .with::<ArrowSession>()
            .with::<DTypeSession>()
            .with::<ScalarFnSession>();
        crate::initialize(&session);
        session
    });

    fn json_dtype(nullability: Nullability) -> VortexResult<DType> {
        Ok(DType::Extension(
            ExtDType::<Json>::try_new(EmptyMetadata, DType::Utf8(nullability))?.erased(),
        ))
    }

    fn json_strings(array: &ArrayRef) -> VortexResult<Vec<Option<String>>> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext = array.clone().execute::<ExtensionArray>(&mut ctx)?;
        let storage = ext
            .storage_array()
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?;
        Ok(storage.with_iterator(|iter| {
            iter.map(|value| value.map(|bytes| String::from_utf8_lossy(bytes).into_owned()))
                .collect()
        }))
    }

    fn unshredded_variant(
        rows: impl IntoIterator<Item = PqVariant<'static, 'static>>,
    ) -> VortexResult<ArrayRef> {
        let rows = rows.into_iter().collect::<Vec<_>>();
        let mut builder = VariantArrayBuilder::new(rows.len());
        for row in rows {
            builder.append_variant(row);
        }
        ParquetVariant::from_arrow_variant(&builder.build())
    }

    fn json_rows_to_variant(
        rows: Vec<Option<&str>>,
        shredding: ShreddingSpec,
    ) -> VortexResult<ArrayRef> {
        let input = VarBinViewArray::from_iter_nullable_str(rows).into_array();
        input
            .apply(&json_to_variant(root(), shredding))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
    }

    #[test]
    fn renders_unshredded_values() -> VortexResult<()> {
        let input = unshredded_variant([
            PqVariant::from(42i32),
            PqVariant::from("hello"),
            PqVariant::from(true),
            PqVariant::Null,
        ])?;

        let result = input
            .apply(&variant_to_json(root()))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;

        assert_eq!(result.dtype(), &json_dtype(Nullability::NonNullable)?);
        assert_eq!(
            json_strings(&result)?,
            vec![
                Some("42".to_string()),
                Some(r#""hello""#.to_string()),
                Some("true".to_string()),
                Some("null".to_string()),
            ]
        );
        Ok(())
    }

    #[test]
    fn null_rows_stay_null_and_variant_null_renders_as_json_null() -> VortexResult<()> {
        let input =
            json_rows_to_variant(vec![Some("1"), None, Some("null")], ShreddingSpec::empty())?;

        let result = input
            .apply(&variant_to_json(root()))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;

        assert_eq!(result.dtype(), &json_dtype(Nullability::Nullable)?);
        assert_eq!(
            json_strings(&result)?,
            vec![Some("1".to_string()), None, Some("null".to_string())]
        );
        Ok(())
    }

    fn typed_value_only_variant() -> VortexResult<ArrayRef> {
        let rows = [
            VariantBuilder::new().with_value(10i32).finish(),
            VariantBuilder::new().with_value(20i32).finish(),
            VariantBuilder::new().with_value(30i32).finish(),
        ];
        let metadata =
            VarBinViewArray::from_iter_bin(rows.iter().map(|(metadata, _)| metadata.as_slice()))
                .into_array();
        let typed_value = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
        Ok(
            ParquetVariant::try_new(Validity::NonNullable, metadata, None, Some(typed_value))?
                .into_array(),
        )
    }

    #[test]
    fn unshreds_typed_value_only_storage() -> VortexResult<()> {
        let result = {
            let input = typed_value_only_variant()?;
            input
                .apply(&variant_to_json(root()))?
                .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())
        }?;

        assert_eq!(
            json_strings(&result)?,
            vec![
                Some("10".to_string()),
                Some("20".to_string()),
                Some("30".to_string()),
            ]
        );
        Ok(())
    }

    #[test]
    fn unshreds_partially_shredded_storage() -> VortexResult<()> {
        let spec = ShreddingSpec::try_new([(
            VariantPath::field("a"),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )])?;
        let input = json_rows_to_variant(
            vec![
                Some(r#"{"a": 1, "b": "x"}"#),
                Some(r#"{"a": "not-a-number", "b": "y"}"#),
                Some(r#"{"b": "z"}"#),
            ],
            spec,
        )?;
        assert!(
            input.as_::<ParquetVariant>().typed_value_array().is_some(),
            "fixture must be shredded"
        );

        let result = input
            .apply(&variant_to_json(root()))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;

        assert_eq!(
            json_strings(&result)?,
            vec![
                Some(r#"{"a":1,"b":"x"}"#.to_string()),
                Some(r#"{"a":"not-a-number","b":"y"}"#.to_string()),
                Some(r#"{"b":"z"}"#.to_string()),
            ]
        );
        Ok(())
    }

    /// Shreds `rows` per `spec`, then canonicalizes so the typed values are lifted into a logical
    /// shredded child (as a file read-back would produce).
    fn canonical_shredded(rows: Vec<Option<&str>>, spec: ShreddingSpec) -> VortexResult<ArrayRef> {
        let shredded = json_rows_to_variant(rows, spec)?;
        let Canonical::Variant(canonical) =
            shredded.execute::<Canonical>(&mut SESSION.create_execution_ctx())?
        else {
            vortex_bail!("expected canonical variant");
        };
        Ok(canonical.into_array())
    }

    /// Rendering a canonicalized shredded variant must match rendering the same data unshredded:
    /// the shredding is a storage optimization and is invisible to JSON output.
    fn assert_canonical_matches_unshredded(
        rows: Vec<Option<&str>>,
        spec: ShreddingSpec,
    ) -> VortexResult<()> {
        let unshredded = json_rows_to_variant(rows.clone(), ShreddingSpec::empty())?;
        let want = json_strings(
            &unshredded
                .apply(&variant_to_json(root()))?
                .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?,
        )?;

        let canonical = canonical_shredded(rows, spec)?;
        assert!(
            canonical.as_::<Variant>().shredded().is_some(),
            "fixture must carry a canonical shredded child"
        );
        let got = json_strings(
            &canonical
                .apply(&variant_to_json(root()))?
                .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?,
        )?;

        assert_eq!(got, want);
        Ok(())
    }

    /// Regression for the canonicalization bug: object-field shredding lifted into a logical
    /// shredded child must round-trip back to the full object, including the shredded field and
    /// the non-conforming/missing-field fallbacks.
    #[test]
    fn renders_canonical_object_shredded_variant() -> VortexResult<()> {
        let spec = ShreddingSpec::try_new([(
            VariantPath::field("a"),
            DType::Primitive(PType::I64, Nullability::Nullable),
        )])?;
        let canonical = canonical_shredded(
            vec![
                Some(r#"{"a": 1, "b": "x"}"#),
                Some(r#"{"a": "not-a-number", "b": "y"}"#),
                Some(r#"{"b": "z"}"#),
            ],
            spec,
        )?;
        assert!(
            canonical.as_::<Variant>().shredded().is_some(),
            "fixture must carry a canonical shredded child"
        );

        let result = canonical
            .apply(&variant_to_json(root()))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;

        assert_eq!(
            json_strings(&result)?,
            vec![
                Some(r#"{"a":1,"b":"x"}"#.to_string()),
                Some(r#"{"a":"not-a-number","b":"y"}"#.to_string()),
                Some(r#"{"b":"z"}"#.to_string()),
            ]
        );
        Ok(())
    }

    #[test]
    fn canonical_object_shredding_matches_unshredded() -> VortexResult<()> {
        assert_canonical_matches_unshredded(
            vec![
                Some(r#"{"a": 1, "b": "x", "c": 3}"#),
                Some(r#"{"a": 2, "b": "y", "c": 4}"#),
                None,
            ],
            ShreddingSpec::try_new([
                (
                    VariantPath::field("a"),
                    DType::Primitive(PType::I64, Nullability::Nullable),
                ),
                (
                    VariantPath::field("c"),
                    DType::Primitive(PType::I64, Nullability::Nullable),
                ),
            ])?,
        )
    }

    #[test]
    fn canonical_nested_dotted_shredding_matches_unshredded() -> VortexResult<()> {
        assert_canonical_matches_unshredded(
            vec![
                Some(r#"{"a": {"b": 100}, "c": "keep"}"#),
                Some(r#"{"a": {"b": 200}, "c": "keep2"}"#),
            ],
            ShreddingSpec::try_new([(
                VariantPath::new([
                    VariantPathElement::field("a"),
                    VariantPathElement::field("b"),
                ]),
                DType::Primitive(PType::I64, Nullability::Nullable),
            )])?,
        )
    }

    #[test]
    fn variant_only_types_are_stringified_so_reparsing_loses_types() -> VortexResult<()> {
        let date =
            NaiveDate::from_ymd_opt(2026, 6, 11).ok_or_else(|| vortex_err!("invalid test date"))?;
        let input = unshredded_variant([PqVariant::from(date)])?;

        let rendered = input
            .apply(&variant_to_json(root()))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;
        let json = json_strings(&rendered)?;
        assert_eq!(json, vec![Some(r#""2026-06-11""#.to_string())]);

        // Re-parsing the rendered JSON yields a plain string variant, not a date: the type
        // information is lost, demonstrating that the conversions are not inverses.
        let reparsed = rendered
            .apply(&json_to_variant(root(), ShreddingSpec::empty()))?
            .execute::<ArrayRef>(&mut SESSION.create_execution_ctx())?;
        let mut ctx = SESSION.create_execution_ctx();
        let row0 = reparsed.execute_scalar(0, &mut ctx)?;
        let value = row0
            .as_variant()
            .value()
            .ok_or_else(|| vortex_err!("expected non-null variant"))?;
        assert_eq!(
            value.as_utf8().value().map(|value| value.to_string()),
            Some("2026-06-11".to_string())
        );
        Ok(())
    }
}
