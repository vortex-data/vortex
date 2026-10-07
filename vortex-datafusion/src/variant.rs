// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A DataFusion `variant_get` scalar function over Arrow Parquet Variant columns.
//!
//! `variant_get(column, 'a.b', 'Utf8')` extracts the value at path `a.b` of each Variant row and
//! casts it to the named Arrow type, producing null for missing paths and failed casts. It works
//! on any Arrow `arrow.parquet.variant` column, so the same SQL runs over Parquet and Vortex.
//!
//! The function reports [`ExpressionPlacement::MoveTowardsLeafNodes`], so DataFusion's optimizer
//! moves it out of filters and aggregates towards the scan, where the Vortex integration converts
//! it into a Vortex `variant_get` expression evaluated by the scan itself.

use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_common::internal_err;
use datafusion_expr::ColumnarValue;
use datafusion_expr::ExpressionPlacement;
use datafusion_expr::ReturnFieldArgs;
use datafusion_expr::ScalarFunctionArgs;
use datafusion_expr::ScalarUDF;
use datafusion_expr::ScalarUDFImpl;
use datafusion_expr::Signature;
use datafusion_expr::Volatility;
use parquet_variant::VariantPath as ArrowVariantPath;
use parquet_variant_compute::GetOptions;
use vortex::scalar_fn::fns::variant_get::VariantPath;
use vortex::scalar_fn::fns::variant_get::VariantPathElement;

/// The `variant_get(variant, path, type)` scalar function. See the [module docs](self).
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct VariantGetFunc {
    signature: Signature,
}

impl Default for VariantGetFunc {
    fn default() -> Self {
        Self {
            signature: Signature::any(3, Volatility::Immutable),
        }
    }
}

impl VariantGetFunc {
    /// The name the function is registered under.
    pub const NAME: &'static str = "variant_get";
}

/// The `variant_get` function as a DataFusion [`ScalarUDF`], ready to register with a session.
pub fn variant_get_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(VariantGetFunc::default())
}

/// Returns the string value of a `Utf8` scalar literal.
fn utf8_literal(value: Option<&ScalarValue>, what: &str) -> DFResult<String> {
    value
        .and_then(|value| value.try_as_str().flatten())
        .map(str::to_string)
        .ok_or_else(|| exec_datafusion_err!("variant_get {what} must be a string literal"))
}

/// Parses the Arrow type named by a `variant_get` type argument, such as `Utf8` or `Int64`.
pub(crate) fn parse_target_type(name: &str) -> DFResult<DataType> {
    DataType::from_str(name)
        .map_err(|e| exec_datafusion_err!("variant_get type {name:?} is not an Arrow type: {e}"))
}

/// Splits a dotted `variant_get` path such as `commit.collection` into object field names.
fn path_fields(path: &str) -> impl Iterator<Item = &str> {
    path.split('.').filter(|field| !field.is_empty())
}

/// Converts a dotted `variant_get` path into a Vortex [`VariantPath`].
pub(crate) fn vortex_variant_path(path: &str) -> VariantPath {
    VariantPath::new(path_fields(path).map(VariantPathElement::field))
}

impl ScalarUDFImpl for VariantGetFunc {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        internal_err!("variant_get computes its return field from its arguments")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> DFResult<FieldRef> {
        let type_name = utf8_literal(args.scalar_arguments.get(2).copied().flatten(), "type")?;
        Ok(Arc::new(Field::new(
            Self::NAME,
            parse_target_type(&type_name)?,
            true,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [input, path, _type_name] = args.args.as_slice() else {
            return internal_err!("variant_get takes exactly three arguments");
        };
        let ColumnarValue::Scalar(path) = path else {
            return internal_err!("variant_get path must be a string literal");
        };
        let path = utf8_literal(Some(path), "path")?;
        let path = ArrowVariantPath::from_iter(
            path_fields(&path).map(parquet_variant::VariantPathElement::from),
        );

        let input = input.to_array(args.number_rows)?;
        let options =
            GetOptions::new_with_path(path).with_as_type(Some(Arc::clone(&args.return_field)));
        Ok(ColumnarValue::Array(parquet_variant_compute::variant_get(
            &input, options,
        )?))
    }

    fn placement(&self, args: &[ExpressionPlacement]) -> ExpressionPlacement {
        match args {
            [
                ExpressionPlacement::Column | ExpressionPlacement::MoveTowardsLeafNodes,
                ExpressionPlacement::Literal,
                ExpressionPlacement::Literal,
            ] => ExpressionPlacement::MoveTowardsLeafNodes,
            _ => ExpressionPlacement::KeepInPlace,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::ArrayRef;
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::Schema;
    use datafusion::prelude::SessionContext;
    use parquet_variant_compute::json_to_variant;

    use super::*;

    #[tokio::test]
    async fn extracts_typed_paths() -> anyhow::Result<()> {
        let json: ArrayRef = Arc::new(StringArray::from(vec![
            r#"{"kind": "commit", "commit": {"collection": "post"}, "time_us": 7}"#,
            r#"{"kind": "identity", "time_us": "late"}"#,
        ]));
        let variant = json_to_variant(&json)?;
        let schema = Arc::new(Schema::new(vec![variant.field("data")]));
        let batch = RecordBatch::try_new(schema, vec![ArrayRef::from(variant)])?;

        let ctx = SessionContext::new();
        ctx.register_udf(variant_get_udf());
        ctx.register_batch("t", batch)?;
        let result = ctx
            .sql(
                "SELECT variant_get(data, 'commit.collection', 'Utf8') AS c, \
                 variant_get(data, 'time_us', 'Int64') AS t FROM t",
            )
            .await?
            .collect()
            .await?;

        let collection = result[0].column(0).as_string::<i32>();
        assert_eq!(collection.value(0), "post");
        assert!(collection.is_null(1));
        let time = result[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| anyhow::anyhow!("time_us must be Int64"))?;
        assert_eq!(time.value(0), 7);
        assert!(time.is_null(1));
        Ok(())
    }
}
