// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A `variant_get(variant, path, type)` function for DataFusion.
//!
//! This is a stand-in for the function of the same name and signature in
//! [`datafusion-variant`](https://github.com/datafusion-contrib/datafusion-variant), which has no
//! release for the DataFusion and Arrow versions that Vortex builds against. Both evaluate with the
//! `variant_get` kernel of `parquet-variant-compute`. Replace this module with
//! `datafusion_variant::register_all` once the versions line up.

use std::sync::Arc;

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Field;
use datafusion::arrow::datatypes::FieldRef;
use datafusion::logical_expr::ColumnarValue;
use datafusion::logical_expr::ExpressionPlacement;
use datafusion::logical_expr::ReturnFieldArgs;
use datafusion::logical_expr::ScalarFunctionArgs;
use datafusion::logical_expr::ScalarUDF;
use datafusion::logical_expr::ScalarUDFImpl;
use datafusion::logical_expr::Signature;
use datafusion::logical_expr::Volatility;
use datafusion::prelude::SessionContext;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_err;
use datafusion_common::plan_err;
use parquet_variant::VariantPath;
use parquet_variant_compute::GetOptions;
use parquet_variant_compute::variant_get;

/// Registers the Variant functions with `ctx`.
pub fn register_variant_functions(ctx: &SessionContext) {
    ctx.register_udf(ScalarUDF::new_from_impl(VariantGetUdf::new()));
}

/// `variant_get(variant, path, type)`: the value at `path` of each Variant, cast to the SQL type
/// named `type`.
///
/// `path` is a dot-separated list of object fields, such as `'commit.collection'`. A missing path
/// or a value that cannot be cast yields null. Both `path` and `type` must be string literals.
#[derive(Debug, PartialEq, Eq, Hash)]
struct VariantGetUdf {
    signature: Signature,
}

impl VariantGetUdf {
    fn new() -> Self {
        Self {
            signature: Signature::any(3, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantGetUdf {
    fn name(&self) -> &str {
        "variant_get"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        plan_err!("variant_get derives its return type from its type argument")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> DFResult<FieldRef> {
        let Some(Some(type_name)) = args.scalar_arguments.get(2) else {
            return plan_err!("the type argument of variant_get must be a literal");
        };
        let data_type = sql_type(string_literal("type", type_name)?)?;
        Ok(Arc::new(Field::new(self.name(), data_type, true)))
    }

    /// Extraction is cheap next to reading a whole Variant, so DataFusion moves it into a
    /// projection above the scan, which a source that understands `variant_get` can absorb.
    fn placement(&self, args: &[ExpressionPlacement]) -> ExpressionPlacement {
        match args {
            [
                ExpressionPlacement::Column,
                ExpressionPlacement::Literal,
                ExpressionPlacement::Literal,
            ] => ExpressionPlacement::MoveTowardsLeafNodes,
            _ => ExpressionPlacement::KeepInPlace,
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let [variant, path, _type] = args.args.as_slice() else {
            return exec_err!("variant_get expects 3 arguments, got {}", args.args.len());
        };
        let ColumnarValue::Scalar(path) = path else {
            return exec_err!("the path argument of variant_get must be a literal");
        };
        let path = VariantPath::try_from(string_literal("path", path)?)?;
        let options = GetOptions::new_with_path(path).with_as_type(Some(args.return_field));

        let input: ArrayRef = variant.to_array(args.number_rows)?;
        Ok(ColumnarValue::Array(variant_get(&input, options)?))
    }
}

/// The value of the string literal `scalar`, which is the `argument` of `variant_get`.
fn string_literal<'a>(argument: &str, scalar: &'a ScalarValue) -> DFResult<&'a str> {
    match scalar {
        ScalarValue::Utf8(Some(value))
        | ScalarValue::Utf8View(Some(value))
        | ScalarValue::LargeUtf8(Some(value)) => Ok(value),
        other => exec_err!(
            "the {argument} argument of variant_get must be a non-null string literal, got {}",
            other.data_type()
        ),
    }
}

/// The Arrow type that the SQL type `name` maps to.
fn sql_type(name: &str) -> DFResult<DataType> {
    Ok(match name.to_ascii_uppercase().as_str() {
        "VARCHAR" | "TEXT" | "STRING" => DataType::Utf8View,
        "BIGINT" => DataType::Int64,
        "INT" | "INTEGER" => DataType::Int32,
        "DOUBLE" => DataType::Float64,
        "BOOLEAN" => DataType::Boolean,
        _ => return plan_err!("variant_get does not support the type {name}"),
    })
}
