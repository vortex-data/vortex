// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DataFusion functions over Variant (`arrow.parquet.variant`) columns.
//!
//! [`VariantGetUdf`] registers a `variant_get(variant, path [, type])` scalar function. It runs
//! on any Arrow Variant column using the `parquet-variant-compute` kernels, so it works the same
//! for Parquet and Vortex tables. Over Vortex tables, the [`DefaultExpressionConvertor`] pushes
//! calls into the scan as Vortex [`variant_get`] expressions, where shredded paths can be served
//! straight from their typed columns.
//!
//! [`DefaultExpressionConvertor`]: crate::convert::DefaultExpressionConvertor
//! [`variant_get`]: vortex::expr::variant_get

use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::extension::EXTENSION_TYPE_NAME_KEY;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_common::internal_err;
use datafusion_common::plan_datafusion_err;
use datafusion_common::plan_err;
use datafusion_expr::ColumnarValue;
use datafusion_expr::ExpressionPlacement;
use datafusion_expr::ReturnFieldArgs;
use datafusion_expr::ScalarFunctionArgs;
use datafusion_expr::ScalarUDF;
use datafusion_expr::ScalarUDFImpl;
use datafusion_expr::Signature;
use datafusion_expr::TypeSignature;
use datafusion_expr::Volatility;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::expressions::Literal;
use parquet_variant::VariantPath as ArrowVariantPath;
use parquet_variant::VariantPathElement as ArrowVariantPathElement;
use parquet_variant_compute::GetOptions;
use vortex::scalar_fn::fns::variant_get::VariantPath;
use vortex::scalar_fn::fns::variant_get::VariantPathElement;

/// The Arrow canonical extension name of Variant columns.
const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";

/// `variant_get(variant, path [, type])`: extracts `path` from each Variant value.
///
/// - `path` is a string literal using dotted field names and bracketed list indexes, e.g.
///   `'commit.collection'` or `'items[0].id'`. A leading `$` or `$.` is accepted.
/// - `type`, when given, is a string literal naming an Arrow [`DataType`] (e.g. `'Utf8'`,
///   `'Int64'`). Values are cast to that type, producing nulls for missing paths and failed
///   casts. Without `type`, the result is a Variant column.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct VariantGetUdf {
    signature: Signature,
}

impl Default for VariantGetUdf {
    fn default() -> Self {
        Self {
            signature: Signature::one_of(
                vec![TypeSignature::Any(2), TypeSignature::Any(3)],
                Volatility::Immutable,
            ),
        }
    }
}

impl VariantGetUdf {
    /// The SQL name of the function.
    pub const NAME: &'static str = "variant_get";

    /// Returns the function wrapped as a [`ScalarUDF`], ready to register with a session.
    pub fn udf() -> ScalarUDF {
        ScalarUDF::new_from_impl(Self::default())
    }
}

impl ScalarUDFImpl for VariantGetUdf {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        internal_err!("variant_get computes its return type from its literal arguments")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> DFResult<FieldRef> {
        let return_type = match args.scalar_arguments.get(2) {
            None => None,
            Some(Some(scalar)) => Some(parse_data_type(scalar)?),
            Some(None) => {
                return plan_err!("variant_get type argument must be a string literal");
            }
        };
        if !matches!(args.scalar_arguments.get(1), Some(Some(_))) {
            return plan_err!("variant_get path argument must be a string literal");
        }
        Ok(Arc::new(match return_type {
            Some(data_type) => Field::new(Self::NAME, data_type, true),
            None => variant_field(Self::NAME),
        }))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let path = match args.args.get(1) {
            Some(ColumnarValue::Scalar(scalar)) => literal_str(scalar)?.to_string(),
            _ => return internal_err!("variant_get path argument must be a string literal"),
        };
        let input = args.args[0].to_array(args.number_rows)?;
        let as_type = (args.args.len() == 3).then(|| Arc::clone(&args.return_field));
        let options =
            GetOptions::new_with_path(parse_arrow_path(&path)?).with_as_type(as_type.clone());
        let output = parquet_variant_compute::variant_get(&input, options)?;
        Ok(ColumnarValue::Array(output))
    }

    fn placement(&self, args: &[ExpressionPlacement]) -> ExpressionPlacement {
        // Like `get_field`, extracting a path is cheap relative to moving the whole Variant
        // column, and lets sources that understand Variant read only the extracted path.
        match args.split_first() {
            Some((input, rest))
                if matches!(
                    input,
                    ExpressionPlacement::Column | ExpressionPlacement::MoveTowardsLeafNodes
                ) && rest
                    .iter()
                    .all(|p| matches!(p, ExpressionPlacement::Literal)) =>
            {
                ExpressionPlacement::MoveTowardsLeafNodes
            }
            _ => ExpressionPlacement::KeepInPlace,
        }
    }
}

/// An Arrow field for unshredded Variant values.
fn variant_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::Struct(
            vec![
                Field::new("metadata", DataType::BinaryView, false),
                Field::new("value", DataType::BinaryView, true),
            ]
            .into(),
        ),
        true,
    )
    .with_metadata(
        [(
            EXTENSION_TYPE_NAME_KEY.to_string(),
            VARIANT_EXTENSION_NAME.to_string(),
        )]
        .into(),
    )
}

fn literal_str(scalar: &ScalarValue) -> DFResult<&str> {
    match scalar {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::Utf8View(Some(s))
        | ScalarValue::LargeUtf8(Some(s)) => Ok(s.as_str()),
        other => plan_err!("variant_get expected a string literal, found {other}"),
    }
}

/// Parses a type name such as `Utf8`, `Int64`, or `Timestamp(Microsecond, None)`.
fn parse_data_type(scalar: &ScalarValue) -> DFResult<DataType> {
    let name = literal_str(scalar)?;
    DataType::from_str(name)
        .map_err(|e| plan_datafusion_err!("variant_get: invalid type '{name}': {e}"))
}

fn strip_root(path: &str) -> &str {
    let path = path.strip_prefix('$').unwrap_or(path);
    path.strip_prefix('.').unwrap_or(path)
}

fn parse_arrow_path(path: &str) -> DFResult<ArrowVariantPath<'_>> {
    ArrowVariantPath::try_from(strip_root(path))
        .map_err(|e| exec_datafusion_err!("variant_get: invalid path '{path}': {e}"))
}

/// Parses a `variant_get` path string into a Vortex [`VariantPath`].
pub fn parse_variant_path(path: &str) -> DFResult<VariantPath> {
    parse_arrow_path(path)?
        .path()
        .iter()
        .map(|element| match element {
            ArrowVariantPathElement::Field { name } => Ok(VariantPathElement::field(name.as_ref())),
            ArrowVariantPathElement::Index { index } => u64::try_from(*index)
                .map(VariantPathElement::index)
                .map_err(|e| exec_datafusion_err!("variant_get: invalid index {index}: {e}")),
        })
        .collect()
}

/// Extracts the path and optional output type of a `variant_get` call's literal arguments.
pub(crate) fn variant_get_literal_args(args: &[Arc<dyn PhysicalExpr>]) -> DFResult<String> {
    let path = args
        .get(1)
        .and_then(|arg| arg.downcast_ref::<Literal>())
        .ok_or_else(|| exec_datafusion_err!("variant_get path argument must be a literal"))?;
    Ok(literal_str(path.value())?.to_string())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::scalar_fn::fns::variant_get::VariantPath;
    use vortex::scalar_fn::fns::variant_get::VariantPathElement;

    use super::parse_variant_path;

    #[rstest]
    #[case("a", VariantPath::field("a"))]
    #[case("$.a.b", VariantPath::new([VariantPathElement::field("a"), VariantPathElement::field("b")]))]
    #[case("a[2].c", VariantPath::new([
        VariantPathElement::field("a"),
        VariantPathElement::index(2),
        VariantPathElement::field("c"),
    ]))]
    fn parses_paths(#[case] path: &str, #[case] expected: VariantPath) {
        assert_eq!(parse_variant_path(path).unwrap(), expected);
    }
}
