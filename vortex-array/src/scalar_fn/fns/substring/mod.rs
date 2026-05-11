// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;

use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::LEGACY_SESSION;
use crate::arrow::ArrowSessionExt;
use crate::arrow::from_arrow_array_with_len;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::and;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;

/// Parses "start" and optional "length" arguments of "substr" call into a
/// 0-based byte offset and an optional byte count.
/// "start" must be a non-null constant integer >= 1
/// "length" must be a non-null constant non-negative integer.
pub(crate) fn parse_byte_range(
    start: &ArrayRef,
    length: Option<&ArrayRef>,
) -> VortexResult<(usize, Option<usize>)> {
    let start: i64 = start
        .as_constant()
        .ok_or_else(|| vortex_err!("Substring: start must be a constant"))?
        .as_primitive_opt()
        .ok_or_else(|| vortex_err!("Substring: start must be a primitive integer"))?
        .as_::<i64>()
        .ok_or_else(|| vortex_err!("Substring: start must be non-null"))?;
    vortex_ensure!(start >= 1, "Substring: start must be >= 1, got {start}");
    let byte_start =
        usize::try_from(start - 1).map_err(|_| vortex_err!("Substring: start overflows"))?;

    let byte_length = length
        .map(|len| -> VortexResult<usize> {
            let length: u64 = len
                .as_constant()
                .ok_or_else(|| vortex_err!("Substring: length must be a constant"))?
                .as_primitive_opt()
                .ok_or_else(|| vortex_err!("Substring: length must be a primitive integer"))?
                .as_::<u64>()
                .ok_or_else(|| vortex_err!("Substring: length must be non-null"))?;
            usize::try_from(length).map_err(|_| vortex_err!("Substring: length overflows"))
        })
        .transpose()?;

    Ok((byte_start, byte_length))
}

/// SQL SUBSTRING / SUBSTR expression.
#[derive(Clone)]
pub struct Substring;

impl ScalarFnVTable for Substring {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new("vortex.substring")
    }

    fn serialize(&self, _instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
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
        Arity::Variadic {
            min: 2,
            max: Some(3),
        }
    }

    fn child_name(&self, _instance: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("string"),
            1 => ChildName::from("start"),
            2 => ChildName::from("length"),
            _ => unreachable!("Invalid child index {child_idx} for Substring expression"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "substr(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        if expr.children().len() > 2 {
            write!(f, ", ")?;
            expr.child(2).fmt_sql(f)?;
        }
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        let input = &arg_dtypes[0];
        vortex_ensure!(
            input.is_utf8(),
            "Substring: expected UTF8 input, got {input}"
        );
        Ok(DType::Utf8(input.is_nullable().into()))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let string_arr = args.get(0)?;
        let start_arr = args.get(1)?;
        let length_arr = (args.num_inputs() > 2).then(|| args.get(2)).transpose()?;
        let (byte_start, byte_length) = parse_byte_range(&start_arr, length_arr.as_ref())?;
        let len = args.row_count();

        let nullable = string_arr.dtype().is_nullable();
        // arrow_string::substring does not support Utf8View; force Utf8.
        let field = Field::new("", DataType::Utf8, nullable);
        let arrow_array = LEGACY_SESSION
            .arrow()
            .execute_arrow(string_arr, Some(&field), ctx)?;
        let result = arrow_string::substring::substring(
            arrow_array.as_ref(),
            byte_start as i64,
            byte_length.map(|l| l as u64),
        )?;
        from_arrow_array_with_len(result.as_ref(), len, nullable)
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let string_validity = expression.child(0).validity()?;
        let start_validity = expression.child(1).validity()?;
        let combined = and(string_validity, start_validity);
        if expression.children().len() > 2 {
            let length_validity = expression.child(2).validity()?;
            Ok(Some(and(combined, length_validity)))
        } else {
            Ok(Some(combined))
        }
    }

    fn is_null_sensitive(&self, _instance: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::substr;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_display() {
        let expr = substr(root(), lit(1i64), None);
        assert_eq!(expr.to_string(), "substr($, 1i64)");

        let expr = substr(root(), lit(1i64), Some(lit(3i64)));
        assert_eq!(expr.to_string(), "substr($, 1i64, 3i64)");
    }

    #[test]
    fn test_start() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), None))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str(["ello", "orld"]));
        Ok(())
    }

    #[test]
    fn test_start_length() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter_str(["hello", "world"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), Some(lit(3i64))))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str(["ell", "orl"]));
        Ok(())
    }

    #[test]
    fn test_outlined_stays_outlined() -> VortexResult<()> {
        // "this string is outlined" has 23 bytes; substr(2, 20) -> 20 bytes, still outlined
        let arr =
            VarBinViewArray::from_iter_str(["this string is outlined", "another long string here"])
                .into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), Some(lit(20i64))))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            result,
            VarBinViewArray::from_iter_str(["his string is outlin", "nother long string h"])
        );
        Ok(())
    }

    #[test]
    fn test_outlined_becomes_inlined() -> VortexResult<()> {
        // "this string is outlined" -> substr(1, 5) -> "this " (5 bytes, inlined)
        let arr = VarBinViewArray::from_iter_str(["this string is outlined"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(1i64), Some(lit(5i64))))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str(["this "]));
        Ok(())
    }

    #[test]
    fn test_null_values() -> VortexResult<()> {
        let arr = VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("world")])
            .into_array();
        let result = arr
            .apply(&substr(root(), lit(2i64), Some(lit(3i64))))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(
            result,
            VarBinViewArray::from_iter_nullable_str([Some("ell"), None, Some("orl")])
        );
        Ok(())
    }

    #[test]
    fn test_start_beyond_length() -> VortexResult<()> {
        // start > string length -> empty string
        let arr = VarBinViewArray::from_iter_str(["hi"]).into_array();
        let result = arr
            .apply(&substr(root(), lit(10i64), None))?
            .execute::<VarBinViewArray>(&mut SESSION.create_execution_ctx())?;
        assert_arrays_eq!(result, VarBinViewArray::from_iter_str([""]));
        Ok(())
    }
}
