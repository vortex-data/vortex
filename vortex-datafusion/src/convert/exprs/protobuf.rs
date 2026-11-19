// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conversion of the [protobuf functions](crate::protobuf) to the Vortex expressions that they
//! evaluate.

use arrow_schema::Schema;
use datafusion_common::exec_datafusion_err;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::expressions as df_expr;
use itertools::Itertools;
use vortex::dtype::DType;
use vortex::expr::Expression;
use vortex_protobuf::Protobuf;

use super::Conversion;
use super::DefaultExpressionConverter;
use super::Unconverted;
use super::converted_dtype;
use crate::protobuf::ProtobufFunction;

impl DefaultExpressionConverter {
    /// Converts a call of a protobuf function, or returns `None` for any other function.
    ///
    /// The call becomes the expression that the function evaluates in DataFusion, over the
    /// converted messages. Messages whose dtype the session does not know as protobuf stay in
    /// DataFusion.
    pub(super) fn convert_protobuf_function(
        &self,
        scalar_fn: &ScalarFunctionExpr,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Option<Conversion<Expression>> {
        let function = scalar_fn.fun().inner().downcast_ref::<ProtobufFunction>()?;
        Some(self.convert_protobuf_call(function, scalar_fn, schema, input_dtype))
    }

    /// Converts a call of `function`.
    fn convert_protobuf_call(
        &self,
        function: &ProtobufFunction,
        scalar_fn: &ScalarFunctionExpr,
        schema: &Schema,
        input_dtype: &DType,
    ) -> Conversion<Expression> {
        let Some((messages, literals)) = scalar_fn.args().split_first() else {
            return Err(exec_datafusion_err!("{} requires messages", scalar_fn.name()).into());
        };
        let messages = self.convert_expr(messages, schema, input_dtype)?;
        let DType::Extension(ext_dtype) = converted_dtype(&messages, input_dtype)? else {
            return Err(Unconverted::Unsupported);
        };
        let Some(metadata) = ext_dtype.metadata_opt::<Protobuf>() else {
            return Err(Unconverted::Unsupported);
        };
        let literals = literals
            .iter()
            .map(|literal| {
                literal
                    .downcast_ref::<df_expr::Literal>()
                    .map(df_expr::Literal::value)
            })
            .collect_vec();
        Ok(function.expression(metadata, &literals, messages)?)
    }
}
