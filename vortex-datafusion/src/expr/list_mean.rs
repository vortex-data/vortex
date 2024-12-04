use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef};
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{Expr, ScalarUDF};

pub fn list_mean(child: Expr) -> Expr {
    ListMean::new_expr(child)
}

#[derive(Debug)]
pub struct ListMean {
    signature: Signature,
}

impl ListMean {
    pub(crate) const NAME: &'static str = "list.mean";

    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_expr(child: Expr) -> Expr {
        Expr::ScalarFunction(ScalarFunction::new_udf(
            Arc::new(ScalarUDF::new_from_impl(ListMean::default())),
            vec![child],
        ))
    }
}

impl Default for ListMean {
    fn default() -> Self {
        Self {
            signature: Signature::new(
                TypeSignature::Coercible(vec![DataType::List(FieldRef::new(
                    Field::new_list_field(DataType::Float64, true),
                ))]),
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for ListMean {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        Self::NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_batch(
        &self,
        _args: &[ColumnarValue],
        _number_rows: usize,
    ) -> DFResult<ColumnarValue> {
        // This is optimized away (pushed down into vortex)
        todo!("FAILED TO PUSHDOWN");
    }
}
