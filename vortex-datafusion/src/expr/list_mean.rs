use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::PrimitiveArray;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::exec_err;
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

    fn invoke_batch(&self, args: &[ColumnarValue], _number_rows: usize) -> DFResult<ColumnarValue> {
        let [list] = args else {
            return exec_err!("The number of arguments provided must be exactly 1");
        };

        let list_arr = match list {
            ColumnarValue::Array(arr) => arr.as_list::<i32>(),
            // TODO(marko): Support scalar.
            _ => exec_err!("first arg must be an array")?,
        };

        let offsets = list_arr.offsets();
        // TODO(marko): numeric is valid.
        let elements = list_arr.values().as_primitive::<Float64Type>();

        let mut sum = 0f64;
        let mut count = 0;
        let mut current_end = offsets[1];
        let mut current_end_idx = 1;
        let mut means = PrimitiveArray::<Float64Type>::builder(offsets.len() - 1);
        for i in 0..elements.len() {
            if i == current_end as usize {
                let mean = sum / count as f64;
                means.append_value(mean);
                sum = 0.0; // elements[i];
                count = 1;
                current_end_idx += 1;
                current_end = offsets[current_end_idx];
            } else {
                sum += 0.0;
                count += 1;
            }
        }

        let mean_array = means.finish();
        Ok(ColumnarValue::Array(Arc::from(mean_array)))
    }
}
