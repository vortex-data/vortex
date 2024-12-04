use arrow_schema::{DataType, Field, FieldRef};
use datafusion::common::exec_err;
use datafusion::error::Result as DFResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarUDFImpl, Signature, TypeSignature, Volatility,
};

#[derive(Debug)]
pub struct ListMean {
    signature: Signature,
}

impl ListMean {
    pub(crate) const NAME: &'static str = "list.mean";

    pub fn new() -> Self {
        Self::default()
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

        let _list_arr = match list {
            ColumnarValue::Array(arr) => arr,
            // TODO(marko): Support scalar.
            _ => exec_err!("first arg must be an array")?,
        };

        todo!()
    }
}
