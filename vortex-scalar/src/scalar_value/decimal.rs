use crate::scalar_value::InnerScalarValue;
use crate::{DecimalValue, ScalarValue};

impl From<DecimalValue> for ScalarValue {
    fn from(value: DecimalValue) -> Self {
        Self(InnerScalarValue::Decimal(value))
    }
}
