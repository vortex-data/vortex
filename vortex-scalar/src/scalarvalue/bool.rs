use vortex_error::{VortexError, VortexResult, vortex_err};

use crate::ScalarValue;

impl TryFrom<&ScalarValue> for bool {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> VortexResult<Self> {
        <Option<bool>>::try_from(value)?
            .ok_or_else(|| vortex_err!("Can't extract present value from null scalar"))
    }
}

impl TryFrom<&ScalarValue> for Option<bool> {
    type Error = VortexError;

    fn try_from(value: &ScalarValue) -> VortexResult<Self> {
        value.as_bool()
    }
}
