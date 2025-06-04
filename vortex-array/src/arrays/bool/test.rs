use itertools::Itertools;
use vortex_error::VortexResult;

use crate::arrays::BoolArray;

impl BoolArray {
    pub fn opt_iter(&self) -> VortexResult<impl IntoIterator<Item = Option<bool>>> {
        let values = self.boolean_buffer().clone();
        let mask = self.validity_mask()?.to_boolean_buffer();
        Ok(mask
            .into_iter()
            .zip(values.iter())
            .map(|(valid, value)| if valid { Some(value) } else { None })
            .collect_vec())
    }
}
