// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::make_array;
use arrow_data::transform::MutableArrayData;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// Concatenate Arrow arrays of the same data type into a single array.
pub(crate) fn concat(arrays: &[&dyn Array]) -> VortexResult<ArrayRef> {
    let Some(first) = arrays.first() else {
        vortex_bail!("Cannot concatenate zero Arrow arrays");
    };
    vortex_ensure!(
        arrays
            .iter()
            .all(|array| array.data_type() == first.data_type()),
        "Cannot concatenate Arrow arrays with different data types"
    );

    let data = arrays.iter().map(|array| array.to_data()).collect::<Vec<_>>();
    let capacity = data.iter().map(|data| data.len()).sum();
    let mut mutable = MutableArrayData::new(data.iter().collect(), false, capacity);
    for (index, data) in data.iter().enumerate() {
        mutable.extend(index, 0, data.len());
    }
    Ok(make_array(mutable.freeze()))
}
