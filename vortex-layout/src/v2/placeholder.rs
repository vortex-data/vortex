// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Helpers for row-shaped placeholder arrays.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;

/// Return a cheap array with the requested dtype and length.
///
/// This is only valid for rows known to be unobservable, for example
/// a row-preserving projection range that `RowDemand` has already
/// proved cannot contribute to the final filtered output.
pub(crate) fn default_array(dtype: &DType, len: usize) -> ArrayRef {
    match dtype {
        DType::Null => NullArray::new(len).into_array(),
        _ => ConstantArray::new(Scalar::default_value(dtype), len).into_array(),
    }
}
