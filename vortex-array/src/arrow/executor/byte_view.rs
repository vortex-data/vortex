// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::types::ByteViewType;
use vortex_compute::arrow::IntoArrow;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::VectorExecutor;
use crate::builtins::ArrayBuiltins;

pub(super) fn to_arrow_byte_view<T: ByteViewType>(
    array: ArrayRef,
    session: &VortexSession,
) -> VortexResult<ArrowArrayRef> {
    // First we cast the array into the desired ByteView type.
    // We do this in case the vortex array is Utf8, and we want Binary or vice versa. By casting
    // first, we may push this down through the Vortex array tree. We choose nullable to be most
    // flexible since there's no prescribed nullability in Arrow types.
    let array = array.cast(DType::from_arrow((&T::DATA_TYPE, Nullability::Nullable)))?;

    // Perform a naive conversion via our VarBinView vector representation
    array.execute_vector(session)?.into_arrow()
}
