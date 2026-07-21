// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;

use crate::ArrayRef;
use crate::arrays::union::array::CHILDREN_OFFSET;
use crate::arrays::union::union_type_ids_dtype;
use crate::dtype::DType;

pub(super) fn validate_union_components(
    type_ids: &ArrayRef,
    variant_arrays: &[&ArrayRef],
    dtype: &DType,
    len: usize,
) -> VortexResult<()> {
    let DType::Union(variants, nullability) = dtype else {
        vortex_bail!("Expected union dtype, found {dtype}")
    };
    vortex_ensure_eq!(
        variant_arrays.len(),
        variants.len(),
        "UnionArray has {} slots but expected {}",
        CHILDREN_OFFSET + variant_arrays.len(),
        CHILDREN_OFFSET + variants.len()
    );

    let expected_union_type_ids_dtype = union_type_ids_dtype(*nullability);
    vortex_ensure_eq!(
        type_ids.dtype(),
        &expected_union_type_ids_dtype,
        "UnionArray type_ids must have dtype {expected_union_type_ids_dtype}, got {}",
        type_ids.dtype()
    );
    vortex_ensure_eq!(
        type_ids.len(),
        len,
        "UnionArray type_ids length {} does not match outer length {len}",
        type_ids.len()
    );

    for (index, (variant_dtype, child)) in variants
        .variants()
        .zip(variant_arrays.iter().copied())
        .enumerate()
    {
        vortex_ensure_eq!(
            child.len(),
            len,
            "UnionArray child {index} length {} does not match outer length {len}",
            child.len()
        );
        vortex_ensure_eq!(
            child.dtype(),
            &variant_dtype,
            "UnionArray child {index} has dtype {} but expected {variant_dtype}",
            child.dtype()
        );
    }

    Ok(())
}
