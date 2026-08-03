// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::matcher::AnyTensor;
use crate::utils::extract_flat_elements;
use crate::utils::unit_norm_tolerance;

/// Reconstructs the original tensor column by scaling each normalized row by its stored norm.
///
/// `dtype` is the parent [`NormalizedArray`]'s dtype, so the reconstructed column carries the
/// unioned nullability of both children.
///
/// [`NormalizedArray`]: crate::encodings::normalized::NormalizedArray
pub(super) fn denormalize(
    normalized: &ArrayRef,
    norms: &ArrayRef,
    row_count: usize,
    dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = normalized.validity()?.and(norms.validity()?)?;

    // Constant norms let us scale the whole backing buffer at once, or skip the multiply entirely
    // when every norm is already 1. The nullability guard keeps us on the general path when the
    // constant is a non-null value inside a nullable column, since the fast path cannot widen the
    // normalized child's dtype to match the parent's.
    if let Some(constant) = norms.as_opt::<Constant>()
        && constant.scalar().value().is_some()
        && normalized.dtype() == &dtype
    {
        return denormalize_constant_norms(normalized, constant.scalar(), dtype, validity, ctx);
    }

    let normalized: ExtensionArray = normalized.clone().execute(ctx)?;
    let norms: PrimitiveArray = norms.clone().execute(ctx)?;

    let tensor_flat_size = tensor_flat_size(normalized.dtype());
    let flat = extract_flat_elements(normalized.storage_array(), tensor_flat_size, ctx)?;

    // TODO(connor): Do we want a "broadcast" expression for the List types, or is this fine?
    match_each_float_ptype!(flat.ptype(), |T| {
        let norms = norms.as_slice::<T>();

        let elements: Buffer<T> = (0..row_count)
            .flat_map(|i| {
                let norm = norms[i];
                flat.row::<T>(i).iter().map(move |&x| x * norm)
            })
            .collect();

        build_tensor_array(dtype, tensor_flat_size, row_count, validity, elements)
    })
}

/// Scales every row by the same stored norm.
///
/// Two things make this cheaper than the general path: a norm of `1.0` is the identity, so the
/// normalized child is already the answer; and otherwise the scale factor applies uniformly to the
/// flat backing buffer, so it becomes one lazy multiply over the elements array instead of a
/// per-row loop.
fn denormalize_constant_norms(
    normalized: &ArrayRef,
    norm: &Scalar,
    dtype: DType,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let tensor_flat_size = tensor_flat_size(normalized.dtype());
    let error = norm
        .value()
        .vortex_expect("the caller only takes this path for a non-null constant norm")
        .as_primitive()
        .as_f64()
        .vortex_expect("norms are validated to be a float column, so the scalar fits in f64")
        - 1.0f64;

    if error.abs() < unit_norm_tolerance(norm.dtype().as_ptype(), tensor_flat_size) {
        return Ok(normalized.clone());
    }

    let normalized: ExtensionArray = normalized.clone().execute(ctx)?;
    let storage: FixedSizeListArray = normalized.storage_array().clone().execute(ctx)?;

    let scale = ConstantArray::new(norm.clone(), storage.elements().len()).into_array();
    let elements = storage.elements().clone().binary(scale, Operator::Mul)?;

    // SAFETY: Only the element values changed; the list size, validity, and row count are carried
    // over from the storage array we just executed.
    let storage = unsafe {
        FixedSizeListArray::new_unchecked(elements, storage.list_size(), validity, storage.len())
    };

    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage.into_array()).into_array())
}

/// Rebuilds a tensor-like extension array from flat primitive elements.
fn build_tensor_array<T: NativePType>(
    dtype: DType,
    tensor_flat_size: usize,
    row_count: usize,
    validity: Validity,
    elements: Buffer<T>,
) -> VortexResult<ArrayRef> {
    let list_size =
        u32::try_from(tensor_flat_size).vortex_expect("tensor flat size must fit into `u32`");

    // SAFETY: Tensor elements are always non-nullable, so the validity carries no length.
    let elements = unsafe { PrimitiveArray::new_unchecked(elements, Validity::NonNullable) };

    let storage =
        FixedSizeListArray::try_new(elements.into_array(), list_size, validity, row_count)?;

    Ok(ExtensionArray::new(dtype.as_extension().clone(), storage.into_array()).into_array())
}

/// Returns the flattened element count of each row of a tensor-like extension dtype.
fn tensor_flat_size(dtype: &DType) -> usize {
    dtype
        .as_extension()
        .metadata_opt::<AnyTensor>()
        .vortex_expect("the normalized child is validated to be an `AnyTensor` on construction")
        .list_size() as usize
}
