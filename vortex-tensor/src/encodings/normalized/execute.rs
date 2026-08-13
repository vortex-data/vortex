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
use crate::utils::reattach_validity;

/// Reconstructs the tensor column and attaches the parent validity.
pub(super) fn denormalize(
    normalized: &ArrayRef,
    norms: &ArrayRef,
    validity: Validity,
    dtype: DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Constant norms let us scale the whole backing buffer at once, or skip the multiply entirely
    // when every norm is exactly 1.
    if let Some(constant) = norms.as_opt::<Constant>()
        && constant.scalar().value().is_some()
    {
        return denormalize_constant_norms(normalized, constant.scalar(), dtype, validity, ctx);
    }

    let row_count = normalized.len();

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

/// Scales a constant-norm column without a per-row loop.
fn denormalize_constant_norms(
    normalized: &ArrayRef,
    norm: &Scalar,
    dtype: DType,
    validity: Validity,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let norm_value = norm
        .value()
        .vortex_expect("the caller only takes this path for a non-null constant norm")
        .as_primitive()
        .as_f64()
        .vortex_expect("norms are validated to be a float column, so the scalar fits in f64");

    // A near-unit norm must still be multiplied, or `scalar_at` can disagree with bulk decoding.
    if norm_value == 1.0 {
        return reattach_validity(normalized.clone(), validity);
    }

    let normalized: ExtensionArray = normalized.clone().execute(ctx)?;
    let storage: FixedSizeListArray = normalized.storage_array().clone().execute(ctx)?;

    let scale = ConstantArray::new(norm.clone(), storage.elements().len()).into_array();
    let elements = storage.elements().clone().binary(scale, Operator::Mul)?;

    // SAFETY: Only the element values changed; the list size and row count are carried over from
    // the storage array we just executed, and the validity is the parent's.
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
