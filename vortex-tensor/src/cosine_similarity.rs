// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cosine similarity expression for [`FixedShapeTensor`] arrays.

use std::fmt::Formatter;

use num_traits::Float;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::ToCanonical;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ConstantVTable;
use vortex::array::arrays::ExtensionVTable;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_float_ptype;
use vortex::dtype::DType;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::scalar_fn::Arity;
use vortex::scalar_fn::ChildName;
use vortex::scalar_fn::EmptyOptions;
use vortex::scalar_fn::ExecutionArgs;
use vortex::scalar_fn::ScalarFnId;
use vortex::scalar_fn::ScalarFnVTable;

/// Cosine similarity between two [`FixedShapeTensor`] columns.
///
/// Computes `dot(a, b) / (||a|| * ||b||)` over the flat backing buffer of each tensor. The
/// shape and permutation do not affect the result because cosine similarity only depends on the
/// element values, not their logical arrangement.
///
/// Both inputs must be [`FixedShapeTensor`] extension arrays with the same dtype and a float
/// element type (`f32` or `f64`). The output is a primitive column of the same float type.
///
/// [`FixedShapeTensor`]: crate::FixedShapeTensor
#[derive(Clone)]
pub struct CosineSimilarity;

impl ScalarFnVTable for CosineSimilarity {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::new_ref("vortex.cosine_similarity")
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("lhs"),
            1 => ChildName::from("rhs"),
            _ => unreachable!("CosineSimilarity must have exactly two children"),
        }
    }

    fn fmt_sql(
        &self,
        _options: &Self::Options,
        expr: &Expression,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "cosine_similarity(")?;
        expr.child(0).fmt_sql(f)?;
        write!(f, ", ")?;
        expr.child(1).fmt_sql(f)?;
        write!(f, ")")
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        debug_assert_eq!(arg_dtypes.len(), 2);

        let lhs = &arg_dtypes[0];
        let rhs = &arg_dtypes[1];

        // Both must have the same dtype (ignoring top-level nullability).
        vortex_ensure!(
            lhs.eq_ignore_nullability(rhs),
            "cosine_similarity requires both inputs to have the same dtype, got {lhs} and {rhs}"
        );

        // We don't need to look at rhs anymore since we know lhs and rhs are equal.

        // Both inputs must be extension types.
        let lhs_ext = lhs.as_extension_opt().ok_or_else(|| {
            vortex_err!("cosine_similarity lhs must be an extension type, got {lhs}")
        })?;

        // Extract the element dtype from the storage FixedSizeList.
        let element_dtype = lhs_ext
            .storage_dtype()
            .as_fixed_size_list_element_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "cosine_similarity storage dtype must be a FixedSizeList, got {}",
                    lhs_ext.storage_dtype()
                )
            })?;

        // Element dtype must be a non-nullable float primitive.
        vortex_ensure!(
            element_dtype.is_float(),
            "cosine_similarity element dtype must be a float primitive, got {element_dtype}"
        );
        vortex_ensure!(
            !element_dtype.is_nullable(),
            "cosine_similarity element dtype must be non-nullable"
        );

        let ptype = element_dtype.as_ptype();
        let nullability = Nullability::from(lhs.is_nullable() || rhs.is_nullable());
        Ok(DType::Primitive(ptype, nullability))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let lhs = args.get(0)?;
        let rhs = args.get(1)?;
        let row_count = args.row_count();

        // Get list size from the dtype. Both sides should have the same dtype.
        let ext = lhs.dtype().as_extension_opt().ok_or_else(|| {
            vortex_err!(
                "cosine_similarity input must be an extension type, got {}",
                lhs.dtype()
            )
        })?;
        let DType::FixedSizeList(_, list_size, _) = ext.storage_dtype() else {
            vortex_bail!("expected FixedSizeList storage dtype");
        };
        let list_size = *list_size as usize;

        // Extract the storage array from each extension input. We pass the storage (FSL) rather
        // than the extension array to avoid canonicalizing the extension wrapper.
        let lhs_storage = extension_storage(&lhs)?;
        let rhs_storage = extension_storage(&rhs)?;

        // Extract the flat primitive elements from each tensor column. When an input is a
        // `ConstantArray` (e.g., a literal query vector), we materialize only a single row
        // instead of expanding it to the full row count.
        let (lhs_elems, lhs_stride) = extract_flat_elements(&lhs_storage, list_size);
        let (rhs_elems, rhs_stride) = extract_flat_elements(&rhs_storage, list_size);

        match_each_float_ptype!(lhs_elems.ptype(), |T| {
            let lhs_slice = lhs_elems.as_slice::<T>();
            let rhs_slice = rhs_elems.as_slice::<T>();

            let result: PrimitiveArray = (0..row_count)
                .map(|i| {
                    let a = &lhs_slice[i * lhs_stride..i * lhs_stride + list_size];
                    let b = &rhs_slice[i * rhs_stride..i * rhs_stride + list_size];
                    cosine_similarity_row(a, b)
                })
                .collect();

            Ok(result.into_array())
        })
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        // The result is null if either input tensor is null.
        let lhs_validity = expression.child(0).validity()?;
        let rhs_validity = expression.child(1).validity()?;

        Ok(Some(vortex::expr::and(lhs_validity, rhs_validity)))
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        // TODO(connor): Is this correct since we need to canonicalize?
        false
    }
}

/// Extracts the storage array from an extension array without canonicalizing.
fn extension_storage(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let ext = array
        .as_opt::<ExtensionVTable>()
        .ok_or_else(|| vortex_err!("cosine_similarity input must be an extension array"))?;
    Ok(ext.storage().clone())
}

/// Extracts the flat primitive elements from a tensor storage array (FixedSizeList).
///
/// When the input is a [`ConstantArray`] (e.g., a literal query vector), only a single row is
/// materialized to avoid expanding it to the full column length. Returns `(elements, stride)`
/// where `stride` is `list_size` for a full array and `0` for a constant.
fn extract_flat_elements(storage: &ArrayRef, list_size: usize) -> (PrimitiveArray, usize) {
    if let Some(constant) = storage.as_opt::<ConstantVTable>() {
        // Rewrite the array as a length 1 array so when we canonicalize, we do not duplicate a
        // huge amount of data.
        let single = ConstantArray::new(constant.scalar().clone(), 1).into_array();
        let elems = single.to_fixed_size_list().elements().to_primitive();
        (elems, 0)
    } else {
        // Otherwise we have to fully expand all of the data.
        let elems = storage.to_fixed_size_list().elements().to_primitive();
        (elems, list_size)
    }
}

// TODO(connor): We should try to use a more performant library instead of doing this ourselves.
/// Computes cosine similarity between two equal-length float slices.
///
/// Returns `dot(a, b) / (||a|| * ||b||)`. When either vector has zero norm, this naturally
/// produces `NaN` via `0.0 / 0.0`, matching standard floating-point semantics.
fn cosine_similarity_row<T: Float + NativePType>(a: &[T], b: &[T]) -> T {
    let mut dot = T::zero();
    let mut norm_a = T::zero();
    let mut norm_b = T::zero();
    for i in 0..a.len() {
        dot = dot + a[i] * b[i];
        norm_a = norm_a + a[i] * a[i];
        norm_b = norm_b + b[i] * b[i];
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
