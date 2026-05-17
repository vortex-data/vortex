// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `RowSize` variadic scalar function: aggregate per-row byte sizes for N input columns.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::dict::Dict;
use vortex_array::arrays::patched::Patched;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::codec;
use crate::codec::RowWidth;
use crate::options::RowEncodeOptions;
use crate::options::SortField;
use crate::options::deserialize_row_encode_options;
use crate::options::serialize_row_encode_options;
use crate::registry;

/// Variadic scalar function that, given N input columns and per-column [`SortField`]s,
/// returns a `Struct { fixed: U32, var: U32 }` array of per-row byte sizes for the
/// row-oriented encoding produced by [`RowEncode`](super::encode::RowEncode).
///
/// The `fixed` field is always a [`ConstantArray`] holding the sum of the per-column
/// constant widths of fixed-width inputs (sentinel + value bytes). The `var` field is a
/// `ConstantArray(0)` when there are no variable-length input columns, and a
/// [`PrimitiveArray<u32>`] of per-row varlen-byte sums otherwise.
///
/// The total per-row byte size is `fixed + var`.
#[derive(Clone, Debug)]
pub struct RowSize;

/// Returns the [`FieldNames`] used by the [`RowSize`] output struct.
pub(crate) fn row_size_field_names() -> FieldNames {
    FieldNames::from([FieldName::from("fixed"), FieldName::from("var")])
}

/// Returns the output [`DType`] of [`RowSize`].
pub(crate) fn row_size_struct_dtype() -> DType {
    DType::Struct(
        StructFields::new(
            row_size_field_names(),
            vec![
                DType::Primitive(PType::U32, Nullability::NonNullable),
                DType::Primitive(PType::U32, Nullability::NonNullable),
            ],
        ),
        Nullability::NonNullable,
    )
}

impl ScalarFnVTable for RowSize {
    type Options = RowEncodeOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.row_size")
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(serialize_row_encode_options(options)))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        deserialize_row_encode_options(metadata)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Variadic { min: 1, max: None }
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        ChildName::from(Arc::from(format!("col_{}", child_idx)))
    }

    fn return_dtype(&self, _options: &Self::Options, _args: &[DType]) -> VortexResult<DType> {
        Ok(row_size_struct_dtype())
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let n_inputs = args.num_inputs();
        if n_inputs == 0 {
            vortex_bail!("RowSize requires at least one input column");
        }
        if options.fields.len() != n_inputs {
            vortex_bail!(
                "RowSize options.fields.len()={} does not match num_inputs={}",
                options.fields.len(),
                n_inputs
            );
        }
        let nrows = args.row_count();

        // Per-column accumulation lives entirely in these stack-local variables. Only at the
        // very end do we materialize the result `StructArray`.
        let mut fixed_per_row: u32 = 0;
        let mut var_lengths: Option<Vec<u32>> = None;

        for i in 0..n_inputs {
            let col = args.get(i)?;
            if col.len() != nrows {
                vortex_bail!(
                    "RowSize: column {} has length {} but expected {}",
                    i,
                    col.len(),
                    nrows
                );
            }
            match codec::row_width_for_dtype(col.dtype())? {
                RowWidth::Fixed(w) => {
                    fixed_per_row = fixed_per_row
                        .checked_add(w)
                        .vortex_expect("row width overflow")
                }
                RowWidth::Variable => {
                    let v = var_lengths.get_or_insert_with(|| vec![0u32; nrows]);
                    dispatch_size(&col, options.fields[i], v, ctx)?;
                }
            }
        }

        let fixed_array = ConstantArray::new(Scalar::from(fixed_per_row), nrows).into_array();
        let var_array = match var_lengths {
            Some(v) => PrimitiveArray::new(Buffer::<u32>::copy_from(&v), Validity::NonNullable)
                .into_array(),
            None => ConstantArray::new(Scalar::from(0u32), nrows).into_array(),
        };

        Ok(StructArray::try_new(
            row_size_field_names(),
            vec![fixed_array, var_array],
            nrows,
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Dispatch a single column's per-row size contribution, trying in-crate fast paths first,
/// then the inventory registry for downstream encodings, falling back to canonicalization.
pub fn dispatch_size(
    col: &ArrayRef,
    field: SortField,
    sizes: &mut [u32],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if let Some(view) = col.as_opt::<Constant>()
        && Constant::row_size_contribution(view, field, sizes, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some(view) = col.as_opt::<Dict>()
        && Dict::row_size_contribution(view, field, sizes, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some(view) = col.as_opt::<Patched>()
        && Patched::row_size_contribution(view, field, sizes, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some((size_fn, _)) = registry::lookup(&col.encoding_id())
        && size_fn(col, field, sizes, ctx)?.is_some()
    {
        return Ok(());
    }
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_size(&canonical, field, sizes, ctx)
}

/// Mutate-buffer kernel: add this column's per-row byte contribution into the shared
/// `sizes` slice. Return `Ok(None)` to decline and fall back to the canonical path.
pub trait RowSizeKernel: VTable {
    /// Add this column's per-row byte contribution into `sizes`.
    fn row_size_contribution(
        column: ArrayView<'_, Self>,
        field: SortField,
        sizes: &mut [u32],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>>;
}
