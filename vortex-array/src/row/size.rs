// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `RowSize` variadic scalar function: aggregate per-row byte sizes for N input columns.

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Constant;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::Dict;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::row::codec;
use crate::row::options::RowEncodeOptions;
use crate::row::options::SortField;
use crate::row::options::deserialize_row_encode_options;
use crate::row::options::serialize_row_encode_options;
use crate::row::registry;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

/// Variadic scalar function that, given N input columns and per-column [`SortField`]s,
/// returns a `U32` array of per-row byte sizes for the row-oriented encoding produced by
/// [`RowEncode`](super::encode::RowEncode).
#[derive(Clone, Debug)]
pub struct RowSize;

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
        Ok(DType::Primitive(PType::U32, Nullability::NonNullable))
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
        let mut sizes = vec![0u32; nrows];
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
            dispatch_size(&col, options.fields[i], &mut sizes, ctx)?;
        }
        Ok(
            PrimitiveArray::new(Buffer::<u32>::copy_from(&sizes), Validity::NonNullable)
                .into_array(),
        )
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
