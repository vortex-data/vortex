// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(
    clippy::cast_possible_truncation,
    reason = "row encoding indexes into u32-sized buffers; lengths are validated to fit in u32"
)]

//! `RowEncode` variadic scalar function: encode N input columns into a single `ListView<u8>`.
//!
//! The output's `(elements, offsets, sizes)` triple is built up in a single left-to-right
//! pass over the input columns. The `sizes` array doubles as the per-row write cursor, so
//! when the last column finishes encoding, the accumulator is the final array - no separate
//! conversion step is needed.

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
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
use crate::arrays::ListViewArray;
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
use crate::row::size::dispatch_size;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::validity::Validity;

/// Variadic scalar function that encodes N input columns into a single `List<u8>`
/// [`ListViewArray`] where row `i` contains the row-encoded bytes for column values
/// `cols[0][i], cols[1][i], ...` concatenated left-to-right.
#[derive(Clone, Debug)]
pub struct RowEncode;

impl ScalarFnVTable for RowEncode {
    type Options = RowEncodeOptions;

    fn id(&self) -> ScalarFnId {
        ScalarFnId::from("vortex.row_encode")
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
        Ok(DType::List(
            Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
            Nullability::NonNullable,
        ))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_row_encode(options, args, ctx)
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

fn execute_row_encode(
    options: &RowEncodeOptions,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let n_inputs = args.num_inputs();
    if n_inputs == 0 {
        vortex_bail!("RowEncode requires at least one input column");
    }
    if options.fields.len() != n_inputs {
        vortex_bail!(
            "RowEncode options.fields.len()={} does not match num_inputs={}",
            options.fields.len(),
            n_inputs
        );
    }
    let nrows = args.row_count();

    // Collect inputs once; we walk them twice (size pass + encode pass).
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let col = args.get(i)?;
        if col.len() != nrows {
            vortex_bail!(
                "RowEncode: column {} has length {} but expected {}",
                i,
                col.len(),
                nrows
            );
        }
        columns.push(col);
    }

    // Size pass: per-row sizes.
    let mut sizes = vec![0u32; nrows];
    for (i, col) in columns.iter().enumerate() {
        dispatch_size(col, options.fields[i], &mut sizes, ctx)?;
    }

    // Exclusive prefix sum to get per-row offsets, with overflow check.
    let mut offsets = vec![0u32; nrows];
    let mut total: u64 = 0;
    for i in 0..nrows {
        if total > u32::MAX as u64 {
            vortex_bail!("row-encoded output size {} bytes exceeds u32::MAX", total);
        }
        offsets[i] = total as u32;
        total += u64::from(sizes[i]);
    }
    if total > u32::MAX as u64 {
        vortex_bail!("row-encoded output size {} bytes exceeds u32::MAX", total);
    }
    let total_len = total as usize;

    // Allocate the elements buffer (zero-initialized) and the cursor / lengths array.
    let mut out_buf: BufferMut<u8> = BufferMut::with_capacity(total_len);
    out_buf.push_n(0u8, total_len);

    // `lengths[i]` is both the per-row write cursor during encoding AND the final
    // ListView `sizes` slot. When all columns have been encoded, `lengths == sizes`.
    let mut lengths = vec![0u32; nrows];

    for (i, col) in columns.iter().enumerate() {
        dispatch_encode(
            col,
            options.fields[i],
            &offsets,
            &mut lengths,
            &mut out_buf,
            ctx,
        )?;
    }

    debug_assert_eq!(lengths, sizes);

    let elements = PrimitiveArray::new(out_buf.freeze(), Validity::NonNullable).into_array();
    let offsets_arr =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&offsets), Validity::NonNullable).into_array();
    let sizes_arr =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&lengths), Validity::NonNullable).into_array();
    Ok(
        ListViewArray::try_new(elements, offsets_arr, sizes_arr, Validity::NonNullable)?
            .into_array(),
    )
}

/// Dispatch a single column's encoding into the shared `out` buffer.
pub fn dispatch_encode(
    col: &ArrayRef,
    field: SortField,
    offsets: &[u32],
    cursors: &mut [u32],
    out: &mut [u8],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    if let Some(view) = col.as_opt::<Constant>()
        && Constant::row_encode_into(view, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some(view) = col.as_opt::<Dict>()
        && Dict::row_encode_into(view, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    if let Some((_, encode_fn)) = registry::lookup(&col.encoding_id())
        && encode_fn(col, field, offsets, cursors, out, ctx)?.is_some()
    {
        return Ok(());
    }
    let canonical = col.clone().execute::<Canonical>(ctx)?;
    codec::field_encode(&canonical, field, offsets, cursors, out, ctx)
}

/// Mutate-buffer kernel: write this column's per-row bytes into `out` at
/// `offsets[i] + cursors[i]`, advancing `cursors[i]` by the bytes written.
///
/// Return `Ok(None)` to decline and fall back to the canonical path.
pub trait RowEncodeKernel: VTable {
    /// Write this column's per-row bytes into `out` at `offsets[i] + cursors[i]`, advancing
    /// `cursors[i]` by the bytes written.
    fn row_encode_into(
        column: ArrayView<'_, Self>,
        field: SortField,
        offsets: &[u32],
        cursors: &mut [u32],
        out: &mut [u8],
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<()>>;
}
