// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::NumCast;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::ListViewRebuildMode;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::FixedSizeListBuilder;
use vortex_array::builders::ListViewBuilder;
use vortex_array::builders::VarBinBuilder;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::OffsetBuilderPType;
use vortex_array::dtype::StructFields;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::match_smallest_list_offset_type;
use vortex_array::patches::Patches;
use vortex_array::scalar::DecimalScalar;
use vortex_array::scalar::ListScalar;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::StructScalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::BufferString;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_buffer::buffer_mut;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ConstantArray;
use crate::Sparse;
use crate::SparseExt;
use crate::SparseParts;

mod lists;

/// The pieces every string decode of a sparse array starts from: the offset-resolved patch
/// indices and canonical patch values, plus the fill value's bytes (`None` for a null fill).
struct SparseStringParts {
    len: usize,
    patches: Patches,
    indices: PrimitiveArray,
    values: VarBinViewArray,
    fill: Option<ByteBuffer>,
}

impl SparseStringParts {
    fn new(array: ArrayView<'_, Sparse>, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let patches = array.resolved_patches()?;
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches
            .values()
            .clone()
            .execute::<Canonical>(ctx)?
            .into_varbinview();
        let fill_scalar = array.data().fill_scalar();
        let fill = match array.dtype() {
            DType::Utf8(_) => fill_scalar
                .as_utf8()
                .value()
                .cloned()
                .map(BufferString::into_inner),
            DType::Binary(_) => fill_scalar.as_binary().value().cloned(),
            dtype => {
                vortex_bail!("Sparse string decode of non-string dtype {dtype}")
            }
        };
        Ok(Self {
            len: array.len(),
            patches,
            indices,
            values,
            fill,
        })
    }
}

/// Scatters the patch views over the fill value straight into `builder`.
///
/// The canonical route materializes the same scattered views buffer, wraps it in an array, and
/// then `append_varbinview_array` walks all `len` views a second time to rebase their buffer
/// indices onto the builder's. Scattering into the builder instead adopts the patch buffers once
/// and costs one bulk view fill plus one view write per patch, with no byte copy.
pub(super) fn append_sparse_to_varbinview(
    array: ArrayView<'_, Sparse>,
    builder: &mut VarBinViewBuilder,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let parts = SparseStringParts::new(array, ctx)?;
    let validity = sparse_validity(
        &parts.patches,
        array.data().fill_scalar(),
        array.dtype().nullability(),
        parts.len,
        ctx,
    )?
    .execute_mask(parts.len, ctx)?;

    let mut buffers = parts
        .values
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host().clone())
        .collect::<Vec<_>>();
    // A fill value short enough to inline lives in its view; only a longer one needs its bytes
    // adopted as a (tiny) data buffer.
    let fill_view = match &parts.fill {
        Some(bytes) if bytes.len() > BinaryView::MAX_INLINED_SIZE => {
            buffers.push(bytes.clone());
            BinaryView::make_view(
                bytes.as_slice(),
                u32::try_from(buffers.len() - 1).vortex_expect("too many buffers"),
                0,
            )
        }
        Some(bytes) => BinaryView::make_view(bytes.as_slice(), 0, 0),
        None => BinaryView::make_view(&[], 0, 0),
    };

    let views = parts.values.views();
    match_each_integer_ptype!(parts.indices.ptype(), |I| {
        let indices = parts.indices.as_slice::<I>();
        builder.append_views_scattered(
            buffers,
            parts.len,
            fill_view,
            indices
                .iter()
                .zip(views.iter())
                .map(|(row, view)| (AsPrimitive::<usize>::as_(*row), *view)),
            &validity,
        );
    });
    Ok(())
}

/// Walks the rows in order, appending fill runs between the patches, straight into `builder`.
///
/// The canonical route materializes the scattered views array first; the fill-value copies per
/// row are inherent to the offsets layout either way, so appending directly just skips that
/// intermediate.
pub(super) fn append_sparse_to_varbin<O: OffsetBuilderPType>(
    array: ArrayView<'_, Sparse>,
    builder: &mut VarBinBuilder<O>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let parts = SparseStringParts::new(array, ctx)?;
    let patch_validity = parts
        .values
        .validity()?
        .execute_mask(parts.values.len(), ctx)?;

    let views = parts.values.views();
    let buffers = parts
        .values
        .data_buffers()
        .iter()
        .map(|buffer| buffer.as_host().as_slice())
        .collect::<Vec<_>>();

    match_each_integer_ptype!(parts.indices.ptype(), |I| {
        append_patch_walk_to_varbin(
            builder,
            parts.indices.as_slice::<I>(),
            parts.len,
            views,
            &buffers,
            &patch_validity,
            parts.fill.as_ref(),
        )
    })
}

/// The typed patch walk behind [`append_sparse_to_varbin`].
fn append_patch_walk_to_varbin<O: OffsetBuilderPType, I: NativePType + AsPrimitive<usize>>(
    builder: &mut VarBinBuilder<O>,
    indices: &[I],
    len: usize,
    views: &[BinaryView],
    buffers: &[&[u8]],
    patch_validity: &Mask,
    fill: Option<&ByteBuffer>,
) -> VortexResult<()>
where
    usize: AsPrimitive<O>,
{
    let fill_run = |builder: &mut VarBinBuilder<O>, n: usize| -> VortexResult<()> {
        match fill {
            Some(bytes) => builder.append_n_values(bytes, n),
            None => {
                builder.push_nulls(n);
                Ok(())
            }
        }
    };

    let mut previous = 0usize;
    for (patch, row) in indices.iter().enumerate() {
        let row = AsPrimitive::<usize>::as_(*row);
        vortex_ensure!(
            previous <= row && row < len,
            "Sparse patch indices must be ascending within the array length {len}"
        );
        fill_run(builder, row - previous)?;
        if patch_validity.value(patch) {
            builder.append_value(views[patch].bytes(buffers));
        } else {
            builder.push_null();
        }
        previous = row + 1;
    }
    fill_run(builder, len - previous)
}

fn sparse_validity(
    patches: &Patches,
    fill_value: &Scalar,
    nullability: Nullability,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    if nullability == Nullability::NonNullable {
        return Ok(Validity::NonNullable);
    }

    let fill_validity = if fill_value.is_valid() {
        Validity::AllValid
    } else {
        Validity::AllInvalid
    };
    let patch_validity = patches.values().validity()?.into_nullable();

    fill_validity.patch(
        len,
        patches.offset(),
        patches.indices(),
        &patch_validity,
        ctx,
    )
}

pub(super) fn execute_sparse(parts: SparseParts, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let SparseParts {
        patches,
        fill_value,
        dtype,
        len,
    } = parts;

    if patches.num_patches() == 0 {
        return Ok(ConstantArray::new(fill_value, len).into_array());
    }

    // Patches are already resolved (offset subtracted) by SparseParts::resolve_patches().
    Ok(match &dtype {
        DType::FixedSizeBinary(..) => vortex_bail!("FixedSizeBinary is not supported yet"),
        DType::Null => {
            assert!(fill_value.is_null());
            NullArray::new(len).into_array()
        }
        DType::Bool(..) => execute_sparse_bools(&patches, &fill_value, ctx)?,
        DType::Primitive(ptype, ..) => {
            match_each_native_ptype!(ptype, |P| {
                execute_sparse_primitives::<P>(&patches, &fill_value, ctx)?
            })
        }
        DType::Decimal(decimal_dtype, nullability) => {
            let canonical_decimal_value_type = decimal_dtype.canonical_values_type();
            let fill_decimal = fill_value.as_decimal();
            match_each_decimal_value_type!(canonical_decimal_value_type, |D| {
                execute_sparse_decimal::<D>(
                    *decimal_dtype,
                    *nullability,
                    fill_decimal,
                    &patches,
                    len,
                    ctx,
                )?
            })
        }
        dtype @ DType::Utf8(..) => {
            let fill = fill_value.as_utf8().value().cloned();
            let fill = fill.map(BufferString::into_inner);
            execute_varbin(&patches, &fill_value, dtype.clone(), fill, len, ctx)?
        }
        dtype @ DType::Binary(..) => {
            let fill = fill_value.as_binary().value().cloned();
            execute_varbin(&patches, &fill_value, dtype.clone(), fill, len, ctx)?
        }
        DType::List(values_dtype, nullability) => execute_sparse_lists(
            &patches,
            &fill_value,
            Arc::clone(values_dtype),
            len,
            *nullability,
            ctx,
        )?,
        DType::FixedSizeList(.., nullability) => {
            execute_sparse_fixed_size_list(&patches, &fill_value, len, *nullability, ctx)?
        }
        DType::Map(..) => {
            vortex_bail!("Sparse canonicalization does not support Map arrays yet")
        }
        DType::Struct(struct_fields, ..) => execute_sparse_struct(
            struct_fields,
            fill_value.as_struct(),
            dtype.nullability(),
            &patches,
            len,
            ctx,
        )?,
        DType::Union(..) => vortex_bail!("TODO(connor)[Union]: unimplemented"),
        DType::Variant(_) => {
            vortex_bail!("Sparse canonicalization does not support Variant")
        }
        DType::Extension(..) => {
            vortex_bail!("Sparse canonicalization does not support Extension arrays yet")
        }
    })
}

fn execute_sparse_lists(
    resolved: &Patches,
    fill_value: &Scalar,
    values_dtype: Arc<DType>,
    len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Patch indices are non-negative; reinterpret to unsigned so this dispatches over 4 widths
    // instead of 8. `O` is already unsigned (from `match_smallest_list_offset_type`).
    let indices = resolved.indices().as_::<Primitive>().into_owned();
    let indices = indices.reinterpret_cast(indices.ptype().to_unsigned());
    let fill_list = fill_value.as_list();

    let values = resolved.values().clone().downcast::<ListView>();

    // Keep the builder for at most one valid list: there is no repeated element gather to
    // avoid, and constructing run-end offsets would add setup work.
    let patch_validity = values.listview_validity().execute_mask(values.len(), ctx)?;
    let use_shared_elements = patch_validity.true_count() > 1;

    // Null lists also have no elements, so null and empty fills take the same path. Its output
    // shares the patch elements, so drop the ones only null lists hold: exports and element-wise
    // functions would otherwise read elements that no valid row shows.
    if fill_list.is_empty() && use_shared_elements {
        let values = values.rebuild(ListViewRebuildMode::MakeExactWithEmptyNulls, ctx)?;
        return lists::empty_fill(resolved, values, fill_value, len, nullability, ctx);
    }

    // Trim the patch elements to those the views reference, so the element count below is what
    // the builder will hold rather than an over-estimate.
    let values = values.rebuild(ListViewRebuildMode::MakeExact, ctx)?;

    // Each gap between patches is appended as one constant array, whose canonical form points
    // every view at a single copy of the fill value, so a gap costs the fill value's elements once
    // however many rows it covers. Bound the number of gaps: there is at most one on either side of
    // each patch, and each one covers at least one row.
    let n_filled = len - resolved.num_patches();
    let n_fill_runs = (resolved.num_patches() + 1).min(n_filled);
    let total_canonical_values = values.elements().len() + fill_list.len() * n_fill_runs;

    Ok(match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        match_smallest_list_offset_type!(total_canonical_values, |O| {
            execute_sparse_lists_inner::<I, O>(
                indices.as_slice(),
                values,
                &patch_validity,
                fill_value,
                values_dtype,
                len,
                nullability,
                ctx,
            )
        })
    }))
}

#[expect(clippy::too_many_arguments)]
fn execute_sparse_lists_inner<I: IntegerPType, O: OffsetBuilderPType>(
    patch_indices: &[I],
    patch_values: ListViewArray,
    patch_validity: &Mask,
    fill_value: &Scalar,
    values_dtype: Arc<DType>,
    len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> ArrayRef {
    // Create the builder with appropriate types. It is easy to just use the same type for both
    // `offsets` and `sizes` since we have no other constraints.
    let mut builder =
        ListViewBuilder::<O, O>::with_capacity_in(values_dtype, nullability, len, ctx.allocator());
    // The fill's elements become an array once, up front. Every gap then appends that same array,
    // so the fill's elements are stored once for the whole result however many gaps reference them.
    let fill_elements = list_scalar_elements_array(fill_value.as_list(), ctx.allocator());

    let mut next_index = 0;

    for ((patch_idx, sparse_idx), patch_valid) in
        patch_indices.iter().enumerate().zip(patch_validity.iter())
    {
        let sparse_idx = sparse_idx
            .to_usize()
            .vortex_expect("patch index must fit in usize");

        append_list_fill(
            &mut builder,
            fill_elements.as_ref(),
            sparse_idx - next_index,
            ctx,
        );

        // Take each patch's elements rather than slicing the patch array itself: slicing a
        // `ListView` slices its offsets, its sizes and its elements, and every one of those slices
        // pays an optimizer pass, where this pays one for the elements alone.
        if patch_valid {
            let patch_list = patch_values
                .list_elements_at(patch_idx, ctx)
                .vortex_expect("list_elements_at");
            builder
                .append_array_as_list(&patch_list, ctx)
                .vortex_expect("Failed to append sparse value");
        } else {
            builder.append_null();
        }

        next_index = sparse_idx + 1;
    }

    append_list_fill(&mut builder, fill_elements.as_ref(), len - next_index, ctx);

    builder.finish()
}

/// Materializes a list scalar's elements into an array, or `None` if the scalar is null.
fn list_scalar_elements_array(
    list: ListScalar,
    allocator: &vortex_buffer::BufferAllocatorRef,
) -> Option<ArrayRef> {
    list.elements().map(|elements| {
        let mut builder = builder_with_capacity_in(list.element_dtype(), elements.len(), allocator);
        for element in elements {
            builder
                .append_scalar(&element)
                .vortex_expect("list element scalar was invalid");
        }
        builder.finish()
    })
}

/// Appends the run of `count` fill lists that covers the gap before the next patch.
///
/// The whole run goes in as one append that points `count` views at a single copy of
/// `fill_elements`, so a gap costs nothing per row it covers.
fn append_list_fill<O: OffsetBuilderPType, S: OffsetBuilderPType>(
    builder: &mut ListViewBuilder<O, S>,
    fill_elements: Option<&ArrayRef>,
    count: usize,
    ctx: &mut ExecutionCtx,
) {
    if count == 0 {
        return;
    }

    match fill_elements {
        Some(fill_elements) => builder
            .append_array_as_repeated_list(fill_elements, count, ctx)
            .vortex_expect("Failed to append sparse fill value"),
        // A null fill has no elements to share, and the builder can record the nulls without
        // going through an array at all.
        None => builder.append_nulls(count),
    }
}

/// Canonicalize a sparse [`FixedSizeListArray`] by expanding it into a dense representation.
fn execute_sparse_fixed_size_list(
    resolved: &Patches,
    fill_value: &Scalar,
    len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices = resolved.indices().as_::<Primitive>().into_owned();
    let values = resolved.values().as_::<FixedSizeList>().into_owned();

    Ok(match_each_integer_ptype!(indices.ptype(), |I| {
        execute_sparse_fixed_size_list_inner::<I>(
            indices.as_slice(),
            values,
            fill_value,
            len,
            nullability,
            ctx,
        )
        .into_array()
    }))
}

/// Build a canonical [`FixedSizeListArray`] from sparse patches by interleaving patch values with
/// fill values.
///
/// This algorithm walks through the sparse indices sequentially, filling gaps with the fill value's
/// elements (or defaults if null). Since all lists have the same size, we can directly append
/// elements without tracking offsets.
fn execute_sparse_fixed_size_list_inner<I: IntegerPType>(
    patch_indices: &[I],
    values: FixedSizeListArray,
    fill_value: &Scalar,
    array_len: usize,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> FixedSizeListArray {
    let list_size = values.list_size();
    let element_dtype = values
        .dtype()
        .as_fixed_size_list_element_opt()
        .vortex_expect("sparse fixed-size-list values must have fixed-size-list dtype");
    let mut builder = FixedSizeListBuilder::with_capacity_in(
        Arc::clone(element_dtype),
        list_size,
        nullability,
        array_len,
        ctx.allocator(),
    );
    // The fill's elements become an array once, up front, so that a gap does not rebuild them.
    // They are tiled per row rather than shared - a fixed-size list holds its elements back to
    // back - unless they are all the same scalar, in which case the tile stays constant-encoded
    // and the tiling costs nothing.
    let fill_elements = fixed_size_list_fill_tile(fill_value.as_list(), list_size, ctx.allocator());

    // One mask for the whole patch array rather than a validity lookup per patch.
    let patch_validity = values
        .validity()
        .vortex_expect("sparse fixed-size-list validity should be derivable")
        .execute_mask(values.len(), ctx)
        .vortex_expect("sparse fixed-size-list validity mask failed to execute");

    let mut next_index = 0;

    for ((patch_idx, sparse_idx), patch_valid) in
        patch_indices.iter().enumerate().zip(patch_validity.iter())
    {
        // Fill gap before this patch with fill values.
        let sparse_idx = sparse_idx
            .to_usize()
            .vortex_expect("patch index must fit in usize");
        append_fixed_size_list_fill(
            &mut builder,
            fill_elements.as_ref(),
            sparse_idx - next_index,
            ctx,
        );

        // Take each patch's elements rather than slicing the patch array itself: slicing a
        // `FixedSizeList` slices its elements and its validity, and every one of those slices pays
        // an optimizer pass, where this pays one for the elements alone.
        if patch_valid {
            let patch_list = values
                .fixed_size_list_elements_at(patch_idx)
                .vortex_expect("fixed_size_list_elements_at");
            builder
                .append_array_as_list(&patch_list, ctx)
                .vortex_expect("Failed to append sparse fixed-size-list value");
        } else {
            builder.append_null();
        }

        next_index = sparse_idx + 1;
    }

    // Fill remaining positions after last patch.
    append_fixed_size_list_fill(
        &mut builder,
        fill_elements.as_ref(),
        array_len - next_index,
        ctx,
    );

    builder.finish_into_fixed_size_list()
}

/// Materializes the elements a fixed-size-list fill value covers each of its rows with, or `None`
/// if the fill is null.
///
/// Elements that are all the same scalar stay a constant array, so tiling them over a gap costs
/// nothing however many rows it covers.
fn fixed_size_list_fill_tile(
    fill: ListScalar,
    list_size: u32,
    allocator: &vortex_buffer::BufferAllocatorRef,
) -> Option<ArrayRef> {
    let elements = fill.elements()?;

    Some(match elements.iter().all_equal_value() {
        Ok(uniform) => ConstantArray::new(uniform.clone(), list_size as usize).into_array(),
        Err(_) => {
            let mut builder =
                builder_with_capacity_in(fill.element_dtype(), elements.len(), allocator);
            for element in &elements {
                builder
                    .append_scalar(element)
                    .vortex_expect("fixed-size-list element scalar was invalid");
            }
            builder.finish()
        }
    })
}

/// Appends the run of `count` fill lists that covers the gap before the next patch.
fn append_fixed_size_list_fill(
    builder: &mut FixedSizeListBuilder,
    fill_elements: Option<&ArrayRef>,
    count: usize,
    ctx: &mut ExecutionCtx,
) {
    if count == 0 {
        return;
    }

    match fill_elements {
        Some(fill_elements) => builder
            .append_array_as_repeated_list(fill_elements, count, ctx)
            .vortex_expect("Failed to append sparse fixed-size-list fill value"),
        // A null fill has no elements of its own, only the placeholders the builder writes.
        None => builder.append_nulls(count),
    }
}

fn execute_sparse_bools(
    patches: &Patches,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (fill_bool, validity) = if fill_value.is_null() {
        (false, Validity::AllInvalid)
    } else {
        (
            fill_value
                .try_into()
                .vortex_expect("Fill value must convert to bool"),
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let bools = BoolArray::new(BitBuffer::full(fill_bool, patches.array_len()), validity);

    Ok(bools.patch(patches, ctx)?.into_array())
}

fn execute_sparse_primitives<T: NativePType + for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
    patches: &Patches,
    fill_value: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (primitive_fill, validity) = if fill_value.is_null() {
        (T::default(), Validity::AllInvalid)
    } else {
        (
            fill_value
                .try_into()
                .vortex_expect("Fill value must convert to target T"),
            if patches.dtype().nullability() == Nullability::NonNullable {
                Validity::NonNullable
            } else {
                Validity::AllValid
            },
        )
    };

    let parray = PrimitiveArray::new(buffer![primitive_fill; patches.array_len()], validity);

    Ok(parray.patch(patches, ctx)?.into_array())
}

fn execute_sparse_struct(
    struct_fields: &StructFields,
    fill_struct: StructScalar,
    nullability: Nullability,
    // Resolution is unnecessary b/c we're just pushing the patches into the fields.
    unresolved_patches: &Patches,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let (fill_values, top_level_fill_validity) = match fill_struct.fields_iter() {
        Some(fill_values) => (fill_values.collect::<Vec<_>>(), Validity::from(nullability)),
        None => (
            struct_fields
                .fields()
                .map(|f| Scalar::default_value(&f))
                .collect::<Vec<_>>(),
            Validity::AllInvalid,
        ),
    };
    let patch_values_as_struct = unresolved_patches.values().as_::<Struct>().into_owned();
    let names = patch_values_as_struct.names();
    let validity = top_level_fill_validity.patch(
        len,
        unresolved_patches.offset(),
        unresolved_patches.indices(),
        &Validity::from_mask(
            patch_values_as_struct
                .validity()
                .vortex_expect("validity_mask")
                .execute_mask(patch_values_as_struct.len(), ctx)
                .vortex_expect("Failed to compute validity mask"),
            nullability,
        ),
        ctx,
    )?;

    let fields = patch_values_as_struct
        .iter_unmasked_fields()
        .cloned()
        .zip_eq(fill_values)
        .map(|(patch_values, fill_value)| {
            sparse_field(unresolved_patches, patch_values, fill_value)
        })
        .collect::<Vec<_>>();

    Ok(StructArray::try_new(names.clone(), fields, len, validity)?.into_array())
}

fn sparse_field(patches: &Patches, values: ArrayRef, fill_value: Scalar) -> ArrayRef {
    let patches = patches
        .clone()
        .map_values(|_| Ok(values))
        .vortex_expect("Replacing patch values");
    unsafe { Sparse::new_unchecked(patches, fill_value) }.into_array()
}

fn execute_sparse_decimal<D: NativeDecimalType>(
    decimal_dtype: DecimalDType,
    nullability: Nullability,
    fill_value: DecimalScalar,
    patches: &Patches,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Fill the buffer in one go rather than a value at a time, as the other fills do.
    let (values, validity) = match fill_value.decimal_value() {
        Some(fill_value) => (
            Buffer::full(
                fill_value
                    .cast::<D>()
                    .vortex_expect("unexpected value type"),
                len,
            ),
            Validity::from(nullability),
        ),
        None => (Buffer::<D>::zeroed(len), Validity::AllInvalid),
    };

    // SAFETY: the buffer holds `len` values of the dtype's value type, and the validity carries no
    // length of its own.
    let filled_array = unsafe { DecimalArray::new_unchecked(values, decimal_dtype, validity) };
    Ok(filled_array.patch(patches, ctx)?.into_array())
}

fn execute_varbin(
    resolved: &Patches,
    fill_scalar: &Scalar,
    dtype: DType,
    fill_value: Option<ByteBuffer>,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let indices = resolved.indices().as_::<Primitive>().into_owned();
    let values = resolved.values().as_::<VarBinView>().into_owned();
    let validity = sparse_validity(resolved, fill_scalar, dtype.nullability(), len, ctx)?;

    Ok(match_each_integer_ptype!(indices.ptype(), |I| {
        let indices = indices.to_buffer::<I>();
        execute_varbin_inner::<I>(fill_value, indices, values, dtype, validity, len).into_array()
    }))
}

fn execute_varbin_inner<I: IntegerPType>(
    fill_value: Option<ByteBuffer>,
    indices: Buffer<I>,
    values: VarBinViewArray,
    dtype: DType,
    validity: Validity,
    len: usize,
) -> VarBinViewArray {
    assert_eq!(dtype.nullability(), validity.nullability());

    let n_patch_buffers = values.data_buffers().len();
    let mut buffers = values.data_buffers().to_vec();

    let fill = match &fill_value {
        // A fill value short enough to inline lives in its view; pushing its buffer would leave
        // that buffer entirely unreferenced.
        Some(buffer) if buffer.len() > BinaryView::MAX_INLINED_SIZE => {
            buffers.push(BufferHandle::new_host(buffer.clone()));
            BinaryView::make_view(
                buffer.as_ref(),
                u32::try_from(n_patch_buffers).vortex_expect("too many buffers"),
                0,
            )
        }
        Some(buffer) => BinaryView::make_view(buffer.as_ref(), 0, 0),
        // any <=12 character value will do
        None => BinaryView::make_view(&[], 0, 0),
    };

    let mut views = buffer_mut![fill; len];
    for (patch_index, &patch) in indices.into_iter().zip_eq(values.views().iter()) {
        let patch_index_usize = <usize as NumCast>::from(patch_index)
            .vortex_expect("var bin view indices must fit in usize");
        views[patch_index_usize] = patch;
    }

    let views = BufferHandle::new_host(views.freeze().into_byte_buffer());

    // SAFETY: views are constructed to maintain the invariants
    unsafe { VarBinViewArray::new_handle_unchecked(views, Arc::from(buffers), dtype, validity) }
}

#[cfg(test)]
mod tests;
