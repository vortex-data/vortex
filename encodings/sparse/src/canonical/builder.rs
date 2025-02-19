use vortex_array::builders::{BoolBuilder, PrimitiveBuilder};
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::IntoArrayVariant;
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{VortexError, VortexExpect, VortexResult};
use vortex_scalar::PValue;

use crate::SparseArray;

#[allow(clippy::cast_possible_truncation)]
pub(super) fn canonicalize_primitive_into<T: NativePType + TryFrom<PValue, Error = VortexError>>(
    sparse: &SparseArray,
    builder: &mut PrimitiveBuilder<T>,
) -> VortexResult<()> {
    // Fill the output buffer with the fill value.
    builder.values.fill(
        sparse
            .fill_scalar()
            .as_primitive()
            .typed_value()
            .vortex_expect("fill value"),
    );

    let patches = sparse.resolved_patches()?;
    let indices = patches.indices().clone().into_primitive()?;
    let values = patches.values().clone().into_primitive()?;

    // Scatter the values into the output buffer
    match_each_integer_ptype!(indices.ptype(), |$I| {
        let indices = indices.as_slice::<$I>();
        for (&index, &value) in indices.iter().zip(values.as_slice::<T>()) {
            builder.values[index as usize] = value;
        }
    });

    // Set the validity from the sparse array.
    builder.nulls.append_validity_mask(sparse.validity_mask()?);

    Ok(())
}

// bool
pub(super) fn canonicalize_bool_into(
    sparse: &SparseArray,
    builder: &mut BoolBuilder,
) -> VortexResult<()> {
    // Populate the buffer with the fill value
    builder.inner.append_n(
        sparse.len(),
        sparse.fill_scalar().as_bool().value().unwrap_or_default(),
    );

    let patches = sparse.resolved_patches()?;
    let indices = patches.indices().clone().into_primitive()?;
    let values = patches.values().clone().into_bool()?;

    // Scatter the values into the output buffer
    let indices = indices.as_slice::<u32>();
    for (&index, value) in indices.iter().zip(values.boolean_buffer().into_iter()) {
        builder.inner.set_bit(index as usize, value);
    }

    // Set the validity from the sparse array.
    builder.nulls.append_validity_mask(sparse.validity_mask()?);

    Ok(())
}

// TODO(aduffy): support for string, binary, float, struct, list
