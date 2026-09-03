// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::BloomPartial;

pub(super) fn accumulate_varbin(
    array: &VarBinViewArray,
    partial: &mut BloomPartial,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Utility function to process views in both validity cases.
    // Other agg. functions handle both cases together and always
    // check the validity, even when all values are valid.
    // I don't think there is much difference.
    let mut process_view = |view: &BinaryView, buffers: &[&Buffer<u8>]| {
        if view.is_inlined() {
            partial.insert(view.as_inlined().value());
        } else {
            let view_ref = view.as_view();
            let value = &buffers[view_ref.buffer_index as usize][view_ref.as_range()];
            partial.insert(value);
        }
    };

    match array.validity()?.execute_mask(array.len(), ctx)? {
        Mask::AllTrue(_) => {
            let buffers = array
                .data_buffers()
                .iter()
                .map(|b| b.as_host())
                .collect::<Vec<_>>();
            array
                .views()
                .iter()
                .for_each(|view| process_view(view, &buffers));
        }
        Mask::AllFalse(_) => {}
        Mask::Values(mask_values) => {
            let views_iter = array.views().iter();
            let buffers = array
                .data_buffers()
                .iter()
                .map(|b| b.as_host())
                .collect::<Vec<_>>();

            views_iter
                .zip(mask_values.bit_buffer())
                .for_each(|(view, valid)| {
                    if valid {
                        process_view(view, &buffers)
                    }
                });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::build_filter;
    use crate::layouts::zoned::aggregates::bloom_filter::test_utils::setup;

    #[rstest]
    #[case::inlined(&["a", "Lorem", "ipsum"], "neverever")] // inlined vs non-inline
    #[case::buffered(
        &[
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua",
        ],
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip",
    )]
    fn roundtrips_options_and_membership_varbin(
        #[case] present: &[&str],
        #[case] absent: &str,
    ) -> VortexResult<()> {
        let ctx = setup()?;
        let dtype = DType::Utf8(Nullability::NonNullable);
        let batch = VarBinViewArray::from_iter_str(present.iter().copied());
        let bloom_filter = build_filter(batch.into_array(), dtype, ctx)?;

        for &v in present {
            let scalar = Scalar::binary(v.as_bytes().to_vec(), Nullability::NonNullable);
            assert!(bloom_filter.contains_scalar(&scalar)?);
        }

        let absent_scalar = Scalar::binary(absent.as_bytes().to_vec(), Nullability::NonNullable);
        assert!(!bloom_filter.contains_scalar(&absent_scalar)?);
        Ok(())
    }

    #[test]
    fn roundtrips_options_and_membership_varbin_mixed_with_nulls() -> VortexResult<()> {
        let ctx = setup()?;
        let dtype = DType::Utf8(Nullability::Nullable);
        let values = VarBinViewArray::from_iter(
            vec![
                Some("short"),
                None,
                Some("Lorem ipsum dolor sit amet, consectetur adipiscing elit"),
                None,
            ],
            dtype.clone(),
        );
        let bloom_filter = build_filter(values.into_array(), dtype, ctx)?;

        let present = Scalar::binary(b"short".to_vec(), Nullability::NonNullable);
        assert!(bloom_filter.contains_scalar(&present)?);

        let present_long = Scalar::binary(
            b"Lorem ipsum dolor sit amet, consectetur adipiscing elit".to_vec(),
            Nullability::NonNullable,
        );
        assert!(bloom_filter.contains_scalar(&present_long)?);

        let absent = Scalar::binary(b"never ever".to_vec(), Nullability::NonNullable);
        assert!(!bloom_filter.contains_scalar(&absent)?);
        Ok(())
    }
}
