// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::arrays::varbinview::ResolvedViews;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::validity::Validity;

const LONG: &str = "a value far too long to live inside its own view";

const ALPHA: &[u8] = b"alpha_long_value_0";
const BETA: &[u8] = b"beta_long_value_00";
const GAMMA: &[u8] = b"gamma_long_value_1";
const SHORT: &[u8] = b"short";

/// The rows of [`two_buffer_array`], in order.
const TWO_BUFFER_VALUES: [&[u8]; 4] = [ALPHA, SHORT, GAMMA, BETA];

fn ref_view(value: &[u8], buffer_index: u32, offset: u32) -> VortexResult<BinaryView> {
    Ok(BinaryView::new_ref(
        u32::try_from(value.len())?,
        BinaryView::prefix_of(value),
        buffer_index,
        offset,
    ))
}

/// Values over two data buffers, one at a non-zero offset, interleaved with an inlined value.
fn two_buffer_array() -> VortexResult<VarBinViewArray> {
    let mut ctx = array_session().create_execution_ctx();
    let views = Buffer::from_iter([
        ref_view(ALPHA, 0, 0)?,
        BinaryView::new_inlined(SHORT),
        ref_view(GAMMA, 1, 0)?,
        ref_view(BETA, 0, u32::try_from(ALPHA.len())?)?,
    ]);
    VarBinViewArray::try_new(
        views,
        Arc::new([
            ByteBuffer::copy_from([ALPHA, BETA].concat()),
            ByteBuffer::copy_from(GAMMA),
        ]),
        DType::Binary(NonNullable),
        Validity::NonNullable,
        &mut ctx,
    )
}

#[test]
fn resolves_values_across_data_buffers() -> VortexResult<()> {
    let array = two_buffer_array()?;
    let resolved = ResolvedViews::new(&array);

    assert_eq!(resolved.len(), TWO_BUFFER_VALUES.len());
    assert!(!resolved.is_empty());
    for (index, expected) in TWO_BUFFER_VALUES.iter().enumerate() {
        assert_eq!(resolved.bytes(index), *expected);
    }
    Ok(())
}

#[rstest]
#[case::inlined(VarBinViewArray::from_iter_str(["a", "", "twelve bytes"]))]
#[case::referenced(VarBinViewArray::from_iter_str([LONG, "another long value here"]))]
#[case::mixed(VarBinViewArray::from_iter_str(["short", LONG, "", "twelve bytes"]))]
fn bytes_matches_bytes_at(#[case] array: VarBinViewArray) {
    let resolved = ResolvedViews::new(&array);
    for index in 0..array.len() {
        assert_eq!(resolved.bytes(index), array.bytes_at(index).as_slice());
    }
}

#[test]
fn view_bytes_matches_bytes() -> VortexResult<()> {
    let array = two_buffer_array()?;
    let resolved = ResolvedViews::new(&array);

    for (index, view) in resolved.views().iter().enumerate() {
        assert_eq!(resolved.view_bytes(view), resolved.bytes(index));
        // SAFETY: index < resolved.len().
        assert_eq!(unsafe { resolved.view_unchecked(index) }, view);
    }
    Ok(())
}

/// Slicing narrows the views, not the buffers.
#[test]
fn resolves_through_a_slice() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = two_buffer_array()?
        .slice(2..4)?
        .execute::<VarBinViewArray>(&mut ctx)?;
    let resolved = ResolvedViews::new(&array);

    assert_eq!(resolved.len(), 2);
    assert_eq!(array.data_buffers().len(), 2, "slice kept both buffers");
    assert_eq!(resolved.bytes(0), GAMMA);
    assert_eq!(resolved.bytes(1), BETA);
    Ok(())
}

#[test]
fn suffix_bytes_are_value_suffixes() -> VortexResult<()> {
    let array = two_buffer_array()?;
    let resolved = ResolvedViews::new(&array);

    for (index, view) in resolved.views().iter().enumerate() {
        let value = resolved.bytes(index);
        for suffix_len in 0..=value.len() {
            // SAFETY: suffix_len <= value.len(), which is view.len().
            let suffix = unsafe { resolved.suffix_bytes_unchecked(view, suffix_len) };
            assert_eq!(suffix, &value[value.len() - suffix_len..]);
        }
    }
    Ok(())
}

#[rstest]
#[case::inlined(["short", "tiny"], true)]
#[case::referenced([LONG, "another long value here"], true)]
#[case::inlined_non_ascii(["short", "é"], false)]
#[case::referenced_non_ascii([LONG, "a long value ending in é"], false)]
fn is_ascii(#[case] values: [&str; 2], #[case] expected: bool) {
    let array = VarBinViewArray::from_iter_str(values);
    assert_eq!(ResolvedViews::new(&array).is_ascii(), expected);
}
