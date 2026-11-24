// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeBinary;
use crate::arrays::FixedSizeBinaryArray;
use crate::arrays::Masked;
use crate::arrays::Slice;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::builtin::mask::Mask as MaskFn;
use crate::scalar_fn::builtin::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(
        Masked,
        FixedSizeBinary,
        const { &FixedSizeBinaryMaskedValidityRule },
    );
    kernels.register_reduce_parent_rule(
        MaskFn,
        FixedSizeBinary,
        const { &MaskReduceAdaptor(FixedSizeBinary) },
    );
    kernels.register_reduce_parent_rule(
        Slice,
        FixedSizeBinary,
        const { &SliceReduceAdaptor(FixedSizeBinary) },
    );
}

/// Rule to push down validity masking from a [`Masked`] parent into a fixed-size binary child.
///
/// The parent's validity is merged with the child's validity, removing the `Masked` wrapper.
#[derive(Default, Debug)]
pub struct FixedSizeBinaryMaskedValidityRule;

impl ArrayParentReduceRule<FixedSizeBinary> for FixedSizeBinaryMaskedValidityRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, FixedSizeBinary>,
        parent: ArrayView<'_, Masked>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: Only validity bits are cleared, so every invariant still holds.
        let masked = unsafe {
            FixedSizeBinaryArray::new_unchecked_handle(
                array.buffer_handle().clone(),
                array.byte_width(),
                array.len(),
                array.validity()?.and(parent.validity()?)?,
            )
        };
        Ok(Some(masked.into_array()))
    }
}

impl SliceReduce for FixedSizeBinary {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let byte_width = array.byte_width() as usize;
        let values = array
            .buffer_handle()
            .slice(range.start * byte_width..range.end * byte_width);
        let validity = array.validity()?.slice(range.clone())?;

        // SAFETY: Slicing on row boundaries keeps `byte_width` bytes per row, and the validity is
        // sliced to the same rows.
        let result = unsafe {
            FixedSizeBinaryArray::new_unchecked_handle(
                values,
                array.byte_width(),
                range.len(),
                validity,
            )
        };
        Ok(Some(result.into_array()))
    }
}
