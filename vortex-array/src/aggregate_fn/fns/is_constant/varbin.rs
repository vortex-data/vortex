// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::Ref;

pub(super) fn check_varbinview_constant(array: &VarBinViewArray) -> bool {
    let mut views_iter = array.views().iter();
    let first_value = views_iter
        .next()
        .vortex_expect("Must have at least one value");

    if first_value.is_inlined() {
        let first_value = first_value.as_inlined();

        for view in views_iter {
            if !view.is_inlined() || view.as_inlined() != first_value {
                return false;
            }
        }
    } else {
        let ref_bytes = |view_ref: &Ref| {
            &array.buffer(view_ref.buffer_index as usize).as_slice()[view_ref.as_range()]
        };

        let first_view_ref = first_value.as_view();
        let first_value_bytes = ref_bytes(first_view_ref);

        for view in views_iter {
            if view.is_inlined() || view.len() != first_value.len() {
                return false;
            }

            let view_ref = view.as_view();
            let value = ref_bytes(view_ref);
            if value != first_value_bytes {
                return false;
            }
        }
    }

    true
}
