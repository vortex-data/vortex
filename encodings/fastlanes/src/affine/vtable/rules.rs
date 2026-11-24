// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_session::VortexSession;

use crate::Affine;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    session.kernels().register_reduce_parent_rule(
        Slice,
        Affine,
        const { &SliceReduceAdaptor(Affine) },
    );
}
