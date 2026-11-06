// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduceAdaptor;
use vortex_session::VortexSession;

use crate::Sequence;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, Sequence, const { &CastReduceAdaptor(Sequence) });
    kernels.register_reduce_parent_rule(
        ListContains,
        Sequence,
        const { &ListContainsElementReduceAdaptor(Sequence) },
    );
    kernels.register_reduce_parent_rule(Slice, Sequence, const { &SliceReduceAdaptor(Sequence) });
}
