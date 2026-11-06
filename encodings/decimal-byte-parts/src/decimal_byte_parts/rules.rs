// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeReduceAdaptor;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::VortexSession;

use crate::DecimalByteParts;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(
        Cast,
        DecimalByteParts,
        const { &CastReduceAdaptor(DecimalByteParts) },
    );
    kernels.register_reduce_parent_rule(
        Filter,
        DecimalByteParts,
        const { &FilterReduceAdaptor(DecimalByteParts) },
    );
    kernels.register_reduce_parent_rule(
        MaskFn,
        DecimalByteParts,
        const { &MaskReduceAdaptor(DecimalByteParts) },
    );
    kernels.register_reduce_parent_rule(
        Slice,
        DecimalByteParts,
        const { &SliceReduceAdaptor(DecimalByteParts) },
    );
    kernels.register_reduce_parent_rule(
        Dict,
        DecimalByteParts,
        const { &TakeReduceAdaptor(DecimalByteParts) },
    );
}
