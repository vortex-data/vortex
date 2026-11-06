// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::arrays::Dict;
use crate::arrays::Slice;
use crate::arrays::Union;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(MaskFn, Union, const { &MaskReduceAdaptor(Union) });
    kernels.register_reduce_parent_rule(Slice, Union, const { &SliceReduceAdaptor(Union) });
    kernels.register_reduce_parent_rule(Dict, Union, const { &TakeReduceAdaptor(Union) });
}
