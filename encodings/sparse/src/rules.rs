// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::not::Not;
use vortex_array::scalar_fn::fns::not::NotReduce;
use vortex_array::scalar_fn::fns::not::NotReduceAdaptor;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::Sparse;
use crate::SparseExt as _;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, Sparse, const { &CastReduceAdaptor(Sparse) });
    kernels.register_reduce_parent_rule(Not, Sparse, const { &NotReduceAdaptor(Sparse) });
}

impl NotReduce for Sparse {
    fn invert(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        let inverted_fill = array.fill_scalar().as_bool().invert().into_scalar();
        let inverted_patches = array.patches().map_values(|values| values.not())?;
        Ok(Some(
            Sparse::try_new_from_patches(inverted_patches, inverted_fill)?.into_array(),
        ))
    }
}
