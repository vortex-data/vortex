// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::len::Len;
use crate::scalar_fn::fns::len::LenMode;

/// Reduce rule for `len`: compute lengths from an array's structure without reading or
/// decoding its value bytes.
///
/// Encodings implement this to push the length computation through their structure. For
/// example, FSST stores the uncompressed byte length of every value as a separate child, so
/// `octet_len` over an FSST array is just that child — the codes are never decompressed.
///
/// Returns `Ok(None)` if the rule does not apply (e.g. a character-length request over an
/// encoding that only stores byte lengths), in which case evaluation falls back to the default
/// [`Len`] execution.
pub trait LenReduce: VTable {
    fn len(array: ArrayView<'_, Self>, mode: LenMode) -> VortexResult<Option<ArrayRef>>;
}

/// Adapts a [`LenReduce`] impl into an [`ArrayParentReduceRule`] for `ScalarFnArray(Len, ...)`.
#[derive(Default, Debug)]
pub struct LenReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for LenReduceAdaptor<V>
where
    V: LenReduce,
{
    type Parent = ExactScalarFn<Len>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Len>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        <V as LenReduce>::len(array, *parent.options)
    }
}
