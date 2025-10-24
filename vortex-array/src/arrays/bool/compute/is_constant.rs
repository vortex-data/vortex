// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::{BoolArray, BoolVTable};
use crate::compute::{IsConstantKernel, IsConstantKernelAdapter, IsConstantOpts};
use crate::register_kernel;

impl IsConstantKernel for BoolVTable {
    fn is_constant(&self, array: &BoolArray, opts: &IsConstantOpts) -> VortexResult<Option<bool>> {
        // If the array is small, then it is a constant time operation.
        if opts.is_negligible_cost() && array.len() > 64 {
            return Ok(None);
        }

        let true_count = array.bit_buffer().true_count();
        Ok(Some(true_count == array.len() || true_count == 0))
    }
}

register_kernel!(IsConstantKernelAdapter(BoolVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(vec![true], true)]
    #[case(vec![false; 65], true)]
    #[case({
        let mut v = vec![true; 64];
        v.push(false);
        v
    }, false)]
    fn test_is_constant(#[case] input: Vec<bool>, #[case] expected: bool) {
        let array = BoolArray::from_iter(input);
        assert_eq!(array.is_constant(), expected);
    }
}
