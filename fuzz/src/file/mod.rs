// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use libfuzzer_sys::arbitrary::{Arbitrary, Unstructured};
use vortex_array::ArrayRef;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_expr::Expression;
use vortex_expr::arbitrary::{filter_expr, projection_expr};

use crate::array::CompressorStrategy;

#[derive(Debug)]
pub struct FuzzFileAction {
    pub array: ArrayRef,
    pub projection_expr: Option<Expression>,
    pub filter_expr: Option<Expression>,
    pub compressor_strategy: CompressorStrategy,
}

impl<'a> Arbitrary<'a> for FuzzFileAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> libfuzzer_sys::arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary(u)?.0;
        let dtype = array.dtype().clone();
        Ok(FuzzFileAction {
            array,
            projection_expr: projection_expr(u, &dtype)?,
            filter_expr: filter_expr(u, &dtype)?,
            compressor_strategy: CompressorStrategy::arbitrary(u)?,
        })
    }
}
