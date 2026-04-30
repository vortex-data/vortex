// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::expr::Expression;
use vortex_array::expr::arbitrary::filter_expr;
use vortex_array::expr::arbitrary::projection_expr;

use crate::FUZZ_FILE_ARRAY_MAX_LEN;
use crate::array::CompressorStrategy;

#[derive(Debug)]
pub struct FuzzFileAction {
    pub array: ArrayRef,
    pub projection_expr: Option<Expression>,
    pub filter_expr: Option<Expression>,
    pub compressor_strategy: CompressorStrategy,
}

impl<'a> Arbitrary<'a> for FuzzFileAction {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let array = ArbitraryArray::arbitrary_with_config(
            u,
            &ArbitraryArrayConfig {
                dtype: None,
                len: 0..=FUZZ_FILE_ARRAY_MAX_LEN,
            },
        )?
        .0;
        let dtype = array.dtype().clone();
        Ok(FuzzFileAction {
            array,
            projection_expr: projection_expr(u, &dtype)?,
            filter_expr: filter_expr(u, &dtype)?,
            compressor_strategy: CompressorStrategy::arbitrary(u)?,
        })
    }
}
