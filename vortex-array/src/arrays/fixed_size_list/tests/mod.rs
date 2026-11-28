// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod basic;
mod common;
mod degenerate;
mod filter;
mod nested;
mod nullability;
mod take;

use std::sync::LazyLock;

use rstest::rstest;
use vortex_session::VortexSession;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::FixedSizeListArray;
use crate::arrays::PrimitiveArray;
use crate::compute::conformance::cast::test_cast_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::dtype::Nullability;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

#[rstest]
#[case::basic(common::create_basic_fsl())]
#[case::nullable(common::create_nullable_fsl())]
#[case::nullable_elements(FixedSizeListArray::new(
    PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array(),
    2,
    Validity::NonNullable,
    2,
))]
#[case::degenerate_size_0(FixedSizeListArray::new(
    PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
    0,
    Validity::from_iter([true, false, true]),
    3,
))]
#[case::single_element(common::create_single_element_fsl())]
#[case::empty(common::create_empty_fsl())]
#[case::large(common::create_large_fsl())]
fn test_fsl_cast_and_mask_conformance(#[case] array: FixedSizeListArray) {
    let array = array.into_array();
    let mut ctx = SESSION.create_execution_ctx();
    test_cast_conformance(&array, &mut ctx);
    test_mask_conformance(&array, &mut ctx);
}
