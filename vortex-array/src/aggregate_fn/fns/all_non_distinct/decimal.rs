// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::dtype::BigCast;
use crate::match_each_decimal_value_type;

#[expect(
    clippy::cognitive_complexity,
    reason = "decimal widening depends on both source value types and the chosen widest type"
)]
pub(super) fn check_decimal_identical(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
) -> VortexResult<bool> {
    if lhs.values_type() == rhs.values_type() {
        return match_each_decimal_value_type!(lhs.values_type(), |S| {
            Ok(lhs.buffer::<S>().as_ref() == rhs.buffer::<S>().as_ref())
        });
    }

    let widest = lhs.values_type().max(rhs.values_type());
    match_each_decimal_value_type!(lhs.values_type(), |L| {
        match_each_decimal_value_type!(rhs.values_type(), |R| {
            match_each_decimal_value_type!(widest, |W| {
                Ok(lhs
                    .buffer::<L>()
                    .iter()
                    .zip(rhs.buffer::<R>().iter())
                    .all(|(lhs, rhs)| {
                        <W as BigCast>::from(*lhs).vortex_expect("decimal widening should succeed")
                            == <W as BigCast>::from(*rhs)
                                .vortex_expect("decimal widening should succeed")
                    }))
            })
        })
    })
}
