// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection of narrower `Widen` children from valid values.
//!
//! Integer narrowing and decimal conversion use separate implementations. Decimal encoding and
//! stored-value reads share one child-selection path so they cannot disagree about its precision.

mod decimal;
#[cfg(test)]
pub(crate) use decimal::smallest_decimal_dtype;

mod integer;
pub(super) use integer::integer_value;
pub(super) use integer::smallest_integer_ptype;
