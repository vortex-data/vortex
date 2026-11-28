// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selection of narrower `Widen` children from valid values.
//!
//! Integer narrowing, floating-point narrowing, and decimal conversion use separate
//! implementations. Decimal encoding and stored-value reads share one child-selection path so they
//! cannot disagree about its precision.

mod decimal;
#[cfg(test)]
pub(crate) use decimal::smallest_decimal_dtype;

mod float;
pub(super) use float::f32_fits_in_f16;
pub(super) use float::f64_fits_in_f16;
pub(super) use float::f64_fits_in_f32;
pub use float::narrow_float_values;
pub use float::narrowest_float_ptype;
pub(super) use float::smallest_float_ptype_for_value;

mod integer;
pub(super) use integer::integer_extrema;
pub(super) use integer::smallest_integer_ptype;
