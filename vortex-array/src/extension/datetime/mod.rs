// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Datetime extension DTypes, compatible with Apache Arrow.
//!
//! The temporal types also own the conversions between them, because only a temporal type knows
//! what its stored counts mean, and the generic `cast` treats extension types as opaque. The
//! `convert` module holds the exact casts that `cast` allows between these types. The
//! `operations` module holds the named conversions that need an explicit rounding mode or time zone
//! rule, such as [`change_time_unit`](crate::expr::change_time_unit).

pub(crate) mod convert;
mod date;
mod matcher;
pub(crate) mod operations;
mod time;
mod timestamp;
mod unit;

pub use date::*;
pub use matcher::*;
pub use operations::TemporalRounding;
pub use time::*;
pub use timestamp::*;
pub use unit::*;
