// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared structural operations for fixed-width canonical arrays.

mod array;
pub(crate) mod filter;
pub(crate) mod take;
pub(crate) mod vtable;

pub(crate) use self::array::FixedWidthArray;
pub(crate) use self::array::with_values;
