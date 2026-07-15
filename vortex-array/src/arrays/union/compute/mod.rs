// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod filter;
pub(crate) use filter::filter_union;

pub(crate) mod rules;

mod slice;

mod take;
pub(crate) use take::take_union;
