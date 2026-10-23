// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions built on the vortex-array scalar function framework that the core array
//! crate does not itself depend on.
//!
//! The scalar functions that canonical arrays, validity and the optimizer rely on (cast,
//! fill_null, mask, binary operators, get_item, pack, literal, zip, ...) live in
//! `vortex_array::scalar_fn::builtin`. This crate holds struct reshaping ([`merge`], [`select`]),
//! conditionals ([`case_when`]), list reductions ([`list_length`], [`list_sum`]), and
//! [`byte_length`], [`ext_storage`] and [`replace_time_zone`], and the scan-time functions
//! [`dynamic`] (comparisons against a value that changes during a scan) and [`stat`] (reads of
//! a statistic), along with their expression constructors in [`expr`].
//!
//! Call [`initialize`] to register them with a session.

use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_session::VortexSession;

pub mod between;
pub mod builtins;
pub mod byte_length;
pub mod case_when;
pub mod dynamic;
pub mod expr;
pub mod ext_storage;
mod field;
pub mod is_nan;
pub mod like;
pub mod list_contains;
pub mod list_length;
pub mod list_sum;
pub mod merge;
pub mod replace_time_zone;
pub mod select;
pub mod stat;
pub mod variant_get;
pub mod zip;

#[cfg(test)]
mod tests;

/// Register this crate's scalar functions with the given session.
pub fn initialize(session: &VortexSession) {
    between::initialize(session);
    like::initialize(session);
    list_contains::initialize(session);
    variant_get::initialize(session);
    zip::initialize(session);
    let scalar_fns = session.scalar_fns();
    scalar_fns.register(byte_length::ByteLength);
    scalar_fns.register(case_when::CaseWhen);
    scalar_fns.register(ext_storage::ExtStorage);
    scalar_fns.register(is_nan::IsNan);
    scalar_fns.register(list_length::ListLength);
    scalar_fns.register(list_sum::ListSum);
    scalar_fns.register(merge::Merge);
    scalar_fns.register(replace_time_zone::ReplaceTimeZone);
    scalar_fns.register(select::Select);
    scalar_fns.register(stat::StatFn);
}

/// A scalar function has a negative cost if applying it to an array and
/// canonicalizing is cheaper than canonicalizing an array and applying it.
///
/// Example of negative cost expressions are byte_length(), ext_storage(), and get_item() since
/// they don't depend on input size.
///
/// Example of non-negative cost expression is like() as it's linear over
/// individual input.
pub fn is_negative_cost(id: ScalarFnId) -> bool {
    id == ScalarFnVTable::id(&byte_length::ByteLength)
        || id == ScalarFnVTable::id(&ext_storage::ExtStorage)
        || id == ScalarFnVTable::id(&GetItem)
        || id == ScalarFnVTable::id(&Literal)
}
