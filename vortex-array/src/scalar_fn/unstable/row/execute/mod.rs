// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-loop execution for owned outputs and output sinks.
//!
//! [`owned`] stores one independent value per row and reduces compact failure evidence. [`sink`]
//! drives output builders whose row handles may share batch state.

mod owned;
pub(super) use owned::execute_owned;
pub(super) use owned::execute_owned_infallible;
pub(super) use owned::execute_owned_infallible_valid_rows;
pub(super) use owned::execute_owned_valid_rows;

mod retry;
pub(super) use retry::DenseAttempt;
pub(super) use retry::execute_owned_dense_attempt;

mod sink;
pub(super) use sink::execute_sink;
pub(super) use sink::execute_sink_valid_rows;
