// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composable scan planning through explicit IO and CPU state machines.
//!
//! Pending planners can move between workers. Live planners and morsels belong to
//! their worker; stage-specific phases remain private to those implementations.

pub mod morsel;
pub mod next;
pub mod planner;
