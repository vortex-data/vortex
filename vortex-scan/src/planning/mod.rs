// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composable scan planning through explicit IO and CPU state machines.
//!
//! Planners and morsels are `Send` and move between threads with the run that owns them;
//! stage-specific phases remain private to those implementations.

pub mod morsel;
pub mod next;
pub mod planner;
