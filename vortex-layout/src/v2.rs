// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LayoutPlan v2 — pushdown-first execution model for Vortex layouts.
//!
//! This module defines the trait skeleton. No layouts implement
//! [`LayoutPlan`] yet; [`crate::Layout::plan`] returns
//! `vortex_bail!`. See `LAYOUT_PLAN.md` at the repo root for the
//! design.

pub mod chunked;
pub mod demand;
pub mod flat;
pub mod plan;
pub mod struct_;
pub mod toggle;
