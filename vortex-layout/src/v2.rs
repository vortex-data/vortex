// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LayoutPlan v2 — pushdown-first execution model for Vortex layouts.
//!
//! This module defines the [`plans::LayoutPlan`] trait and the layout-
//! agnostic plan nodes that compose into a scan tree. The per-layout
//! `Layout::plan` overrides for `Flat` / `Chunked` / `Struct` live
//! alongside their layout definitions in `crate::layouts::*` and route
//! into these nodes.
//!
//! See `LAYOUT_PLAN.md` at the repo root for the design.

pub mod aligned;
pub mod dataflow;
pub mod demand;
pub mod domain;
pub(crate) mod experiment;
pub mod materialised_mask;
pub(crate) mod placeholder;
pub mod plans;
pub mod scan_ctx;
pub mod scheduler;
pub mod tee_stream;
pub mod toggle;
