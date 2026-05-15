// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LayoutPlan v2 — pushdown-first execution model for Vortex layouts.
//!
//! This module defines the [`plan::LayoutPlan`] trait and the layout-
//! agnostic plan nodes that compose into a scan tree (`flat`, `chunked`,
//! `struct_`, `project`). The per-layout `Layout::plan` overrides for
//! `Flat` / `Chunked` / `Struct` live alongside their layout
//! definitions in `crate::layouts::*` and route into these nodes.
//!
//! See `LAYOUT_PLAN.md` at the repo root for the design.

pub mod aligned;
pub mod and_bool;
pub mod chunked;
pub mod cse;
pub mod demand;
pub mod dict;
pub mod empty_struct;
pub mod filter;
pub mod filtered_flat;
pub mod flat;
pub mod let_use;
pub mod mask_collect;
pub mod mask_slice;
pub mod matcher;
pub mod plan;
pub mod project;
pub mod pushdown;
pub mod scan;
pub mod scan_ctx;
pub mod struct_;
pub mod tee_stream;
pub mod toggle;
pub mod zoned;
