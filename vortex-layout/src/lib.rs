#![feature(once_cell_try)]
#![feature(trait_alias)]
mod data;
pub use data::*;
mod context;
pub use context::*;
mod encoding;
pub mod layouts;
pub use encoding::*;
mod row_mask;
pub use row_mask::*;
pub mod operations;
mod reader;
pub use reader::*;
#[cfg(feature = "vortex-scan")]
mod scan;
pub mod segments;
pub mod strategies;

/// The layout ID for a flat layout
pub(crate) const FLAT_LAYOUT_ID: LayoutId = LayoutId(1);
/// The layout ID for a chunked layout
pub(crate) const CHUNKED_LAYOUT_ID: LayoutId = LayoutId(2);
/// The layout ID for a column layout
pub(crate) const COLUMNAR_LAYOUT_ID: LayoutId = LayoutId(3);
