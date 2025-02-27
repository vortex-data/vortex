#![feature(once_cell_try)]
#![feature(trait_alias)]

mod data;
pub use data::*;
mod context;
pub use context::*;
pub mod layouts;

pub use reader::*;
pub use strategy::*;
pub use vtable::*;
pub use writer::*;
mod row_mask;
pub use row_mask::*;
use vortex_array::arcref::ArcRef;

mod reader;
pub mod scan;
pub mod segments;
pub mod stats;
mod strategy;
pub mod vtable;
mod writer;
pub mod writers;

pub type LayoutId = ArcRef<str>;

/// The layout ID for a flat layout
pub const FLAT_LAYOUT_ID: LayoutId = ArcRef::new_ref("vortex.flat");
/// The layout ID for a chunked layout
pub const CHUNKED_LAYOUT_ID: LayoutId = ArcRef::new_ref("vortex.chunked");
/// The layout ID for a struct layout
pub const STRUCT_LAYOUT_ID: LayoutId = ArcRef::new_ref("vortex.struct");
/// The layout ID for a stats layout
pub const STATS_LAYOUT_ID: LayoutId = ArcRef::new_ref("vortex.stats");
