#![feature(once_cell_try)]
#![feature(trait_alias)]
mod data;
pub use data::*;
mod context;
pub use context::*;
pub mod layouts;
use std::fmt::{Display, Formatter};

pub use reader::*;
pub use strategy::*;
pub use vtable::*;
pub use writer::*;
mod row_mask;
pub use row_mask::*;
mod reader;
pub mod scan;
pub mod segments;
pub mod stats;
mod strategy;
pub mod vtable;
mod writer;
pub mod writers;

/// The layout ID for a flat layout
pub const FLAT_LAYOUT_ID: LayoutId = LayoutId(1);
/// The layout ID for a chunked layout
pub const CHUNKED_LAYOUT_ID: LayoutId = LayoutId(2);
/// The layout ID for a column layout
pub const COLUMNAR_LAYOUT_ID: LayoutId = LayoutId(3);
/// The layout ID for a stats layout
pub const STATS_LAYOUT_ID: LayoutId = LayoutId(4);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u16);

impl Display for LayoutId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}
