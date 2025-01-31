#![feature(once_cell_try)]
#![feature(trait_alias)]
mod data;
pub use data::*;
mod context;
pub use context::*;
pub mod layouts;
pub use vtable::*;
mod reader;
use std::fmt::{Display, Formatter};

pub use reader::*;

pub mod segments;
pub mod strategies;
pub mod vtable;

/// The layout ID for a flat layout
pub(crate) const FLAT_LAYOUT_ID: LayoutId = LayoutId(1);
/// The layout ID for a chunked layout
pub(crate) const CHUNKED_LAYOUT_ID: LayoutId = LayoutId(2);
/// The layout ID for a column layout
pub(crate) const COLUMNAR_LAYOUT_ID: LayoutId = LayoutId(3);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayoutId(pub u16);

impl Display for LayoutId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}
