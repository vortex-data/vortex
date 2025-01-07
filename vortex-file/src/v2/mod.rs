mod file;
pub mod footer;
mod open;
mod segments;
mod strategy;
#[cfg(test)]
mod tests;
mod writer;

pub use file::*;
pub use open::*;
// TODO(ngates): probably can separate these APIs? For now, re-export the Scan.
pub use vortex_layout::scanner::Scan;
pub use writer::*;
