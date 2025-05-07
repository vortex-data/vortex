use std::sync::LazyLock;

use vortex_dtype::ExtID;

mod array;
mod arrow;
mod types;

pub use types::*;

pub static WKB_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.wkb"));
/// Point is an N-dimensional point. It is stored using one value per variant here.
/// The actual point is based on something like a Struct of the components.
pub static POINT_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.point"));

pub static LINESTRING_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.linestring"));

/// Polygon consists of a pair of "rings", which are lists of points defining the
/// exterior and interior boundaries of the polygon.
pub static POLYGON_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.polygon"));
