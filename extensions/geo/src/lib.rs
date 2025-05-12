use std::sync::LazyLock;

use vortex_dtype::ExtID;

mod array;
pub mod arrow;
mod types;

pub use array::*;
pub use types::*;

/// An extension type for arrays containing [Well-known Binary](https://libgeos.org/specifications/wkb/).
///
/// This is a fallback and will generally be less performant than one of the native encodings.
///
/// See [`POINT_ID`], [`POLYGON_ID`] for examples of native encodings.
pub static WKB_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.wkb"));

/// Point is an N-dimensional point. It is stored using one value per variant here.
/// The actual point is based on something like a Struct of the components.
pub static POINT_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.point"));

pub static LINESTRING_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.linestring"));

/// Polygon is represented as a list of "rings". The first ring defines the Points that make up the exterior
/// of the shape. Subsequent rings are interior "holes" in the shape.
pub static POLYGON_ID: LazyLock<ExtID> = LazyLock::new(|| ExtID::from("geovortex.polygon"));
