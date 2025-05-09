use std::sync::{Arc, LazyLock};

use arrow_array::ArrayRef;
use vortex_array::arrays::ExtensionArray;
use vortex_dtype::ExtID;

mod array;
mod arrow;
mod types;

pub use array::*;
pub use types::*;
use vortex_error::VortexResult;

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

pub struct ExtensionArray2 {
    ext_type: Arc<dyn ExtensionType>,
    storage: ArrayRef,
}

/// An extension type that can slot into the Vortex type system.
pub trait ExtensionType {
    /// Type ID. Must be globally unique.
    fn id(&self) -> &ExtID;

    /// The logical Vortex type this maps to.
    fn logical_type(&self) -> &vortex_dtype::DType;

    /// Serialized form of the metadata for ExtMetadata
    fn serialize_metadata(&self) -> Vec<u8>;

    /// Wrap up the data set in an ExtensionArray stamped with the type and any
    /// compute function overrides required.
    fn wrap(&self, array: ArrayRef) -> VortexResult<ExtensionArray>;
}

/// PointType is parameterized by the dimensions
struct PointType {
    dimensions: usize,
}

#[allow(unused)]
impl ExtensionType for PointType {
    fn id(&self) -> &ExtID {
        &*POINT_ID
    }

    fn logical_type(&self) -> &vortex_dtype::DType {
        // One of the different struct types, based on our core logical type here.
    }

    /// Wrap the array with a properly built extension array and a runtime-determined VTable pointer.
    fn wrap(&self, array: ArrayRef) -> VortexResult<ExtensionArray> {
        todo!()
    }

    fn serialize_metadata(&self) -> Vec<u8> {
        // Indicator that this is the right family of types instead here.
        //
        // This is just one logical type with some metadata
        vec![self.dimensions as u8]
    }
}
