//! Field mask represents a field projection, which leads to a set of field paths under a given layout.

use vortex_error::{vortex_bail, VortexResult};

use crate::{Field, FieldPath};

/// Represents a field mask, which is a projection of fields under a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMask {
    /// Select all fields in the layout
    All,
    /// Select all with the `FieldPath` prefix
    Prefix(FieldPath),
    /// Select a field matching exactly the `FieldPath`
    Exact(FieldPath),
}

impl FieldMask {
    /// Creates a new field mask stepping one level into the layout structure.
    pub fn step_into(self) -> VortexResult<Self> {
        match self {
            FieldMask::All => Ok(FieldMask::All),
            FieldMask::Prefix(fp) => {
                let Some(stepped_fp) = fp.step_into() else {
                    return Ok(FieldMask::All);
                };
                if stepped_fp.is_root() {
                    Ok(FieldMask::All)
                } else {
                    Ok(FieldMask::Prefix(stepped_fp))
                }
            }
            FieldMask::Exact(fp) => {
                if let Some(stepped_fp) = fp.step_into() {
                    Ok(FieldMask::Exact(stepped_fp))
                } else {
                    vortex_bail!("Cannot step into exact root field path");
                }
            }
        }
    }

    /// Returns the first field explicit select mask, if there is one, failing if mask = `All`.
    pub fn starting_field(&self) -> VortexResult<Option<&Field>> {
        match self {
            FieldMask::All => vortex_bail!("Cannot get starting field from All mask"),
            // We know that fp is non-empty
            FieldMask::Prefix(fp) | FieldMask::Exact(fp) => Ok(fp.path().first()),
        }
    }

    /// True iff all fields are selected (including self).
    pub fn matches_all(&self) -> bool {
        match self {
            FieldMask::All => true,
            FieldMask::Prefix(path) => path.is_root(),
            FieldMask::Exact(_) => false,
        }
    }

    /// True if the mask matches the root field.
    pub fn matches_root(&self) -> bool {
        match self {
            FieldMask::All => true,
            FieldMask::Prefix(path) | FieldMask::Exact(path) => path.is_root(),
        }
    }
}
