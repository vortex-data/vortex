//! Field mask represents a field projection, which leads to a set of field paths under a given layout.

use vortex_error::{vortex_bail, VortexResult};

use crate::{Field, FieldPath};

// TODO(joe): ..
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldMask {
    All,
    Prefix(FieldPath),
    Exact(FieldPath),
}

// TODO(joe): ..
#[allow(missing_docs)]
impl FieldMask {
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

    pub fn field(&self) -> Option<&Field> {
        match self {
            FieldMask::All => None,
            FieldMask::Prefix(fp) | FieldMask::Exact(fp) => Some(&fp.path()[0]),
        }
    }

    pub fn matches_all(&self) -> bool {
        match self {
            FieldMask::All => true,
            FieldMask::Prefix(path) => path.is_root(),
            FieldMask::Exact(_) => false,
        }
    }

    pub fn matches_root(&self) -> bool {
        match self {
            FieldMask::All => true,
            FieldMask::Prefix(path) | FieldMask::Exact(path) => path.is_root(),
        }
    }
}
