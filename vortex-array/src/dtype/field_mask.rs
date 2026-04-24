// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Field mask represents a field projection, which leads to a set of field paths under a given layout.
// TODO(ngates): this API needs work. It could be made a lot easier to use.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::Field;
use crate::dtype::FieldPath;

/// A projection of fields under a layout.
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
            FieldMask::Prefix(fp) | FieldMask::Exact(fp) => Ok(fp.parts().first()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Field;
    use crate::dtype::FieldPath;

    // Test helper functions
    fn all() -> FieldMask {
        FieldMask::All
    }

    fn from_prefix<I: IntoIterator<Item = Field>>(fields: I) -> FieldMask {
        FieldMask::Prefix(FieldPath::from(fields.into_iter().collect::<Vec<_>>()))
    }

    fn from_exact<I: IntoIterator<Item = Field>>(fields: I) -> FieldMask {
        FieldMask::Exact(FieldPath::from(fields.into_iter().collect::<Vec<_>>()))
    }

    fn prefix_from_str(path: &str) -> FieldMask {
        if path.is_empty() {
            return FieldMask::All;
        }
        let fields: Vec<Field> = path.split('.').map(Field::from).collect();
        from_prefix(fields)
    }

    fn exact_from_str(path: &str) -> FieldMask {
        if path.is_empty() {
            return FieldMask::Exact(FieldPath::root());
        }
        let fields: Vec<Field> = path.split('.').map(Field::from).collect();
        from_exact(fields)
    }

    #[test]
    fn test_field_mask_all() {
        let mask = FieldMask::All;
        assert!(mask.matches_all());
        assert!(mask.matches_root());

        // Test builder method
        let mask2 = all();
        assert_eq!(mask, mask2);
    }

    #[test]
    fn test_field_mask_builders() {
        // Test from_prefix
        let mask = from_prefix(vec![Field::from("user")]);
        assert!(!mask.matches_all());
        assert!(!mask.matches_root());

        // Test from_exact
        let mask = from_exact(vec![Field::from("user"), Field::from("name")]);
        assert!(!mask.matches_all());
        assert!(!mask.matches_root());
    }

    #[test]
    fn test_field_mask_from_string() {
        // Test prefix_from_str
        let mask = prefix_from_str("user.profile");
        if let FieldMask::Prefix(path) = mask {
            assert_eq!(path.parts().len(), 2);
            assert_eq!(path.parts()[0], Field::from("user"));
            assert_eq!(path.parts()[1], Field::from("profile"));
        } else {
            unreachable!("Expected Prefix mask");
        }

        // Test exact_from_str
        let mask = exact_from_str("user.profile.name");
        if let FieldMask::Exact(path) = mask {
            assert_eq!(path.parts().len(), 3);
            assert_eq!(path.parts()[0], Field::from("user"));
            assert_eq!(path.parts()[1], Field::from("profile"));
            assert_eq!(path.parts()[2], Field::from("name"));
        } else {
            unreachable!("Expected Exact mask");
        }

        // Test empty string
        let mask = prefix_from_str("");
        assert_eq!(mask, FieldMask::All);

        let mask = exact_from_str("");
        assert_eq!(mask, FieldMask::Exact(FieldPath::root()));
    }

    #[test]
    fn test_field_mask_prefix_root() {
        let path = FieldPath::root();
        let mask = FieldMask::Prefix(path);
        assert!(mask.matches_all());
        assert!(mask.matches_root());
    }

    #[test]
    fn test_field_mask_prefix_non_root() {
        let path = FieldPath::from(vec![Field::from("field1")]);
        let mask = FieldMask::Prefix(path);
        assert!(!mask.matches_all());
        assert!(!mask.matches_root());
    }

    #[test]
    fn test_field_mask_exact_root() {
        let path = FieldPath::root();
        let mask = FieldMask::Exact(path);
        assert!(!mask.matches_all());
        assert!(mask.matches_root());
    }

    #[test]
    fn test_field_mask_exact_non_root() {
        let path = FieldPath::from(vec![Field::from("field1")]);
        let mask = FieldMask::Exact(path);
        assert!(!mask.matches_all());
        assert!(!mask.matches_root());
    }

    #[test]
    fn test_step_into_all() {
        let mask = FieldMask::All;
        let stepped = mask.step_into().unwrap();
        assert_eq!(stepped, FieldMask::All);
    }

    #[test]
    fn test_step_into_prefix_becomes_all() {
        let path = FieldPath::from(vec![Field::from("field1")]);
        let mask = FieldMask::Prefix(path);
        let stepped = mask.step_into().unwrap();
        assert_eq!(stepped, FieldMask::All);
    }

    #[test]
    fn test_step_into_prefix_nested() {
        let path = FieldPath::from(vec![Field::from("field1"), Field::from("field2")]);
        let mask = FieldMask::Prefix(path);
        let stepped = mask.step_into().unwrap();

        if let FieldMask::Prefix(stepped_path) = stepped {
            assert_eq!(stepped_path.parts().len(), 1);
            assert_eq!(stepped_path.parts()[0], Field::from("field2"));
        } else {
            unreachable!("Expected Prefix mask after stepping into nested path");
        }
    }

    #[test]
    fn test_step_into_exact_root_fails() {
        let path = FieldPath::root();
        let mask = FieldMask::Exact(path);
        let result = mask.step_into();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot step into exact root field path")
        );
    }

    #[test]
    fn test_step_into_exact_nested() {
        let path = FieldPath::from(vec![Field::from("field1"), Field::from("field2")]);
        let mask = FieldMask::Exact(path);
        let stepped = mask.step_into().unwrap();

        if let FieldMask::Exact(stepped_path) = stepped {
            assert_eq!(stepped_path.parts().len(), 1);
            assert_eq!(stepped_path.parts()[0], Field::from("field2"));
        } else {
            unreachable!("Expected Exact mask after stepping into nested path");
        }
    }

    #[test]
    fn test_starting_field_all_fails() {
        let mask = FieldMask::All;
        let result = mask.starting_field();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot get starting field from All mask")
        );
    }

    #[test]
    fn test_starting_field_prefix() {
        let field = Field::from("field1");
        let path = FieldPath::from(vec![field.clone()]);
        let mask = FieldMask::Prefix(path);
        let starting = mask.starting_field().unwrap();
        assert_eq!(starting, Some(&field));
    }

    #[test]
    fn test_starting_field_exact() {
        let field = Field::from("field1");
        let path = FieldPath::from(vec![field.clone(), Field::from("field2")]);
        let mask = FieldMask::Exact(path);
        let starting = mask.starting_field().unwrap();
        assert_eq!(starting, Some(&field));
    }

    #[test]
    fn test_starting_field_empty_path() {
        let path = FieldPath::root();
        let mask = FieldMask::Prefix(path);
        let starting = mask.starting_field().unwrap();
        assert_eq!(starting, None);
    }
}
