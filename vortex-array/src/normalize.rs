// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::Array;
use crate::ArrayEq;
use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::Precision;
use crate::session::ArrayRegistry;
use crate::vtable::ArrayId;

/// Options for normalizing an array.
pub struct NormalizeOptions<'a> {
    /// The set of allowed array encodings (in addition to the canonical ones) that are permitted
    /// in the normalized array.
    allowed: &'a ArrayRegistry,
    /// The operation to perform when a non-allowed encoding is encountered.
    operation: Operation<'a>,
}

/// The operation to perform when a non-allowed encoding is encountered.
enum Operation<'a> {
    IntoCanonical(&'a mut ExecutionCtx),
    Error,
}

impl<'a> NormalizeOptions<'a> {
    /// Create a new `NormalizeOptions` that returns an error for non-allowed encodings.
    pub fn error(allowed: &'a ArrayRegistry) -> Self {
        Self {
            allowed,
            operation: Operation::Error,
        }
    }

    /// Create a new `NormalizeOptions` that canonicalizes non-allowed encodings.
    pub fn canonicalize(allowed: &'a ArrayRegistry, ctx: &'a mut ExecutionCtx) -> Self {
        Self {
            allowed,
            operation: Operation::IntoCanonical(ctx),
        }
    }

    /// Check if the given array ID is allowed.
    fn is_allowed(&self, id: &ArrayId) -> bool {
        self.allowed.find(id).is_some()
    }
}

pub trait Normalize {
    /// Normalize the array according to given options.
    ///
    /// This operation performs a recursive traversal of the array. Any non-allowed encoding is
    /// normalized per the configured operation.
    fn normalize(&self, options: &mut NormalizeOptions) -> VortexResult<ArrayRef>;
}

impl Normalize for ArrayRef {
    fn normalize(&self, options: &mut NormalizeOptions) -> VortexResult<ArrayRef> {
        if !self.is_canonical() && !options.is_allowed(&self.encoding_id()) {
            match &mut options.operation {
                Operation::IntoCanonical(ctx) => {
                    return self
                        .clone()
                        .execute::<Canonical>(ctx)?
                        .into_array()
                        .normalize(options);
                }
                Operation::Error => vortex_bail!(
                    "Array encoding '{}' is not allowed in normalized array",
                    self.encoding_id()
                ),
            }
        }

        let children = self.children();
        let mut new_children = Vec::with_capacity(children.len());
        for child in &children {
            new_children.push(child.normalize(options)?);
        }

        if children
            .iter()
            .zip(new_children.iter())
            .all(|(a, b)| a.array_eq(b, Precision::Ptr))
        {
            // No children changed; clone self.
            return Ok(self.clone());
        }

        self.with_children(new_children)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::FieldNames;
    use vortex_error::VortexResult;

    use super::*;
    use crate::LEGACY_SESSION;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::ConstantVTable;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::assert_arrays_eq;
    use crate::session::ArraySessionExt;
    use crate::validity::Validity;

    #[test]
    fn canonical_array_passes_through() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let registry = LEGACY_SESSION.arrays().registry().clone();
        let mut opts = NormalizeOptions::error(&registry);

        let result = array.normalize(&mut opts)?;
        assert_arrays_eq!(&result, &array);
        Ok(())
    }

    #[test]
    fn non_allowed_encoding_errors() {
        let array = ConstantArray::new(42i32, 5).into_array();
        let registry = ArrayRegistry::empty();
        let mut opts = NormalizeOptions::error(&registry);

        let result = array.normalize(&mut opts);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("vortex.constant"),
            "Expected error to mention encoding id, got: {err}"
        );
    }

    #[test]
    fn non_allowed_encoding_is_canonicalized() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 5).into_array();
        // Use an empty registry so ConstantArray is not allowed.
        // Canonical encodings are always allowed via is_canonical().
        let registry = ArrayRegistry::empty();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut opts = NormalizeOptions::canonicalize(&registry, &mut ctx);

        let result = array.normalize(&mut opts)?;
        assert_arrays_eq!(&result, &array);
        assert!(result.is_canonical());
        Ok(())
    }

    #[test]
    fn allowed_encoding_passes_through() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 5).into_array();
        // Create a registry that allows ConstantArray.
        let registry = ArrayRegistry::default();
        registry.register(ConstantVTable::ID, ConstantVTable);
        let mut opts = NormalizeOptions::error(&registry);

        let result = array.normalize(&mut opts)?;
        assert_arrays_eq!(&result, &array);
        Ok(())
    }

    #[test]
    fn recursive_children_are_normalized() -> VortexResult<()> {
        // Struct with a constant child - canonical struct is allowed via is_canonical(),
        // but its constant child is not in the empty registry and should be canonicalized.
        let child = ConstantArray::new(7i32, 3).into_array();
        let struct_array = StructArray::try_new(
            FieldNames::from(["values"]),
            vec![child],
            3,
            Validity::NonNullable,
        )?
        .into_array();

        let registry = ArrayRegistry::empty();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut opts = NormalizeOptions::canonicalize(&registry, &mut ctx);

        let result = struct_array.normalize(&mut opts)?;
        assert_arrays_eq!(&result, &struct_array);
        let result_child = &result.children()[0];
        assert!(result_child.is_canonical());
        Ok(())
    }

    #[test]
    fn canonical_children_are_not_reconstructed() -> VortexResult<()> {
        // When all children are already canonical, the original array is returned (by pointer).
        let child = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let struct_array = StructArray::try_new(
            FieldNames::from(["values"]),
            vec![child],
            3,
            Validity::NonNullable,
        )?
        .into_array();

        let registry = LEGACY_SESSION.arrays().registry().clone();
        let mut opts = NormalizeOptions::error(&registry);

        let result = struct_array.normalize(&mut opts)?;
        assert!(result.array_eq(&struct_array, Precision::Ptr));
        Ok(())
    }
}
