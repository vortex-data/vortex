// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.variant_shred` scalar function.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::VariantArray;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::dtype::DType;
use crate::dtype::proto::dtype as pb;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// Shreds an unshredded Variant by a schema.
///
/// The schema is a tree of the typed columns to shred into: a struct for an object, a list for
/// an array, and any other dtype for a typed leaf. The result holds the same values as the input.
/// Each value that fits the schema moves into the shredded child, and the core storage keeps the
/// rest. So the schema only says where the values are stored, not what they are.
///
/// The shredded child follows the shape of the schema, but its dtype is not the schema: a leaf
/// can be a Variant, which also holds the values of other types.
///
/// The encoding of the Variant storage does the work, through a kernel that it registers for this
/// function. An encoding without such a kernel cannot be shredded, and execution fails.
#[derive(Clone)]
pub struct VariantShred;

impl VariantShred {
    /// Creates a lazy shredding of the Variant `input` by `schema`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not a Variant.
    pub fn try_new(input: ArrayRef, schema: DType) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(VariantShred.bind(VariantShredOptions::new(schema)), [input])
    }
}

/// Options for [`VariantShred`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantShredOptions {
    schema: DType,
}

impl VariantShredOptions {
    /// Creates options that shred by `schema`.
    pub fn new(schema: DType) -> Self {
        Self { schema }
    }

    /// The tree of the typed columns to shred into.
    pub fn schema(&self) -> &DType {
        &self.schema
    }
}

impl Display for VariantShredOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.schema)
    }
}

impl ScalarFnVTable for VariantShred {
    type Options = VariantShredOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_shred");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(pb::DType::try_from(&options.schema)?.encode_to_vec()))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let dtype = pb::DType::decode(metadata)?;
        Ok(VariantShredOptions::new(DType::from_proto(
            &dtype, session,
        )?))
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {child_idx} for VariantShred expression"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].is_variant(),
            "VariantShred input must be a Variant, found {}",
            arg_dtypes[0]
        );
        Ok(arg_dtypes[0].clone())
    }

    /// Hands the core storage of the input to the kernel of its encoding.
    ///
    /// The kernels are keyed on the storage encodings. An input that is a canonical Variant, or
    /// a lazy array over one, reaches this point first.
    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        let variant = input.clone().execute::<VariantArray>(ctx)?;
        vortex_ensure!(
            variant.shredded().is_none(),
            "VariantShred takes a Variant that is not shredded"
        );

        let core = variant.core_storage();
        if core.encoding_id() == input.encoding_id() {
            vortex_bail!(
                "Variant storage {} has no kernel to shred it",
                core.encoding_id()
            );
        }
        VariantShred::try_new(core.clone(), options.schema.clone())?
            .into_array()
            .execute::<ArrayRef>(ctx)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;
    use crate::dtype::FieldName;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;

    #[test]
    fn options_roundtrip() -> VortexResult<()> {
        let dtype = DType::Struct(
            StructFields::from_iter([
                (
                    FieldName::from("a"),
                    DType::Primitive(PType::I64, Nullability::Nullable),
                ),
                (FieldName::from("b"), DType::Utf8(Nullability::Nullable)),
            ]),
            Nullability::Nullable,
        );
        let options = VariantShredOptions::new(dtype);

        let metadata = VariantShred.serialize(&options)?.unwrap_or_default();
        let decoded = VariantShred.deserialize(&metadata, &VortexSession::empty())?;

        assert_eq!(decoded, options);
        Ok(())
    }

    #[test]
    fn return_dtype_is_the_input_dtype() -> VortexResult<()> {
        let options = VariantShredOptions::new(DType::Utf8(Nullability::Nullable));
        let variant = DType::Variant(Nullability::NonNullable);

        assert_eq!(
            VariantShred.return_dtype(&options, std::slice::from_ref(&variant))?,
            variant
        );
        assert!(
            VariantShred
                .return_dtype(&options, &[DType::Utf8(Nullability::Nullable)])
                .is_err()
        );
        Ok(())
    }
}
