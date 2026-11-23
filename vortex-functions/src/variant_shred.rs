// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scalar functions that shred an unshredded Variant by a schema.
//!
//! Each function is one shredding behaviour, and differs only in what happens to a value that
//! does not fit the schema:
//!
//! - [`VariantShred`] keeps it in the core storage, as Parquet shredding does.
//! - [`VariantShredStrict`] fails.
//! - [`VariantShredUnion`] keeps it in the core storage too, but its schema can hold unions, so
//!   that a value of each type in the union fits.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::proto::dtype as pb;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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

/// Shreds an unshredded Variant by a schema, and fails on a value that does not fit it.
///
/// A value fits the schema as for [`VariantShred`], and a Variant null fits every position of
/// it. So the shredded child holds every value at the paths that the schema names, and the core
/// storage holds only the object fields that the schema does not name.
///
/// The encoding of the Variant storage does the work, through a kernel that it registers for this
/// function. An encoding without such a kernel cannot be shredded, and execution fails.
#[derive(Clone)]
pub struct VariantShredStrict;

/// Shreds an unshredded Variant by a schema that can hold unions.
///
/// A union can be at any position of the schema, and its members can be leaves, structs, lists
/// and unions. A value goes to the first member that holds all of it: a leaf of its type, a
/// struct that names each field of the object and holds the value of each, or a list that holds
/// each element. The core storage keeps a value that no member holds, as [`VariantShred`] keeps
/// a value that does not fit.
///
/// A member holds the whole value, so the core storage has nothing of a value that a member
/// holds. A union is the only node of the schema whose shredded child is not the typed column
/// that Parquet shreds into, so a Variant shredded by such a schema is not Parquet storage.
///
/// The encoding of the Variant storage does the work, through a kernel that it registers for this
/// function. An encoding without such a kernel cannot be shredded, and execution fails.
#[derive(Clone)]
pub struct VariantShredUnion;

impl VariantShredUnion {
    /// Creates a lazy shredding of the Variant `input` by `schema`, which can hold unions.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not a Variant.
    pub fn try_new(input: ArrayRef, schema: DType) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            VariantShredUnion.bind(VariantShredOptions::new(schema)),
            [input],
        )
    }
}

impl VariantShredStrict {
    /// Creates a lazy strict shredding of the Variant `input` by `schema`.
    ///
    /// # Errors
    ///
    /// Returns an error if `input` is not a Variant.
    pub fn try_new(input: ArrayRef, schema: DType) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            VariantShredStrict.bind(VariantShredOptions::new(schema)),
            [input],
        )
    }
}

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

    fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(pb::DType::try_from(&self.schema)?.encode_to_vec()))
    }

    fn deserialize(metadata: &[u8], session: &VortexSession) -> VortexResult<Self> {
        let dtype = pb::DType::decode(metadata)?;
        Ok(Self::new(DType::from_proto(&dtype, session)?))
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
        options.serialize()
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        VariantShredOptions::deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        input_name(child_idx)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        input_dtype(arg_dtypes)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_on_storage(self, options, args, ctx)
    }
}

impl ScalarFnVTable for VariantShredStrict {
    type Options = VariantShredOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_shred.strict");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.serialize()
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        VariantShredOptions::deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        input_name(child_idx)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        input_dtype(arg_dtypes)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_on_storage(self, options, args, ctx)
    }
}

impl ScalarFnVTable for VariantShredUnion {
    type Options = VariantShredOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_shred.union");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        options.serialize()
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        VariantShredOptions::deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        input_name(child_idx)
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        input_dtype(arg_dtypes)
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        execute_on_storage(self, options, args, ctx)
    }
}

fn input_name(child_idx: usize) -> ChildName {
    match child_idx {
        0 => ChildName::from("input"),
        _ => unreachable!("Invalid child index {child_idx} for a Variant shredding"),
    }
}

/// A shredding returns a Variant of the dtype of its `input`.
fn input_dtype(arg_dtypes: &[DType]) -> VortexResult<DType> {
    vortex_ensure!(
        arg_dtypes[0].is_variant(),
        "A Variant shredding input must be a Variant, found {}",
        arg_dtypes[0]
    );
    Ok(arg_dtypes[0].clone())
}

/// Hands the core storage of the input to the kernel of its encoding for `shredding`.
///
/// The kernels are keyed on the storage encodings. An input that is a canonical Variant, or a
/// lazy array over one, reaches this point first.
fn execute_on_storage<V: ScalarFnVTable<Options = VariantShredOptions>>(
    shredding: &V,
    options: &VariantShredOptions,
    args: &dyn ExecutionArgs,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let input = args.get(0)?;
    let variant = input.clone().execute::<VariantArray>(ctx)?;
    vortex_ensure!(
        variant.shredded().is_none(),
        "A Variant shredding takes a Variant that is not shredded"
    );

    let core = variant.core_storage();
    if core.encoding_id() == input.encoding_id() {
        vortex_bail!(
            "Variant storage {} has no kernel for {}",
            core.encoding_id(),
            shredding.id()
        );
    }
    ScalarFnArray::try_new(shredding.bind(options.clone()), [core.clone()])?
        .into_array()
        .execute::<ArrayRef>(ctx)
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::*;

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
