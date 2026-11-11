// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.variant_pack` scalar function.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::StructArray;
use crate::arrays::VariantArray;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::validity::Validity;

/// Builds a Variant from its core storage and its shredded child.
///
/// This is how a Variant that is stored as separate columns is put back together: `core` holds
/// the Variant values without what `shredded` holds, and `shredded` is the typed tree of the
/// fields that were shredded out. The result is the canonical [`VariantArray`] over the two.
///
/// A shredded struct is null in the rows that are not objects. Stored with that validity it
/// would be a nullable struct, which a scan cannot push a field read through, so the validity
/// can be a third child, `validity`, over a `shredded` struct that is not nullable. The fields
/// of such a struct are null wherever the struct was.
///
/// [`VariantGet`](super::variant_get::VariantGet) over this function reads a shredded path from
/// `shredded` alone, when [`VariantPackOptions::complete`] says that `core` cannot hold it.
#[derive(Clone)]
pub struct VariantPack;

impl VariantPack {
    /// Creates a lazy Variant over `core` and `shredded`.
    ///
    /// # Errors
    ///
    /// Returns an error if `core` is not Variant data or the two differ in length.
    pub fn try_new(
        core: ArrayRef,
        shredded: ArrayRef,
        options: VariantPackOptions,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(VariantPack.bind(options), [core, shredded])
    }

    /// Creates a lazy Variant over `core` and a `shredded` struct whose validity is `validity`.
    ///
    /// # Errors
    ///
    /// Returns an error if `core` is not Variant data, `shredded` is not a struct that is not
    /// nullable, `validity` is not a boolean array that is not nullable, or they differ in
    /// length.
    pub fn try_new_with_validity(
        core: ArrayRef,
        shredded: ArrayRef,
        validity: ArrayRef,
        complete: bool,
    ) -> VortexResult<ScalarFnArray> {
        let options = VariantPackOptions {
            complete,
            shredded_validity: true,
        };
        ScalarFnArray::try_new(VariantPack.bind(options), [core, shredded, validity])
    }
}

/// Options for [`VariantPack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VariantPackOptions {
    complete: bool,
    shredded_validity: bool,
}

impl VariantPackOptions {
    /// Creates options for a shredded child that is `complete` or not.
    pub fn new(complete: bool) -> Self {
        Self {
            complete,
            shredded_validity: false,
        }
    }

    /// Creates options for a shredded struct that is `complete` or not, whose validity is a
    /// third child.
    pub fn with_shredded_validity(complete: bool) -> Self {
        Self {
            complete,
            shredded_validity: true,
        }
    }

    /// Whether the validity of the shredded struct is a third child.
    pub fn shredded_validity(&self) -> bool {
        self.shredded_validity
    }

    /// Whether the shredded child holds every value it could hold.
    ///
    /// The core storage keeps a value that did not fit the shredded child: a value of another
    /// type than a shredded leaf, or an object that was not split into a shredded struct. A
    /// complete shredded child has no such row, so wherever it is null the path it represents
    /// has no value, and the core storage need not be read to tell.
    pub fn complete(&self) -> bool {
        self.complete
    }
}

impl Display for VariantPackOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", if self.complete { "complete" } else { "partial" })?;
        if self.shredded_validity {
            write!(f, ", validity")?;
        }
        Ok(())
    }
}

impl ScalarFnVTable for VariantPack {
    type Options = VariantPackOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_pack");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![
            u8::from(options.complete),
            u8::from(options.shredded_validity),
        ]))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        match metadata {
            [complete @ (0 | 1), shredded_validity @ (0 | 1)] => Ok(VariantPackOptions {
                complete: *complete == 1,
                shredded_validity: *shredded_validity == 1,
            }),
            _ => vortex_bail!("invalid VariantPack options: {metadata:?}"),
        }
    }

    fn arity(&self, options: &Self::Options) -> Arity {
        Arity::Exact(if options.shredded_validity { 3 } else { 2 })
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("core"),
            1 => ChildName::from("shredded"),
            2 => ChildName::from("validity"),
            _ => unreachable!("Invalid child index {child_idx} for VariantPack expression"),
        }
    }

    fn return_dtype(&self, options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].is_variant(),
            "VariantPack core storage must be a Variant, found {}",
            arg_dtypes[0]
        );
        if options.shredded_validity {
            vortex_ensure!(
                arg_dtypes[1].is_struct() && !arg_dtypes[1].is_nullable(),
                "VariantPack takes the validity of a struct that is not nullable, found {}",
                arg_dtypes[1]
            );
            vortex_ensure_eq!(
                arg_dtypes[2],
                DType::Bool(Nullability::NonNullable),
                "VariantPack validity must be a non-nullable boolean array"
            );
        }
        Ok(arg_dtypes[0].clone())
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let shredded = args.get(1)?;
        let shredded = if options.shredded_validity {
            let fields = shredded.execute::<StructArray>(ctx)?;
            StructArray::try_new(
                fields.names().clone(),
                fields.iter_unmasked_fields().cloned().collect::<Vec<_>>(),
                fields.len(),
                Validity::Array(args.get(2)?),
            )?
            .into_array()
        } else {
            shredded
        };
        Ok(VariantArray::try_new(args.get(0)?, Some(shredded))?.into_array())
    }
}
