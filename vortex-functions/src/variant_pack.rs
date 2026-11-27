// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `vortex.variant_pack` scalar function.

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::arrays::VariantArray;
use vortex_array::dtype::DType;
use vortex_array::proto::expr as pb;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ReduceNode;
use vortex_array::scalar_fn::ReduceNodeValidity;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::is_not_null_node;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

/// Builds a Variant from its core storage and its shredded child.
///
/// This is how a Variant that is stored as separate columns is put back together: `core` holds
/// the Variant values without what `shredded` holds, and `shredded` is the typed tree of the
/// fields that were shredded out. The result is the canonical [`VariantArray`] over the two.
///
/// The packed Variant takes its SQL validity from `core`. A shredded struct carries its
/// own validity, which masks its fields.
///
/// [`VariantGet`](super::variant_get::VariantGet) over this function reads a shredded path from
/// `shredded` and the applicable validity masks when [`VariantPackOptions::complete`] says
/// that `core` cannot hold it.
#[derive(Clone)]
pub struct VariantPack;

impl VariantPack {
    /// Creates a lazy Variant over `core` and `shredded`, which is `complete` or not; see
    /// [`VariantPackOptions::complete`].
    ///
    /// # Errors
    ///
    /// Returns an error if `core` is not Variant data or the two differ in length.
    pub fn try_new(
        core: ArrayRef,
        shredded: ArrayRef,
        complete: bool,
    ) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            VariantPack.bind(VariantPackOptions::new(complete)),
            [core, shredded],
        )
    }
}

/// Options for [`VariantPack`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VariantPackOptions {
    complete: bool,
}

impl VariantPackOptions {
    /// Creates options for a shredded child that is `complete` or not.
    pub fn new(complete: bool) -> Self {
        Self { complete }
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
        f.write_str(if self.complete { "complete" } else { "partial" })
    }
}

impl ScalarFnVTable for VariantPack {
    type Options = VariantPackOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.variant_pack");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::VariantPackOpts {
                complete: options.complete,
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let opts = pb::VariantPackOpts::decode(metadata)?;

        Ok(VariantPackOptions::new(opts.complete))
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("core"),
            1 => ChildName::from("shredded"),
            _ => unreachable!("Invalid child index {child_idx} for VariantPack expression"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            arg_dtypes[0].is_variant(),
            "VariantPack core storage must be a Variant, found {}",
            arg_dtypes[0]
        );
        Ok(arg_dtypes[0].clone())
    }

    fn validity<T: ReduceNode>(
        &self,
        _options: &Self::Options,
        node: &T,
    ) -> VortexResult<ReduceNodeValidity<T>> {
        Ok(ReduceNodeValidity::Reduced(is_not_null_node(
            &node.child(0),
        )?))
    }

    fn execute(
        &self,
        _options: &Self::Options,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(VariantArray::try_new(args.get(0)?, Some(args.get(1)?))?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::VariantPack;
    use super::VariantPackOptions;

    #[rstest]
    fn options_round_trip(#[values(false, true)] complete: bool) -> VortexResult<()> {
        let options = VariantPackOptions::new(complete);
        let metadata = VariantPack.serialize(&options)?.unwrap_or_default();

        let actual = VariantPack.deserialize(&metadata, &VortexSession::empty())?;

        assert_eq!(actual, options);
        Ok(())
    }

    #[test]
    fn malformed_options_are_an_error() {
        // A field-1 varint tag with no value.
        let result = VariantPack.deserialize(&[0x08], &VortexSession::empty());

        assert!(result.is_err());
    }
}
