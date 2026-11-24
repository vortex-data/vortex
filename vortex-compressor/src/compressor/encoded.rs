// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compressing the children a scheme registers and building its array from them.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::child_to_validity;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::CascadingCompressor;
use crate::scheme::Child;
use crate::scheme::ChildDecision;
use crate::scheme::ChildKind;
use crate::scheme::ChildRequest;
use crate::scheme::CompressorContext;
use crate::scheme::Encoded;
use crate::scheme::EncodedChildren;
use crate::scheme::SchemeId;

impl CascadingCompressor {
    /// Compresses the children of an [`Encoded`] and builds its array from them.
    ///
    /// Children in [`Encoded::compress`] and the bitmap of [`Encoded::validity`] are compressed,
    /// children in [`Encoded::keep`] are passed through, unless the compressor's
    /// [`CompressionPolicy`](crate::scheme::CompressionPolicy) decides otherwise. `scheme` names the scheme
    /// that produced the encoding in errors.
    ///
    /// Estimate callbacks that run a trial compression use this to compress a trial encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if compressing a child or building the array fails, or if the constructor
    /// leaves a registered child unused. In debug builds, also returns an error if the built array
    /// holds a canonical array that was not registered.
    pub fn compress_encoded(
        &self,
        scheme: SchemeId,
        encoded: Encoded,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            !encoded.is_uses_compress(),
            "scheme {scheme} implements only Scheme::compress, so it has no encoding to compress"
        );
        let (compress, keep, validity, ctor) = encoded.into_parts();

        let compressed = compress
            .into_iter()
            .map(|child| self.resolve_child(child, ChildDecision::Compress, compress_ctx, exec_ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let kept = keep
            .into_iter()
            .map(|child| self.resolve_child(child, ChildDecision::Keep, compress_ctx, exec_ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        let validity = validity
            .map(|validity| self.resolve_validity(validity, compress_ctx, exec_ctx))
            .transpose()?;

        #[cfg(debug_assertions)]
        let registered = check::Registered::new(&compressed, &kept, validity.as_ref());

        let mut children = EncodedChildren::from_parts(compressed, kept, validity);
        let array = ctor(&mut children, exec_ctx)?;
        children.ensure_used(scheme)?;

        #[cfg(debug_assertions)]
        registered.check(scheme, &array)?;

        Ok(array)
    }

    /// Compresses or keeps a registered child, as the scheme requested unless the policy
    /// overrides it.
    fn resolve_child(
        &self,
        child: Child,
        requested: ChildDecision,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let decision = self.child_decision(child.kind, &child.array, requested, compress_ctx);
        match (decision, child.kind) {
            (ChildDecision::Keep, _) => Ok(child.array),
            (ChildDecision::Compress, ChildKind::Cascade { scheme, index }) => {
                self.compress_child(&child.array, compress_ctx, scheme, index, exec_ctx)
            }
            (ChildDecision::Compress, ChildKind::Validity) => Ok(self
                .compress_bitmap(&child.array, exec_ctx)?
                .unwrap_or(child.array)),
        }
    }

    /// Compresses the bitmap of an array-backed validity, unless the policy keeps it.
    ///
    /// A bitmap that compresses to a constant becomes [`Validity::AllValid`] or
    /// [`Validity::AllInvalid`], which store no bitmap.
    pub(super) fn resolve_validity(
        &self,
        validity: Validity,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Validity> {
        match validity {
            Validity::Array(bitmap) => {
                let bitmap = self.resolve_bitmap(bitmap, compress_ctx, exec_ctx)?;
                Ok(child_to_validity(Some(&bitmap), Nullability::Nullable))
            }
            validity => Ok(validity),
        }
    }

    /// Replaces a validity bitmap known to be all `true` with a constant, unless the policy keeps
    /// it, or the constant is no smaller.
    pub(super) fn resolve_all_valid_bitmap(
        &self,
        bitmap: ArrayRef,
        compress_ctx: &CompressorContext,
    ) -> ArrayRef {
        let decision = self.child_decision(
            ChildKind::Validity,
            &bitmap,
            ChildDecision::Compress,
            compress_ctx,
        );
        if decision == ChildDecision::Keep {
            return bitmap;
        }
        let constant = ConstantArray::new(true, bitmap.len()).into_array();
        if constant.nbytes() < bitmap.nbytes() {
            constant
        } else {
            bitmap
        }
    }

    /// Compresses a validity bitmap, unless the policy keeps it.
    pub(super) fn resolve_bitmap(
        &self,
        bitmap: ArrayRef,
        compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.resolve_child(
            Child::validity(bitmap),
            ChildDecision::Compress,
            compress_ctx,
            exec_ctx,
        )
    }

    /// Asks the policy, if any, what to do with a child.
    fn child_decision(
        &self,
        kind: ChildKind,
        array: &ArrayRef,
        requested: ChildDecision,
        ctx: &CompressorContext,
    ) -> ChildDecision {
        match &self.policy {
            Some(policy) => policy.decide_child(&ChildRequest {
                kind,
                array,
                requested,
                ctx,
            }),
            None => requested,
        }
    }
}

/// The debug check that every canonical array in a scheme's output went through the compressor.
#[cfg(debug_assertions)]
mod check {
    use vortex_array::ArrayRef;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_utils::aliases::hash_set::HashSet;

    use crate::scheme::SchemeId;

    /// The arrays the compressor handed to a constructor.
    pub(super) struct Registered {
        /// The arrays' addresses.
        addrs: HashSet<usize>,
        /// The arrays themselves, held so that a constructor dropping one cannot free its
        /// address for an unregistered array to reuse.
        _alive: Vec<ArrayRef>,
    }

    impl Registered {
        /// Records the compressed and kept children and the validity bitmap.
        pub(super) fn new(
            compressed: &[ArrayRef],
            kept: &[ArrayRef],
            validity: Option<&Validity>,
        ) -> Self {
            let bitmap = match validity {
                Some(Validity::Array(bitmap)) => Some(bitmap),
                _ => None,
            };
            let alive: Vec<ArrayRef> = compressed
                .iter()
                .chain(kept)
                .chain(bitmap)
                .cloned()
                .collect();
            Self {
                addrs: alive.iter().map(ArrayRef::addr).collect(),
                _alive: alive,
            }
        }

        /// Walks `array` and rejects a canonical array that was not registered.
        ///
        /// Registered arrays are not walked: they are the compressor's. Everything else was built
        /// by the constructor, which may build wrappers such as a `BitPacked` array inside a `FoR`
        /// array, but not canonical arrays.
        pub(super) fn check(&self, scheme: SchemeId, array: &ArrayRef) -> VortexResult<()> {
            // A registered root is a child the constructor returned as it is: the compressor's
            // output. A canonical root means the scheme gave up and handed back its input, which
            // the compressor's size check rejects; the compressor compresses its validity itself.
            if self.addrs.contains(&array.addr()) || array.is_canonical() {
                return Ok(());
            }
            for (slot, child) in array.slots().iter().enumerate() {
                if let Some(child) = child {
                    self.walk(scheme, array, slot, child)?;
                }
            }
            Ok(())
        }

        /// Checks the array in `slot` of `parent`, then its own slots.
        fn walk(
            &self,
            scheme: SchemeId,
            parent: &ArrayRef,
            slot: usize,
            array: &ArrayRef,
        ) -> VortexResult<()> {
            if self.addrs.contains(&array.addr()) {
                return Ok(());
            }
            if array.is_canonical() {
                vortex_bail!(
                    "scheme {scheme} left slot {slot} ({}) of {} uncompressed without \
                     registering it: add it to Encoded::compress, or to Encoded::keep to keep it \
                     as it is, or register the validity with Encoded::with_validity",
                    parent.slot_name(slot),
                    parent.encoding_id(),
                );
            }
            for (slot, child) in array.slots().iter().enumerate() {
                if let Some(child) = child {
                    self.walk(scheme, array, slot, child)?;
                }
            }
            Ok(())
        }
    }
}
