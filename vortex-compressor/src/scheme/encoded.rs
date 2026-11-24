// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a scheme hands back to the compressor: an encoding whose children are not compressed yet.
//!
//! A [`Scheme`] does not compress its own children. [`Scheme::encode`] returns an [`Encoded`]: the
//! children to compress, the children to keep as they are, and a constructor that builds the
//! encoded array from them. The compressor compresses the children and then calls the
//! constructor.
//!
//! The point is that leaving a child uncompressed takes a decision. A scheme never holds the
//! compressor while it encodes, and a constructor captures only data, never child arrays: every
//! child reaches it through the compressor, compressed, unless the scheme lists it in
//! [`Encoded::keep`]. In debug builds the compressor walks each built array and rejects any
//! canonical array that was neither compressed nor kept. Validity follows the same rule, so it is
//! compressed once, after the scheme has run, at whatever level it ends up on.
//!
//! The compressor also drives all recursion. `encode` returns before any child is compressed, so a
//! scheme's frame is never on the stack while its children are compressed, and the compressor has
//! the last word on every child: see [`ChildPolicy`].
//!
//! [`Scheme`]: crate::scheme::Scheme
//! [`Scheme::encode`]: crate::scheme::Scheme::encode

use std::fmt;
use std::fmt::Debug;

use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::NullArray;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::compressor::leaf_validity_slot;
use crate::scheme::CompressorContext;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;

/// How the compressor treats a child.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChildKind {
    /// A child of a scheme, compressed one cascade level below it.
    ///
    /// The pair is recorded in the cascade history, so it is what exclusion rules match on.
    Cascade {
        /// The scheme the child belongs to.
        scheme: SchemeId,
        /// The child's index within that scheme.
        index: usize,
    },
    /// A validity bitmap, compressed with the boolean schemes from a fresh root context, and kept
    /// only when it gets smaller.
    Validity,
}

/// A child array a scheme registers with the compressor.
#[derive(Debug, Clone)]
pub struct Child {
    /// How the child is compressed.
    pub kind: ChildKind,
    /// The child array.
    pub array: ArrayRef,
}

impl Child {
    /// Child `index` of `scheme`.
    pub fn new(scheme: &(impl Scheme + ?Sized), index: usize, array: ArrayRef) -> Self {
        Self {
            kind: ChildKind::Cascade {
                scheme: scheme.id(),
                index,
            },
            array,
        }
    }

    /// A validity bitmap.
    pub fn validity(bitmap: ArrayRef) -> Self {
        Self {
            kind: ChildKind::Validity,
            array: bitmap,
        }
    }
}

/// The children handed to an [`Encoded`] constructor, in the order they were registered.
#[derive(Debug, Default)]
pub struct EncodedChildren {
    /// The children from [`Encoded::compress`], compressed.
    compressed: Vec<ArrayRef>,
    /// The children from [`Encoded::keep`], as given.
    kept: Vec<ArrayRef>,
    /// The validity from [`Encoded::validity`], with its bitmap compressed.
    validity: Option<Validity>,
}

impl EncodedChildren {
    /// Takes the next `N` compressed children.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than `N` remain.
    pub fn compressed<const N: usize>(&mut self) -> VortexResult<[ArrayRef; N]> {
        take(&mut self.compressed, "compressed")
    }

    /// Takes the next `N` kept children.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than `N` remain.
    pub fn kept<const N: usize>(&mut self) -> VortexResult<[ArrayRef; N]> {
        take(&mut self.kept, "kept")
    }

    /// Takes the validity.
    ///
    /// # Errors
    ///
    /// Returns an error if the [`Encoded`] registered no validity, or it was already taken.
    pub fn validity(&mut self) -> VortexResult<Validity> {
        self.validity
            .take()
            .ok_or_else(|| vortex_err!("Encoded registered no validity"))
    }
}

/// Takes the first `N` arrays from `arrays`.
fn take<const N: usize>(arrays: &mut Vec<ArrayRef>, what: &str) -> VortexResult<[ArrayRef; N]> {
    vortex_ensure!(
        arrays.len() >= N,
        "asked for {N} {what} children, {} remain",
        arrays.len()
    );
    let rest = arrays.split_off(N);
    let front = std::mem::replace(arrays, rest);
    front
        .try_into()
        .map_err(|_| vortex_err!("asked for {N} {what} children"))
}

/// Builds an encoded array from its children once the compressor has compressed them.
///
/// A constructor captures only data, never child arrays: every child reaches it through the
/// [`EncodedChildren`], so nothing can skip the compressor.
pub type EncodedCtor =
    Box<dyn FnOnce(&mut EncodedChildren, &mut ExecutionCtx) -> VortexResult<ArrayRef>>;

/// A scheme's encoding, with its children not compressed yet.
///
/// The compressor compresses every child in [`compress`](Self::compress) and the bitmap in
/// [`validity`](Self::validity), passes through every child in [`keep`](Self::keep), and calls
/// [`ctor`](Self::ctor) with the results. A [`ChildPolicy`] on the compressor can move a child
/// between the two lists.
pub struct Encoded {
    /// Children the compressor compresses.
    pub compress: Vec<Child>,
    /// Children the scheme keeps as they are: the scheme's opt-out from compressing them.
    pub keep: Vec<Child>,
    /// The encoding's own validity, if it stores one. An array-backed bitmap is compressed.
    pub validity: Option<Validity>,
    /// Builds the encoded array from the compressed and kept children.
    pub ctor: EncodedCtor,
    /// Set by the default [`Scheme::encode`]: the scheme implements only
    /// [`Scheme::compress`], which the compressor calls instead.
    uses_compress: bool,
}

impl Debug for Encoded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoded")
            .field("compress", &self.compress)
            .field("keep", &self.keep)
            .field("validity", &self.validity)
            .finish_non_exhaustive()
    }
}

impl Encoded {
    /// An encoding built by `ctor` from the children in `compress` and `keep`.
    pub fn new(
        compress: Vec<Child>,
        keep: Vec<Child>,
        ctor: impl FnOnce(&mut EncodedChildren, &mut ExecutionCtx) -> VortexResult<ArrayRef> + 'static,
    ) -> Self {
        Self {
            compress,
            keep,
            validity: None,
            ctor: Box::new(ctor),
            uses_compress: false,
        }
    }

    /// An array with no children left to compress.
    ///
    /// The constructor holds `array`, so it must not hold child arrays that should go through the
    /// compressor: the debug check rejects any canonical array below it. A scheme handing back its
    /// input unchanged, for the size check to reject, may use it too.
    pub fn leaf(array: ArrayRef) -> Self {
        Self::new(Vec::new(), Vec::new(), move |_, _| Ok(array))
    }

    /// An encoding that stores the whole of `leaf`, a bool, primitive, decimal or `VarBinView`
    /// array, validity included, as a general-purpose compressor such as Pco does.
    ///
    /// The leaf's validity bitmap is registered, so it is compressed, and `encode` receives the
    /// leaf holding the compressed bitmap. A leaf without a bitmap, and any other array, is handed
    /// to `encode` as it is.
    ///
    /// `encode` may return the leaf, or an array that stores the leaf's validity, but not an array
    /// that holds the leaf as a child: the debug check rejects the canonical leaf there, since it
    /// was not registered.
    ///
    /// # Errors
    ///
    /// Returns an error if the leaf's validity cannot be read.
    pub fn of_leaf(
        leaf: ArrayRef,
        encode: impl FnOnce(ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + 'static,
    ) -> VortexResult<Self> {
        let validity = match leaf_validity_slot(&leaf) {
            Some(slot) => Some((slot, leaf.validity()?)),
            None => None,
        };
        let Some((slot, validity @ Validity::Array(_))) = validity else {
            // No bitmap to compress.
            return Ok(Self::new(Vec::new(), Vec::new(), move |_, exec_ctx| {
                encode(leaf, exec_ctx)
            }));
        };
        Ok(
            Self::new(Vec::new(), Vec::new(), move |children, exec_ctx| {
                let validity = children.validity()?;
                let mut slots: ArraySlots = leaf.slots().iter().cloned().collect();
                // A bitmap that compressed to a constant comes back as `AllValid` or `AllInvalid`.
                // Storing it as a constant keeps the slot present, as `with_slots` requires.
                slots[slot] = Some(validity.to_array(leaf.len()));
                // SAFETY: the validity bitmap is replaced with its compressed form, or kept, so it
                // holds the same values. The leaf's values, dtype, and other slots are unchanged.
                let leaf = unsafe { leaf.with_slots(slots) }?;
                encode(leaf, exec_ctx)
            })
            .with_validity(validity),
        )
    }

    /// Wraps this encoding in an outer one, for a scheme that encodes its input with another
    /// scheme's encoding and then wraps the result, such as frame-of-reference around
    /// bit-packing.
    ///
    /// The outer children in `compress` and `keep` are registered after this encoding's own. The
    /// constructor runs this encoding's constructor first, which must take all of its own
    /// children, then calls `wrap` with its output and the children that remain: the outer ones.
    ///
    /// The outer encoding registers no validity of its own: the result keeps this encoding's,
    /// which [`with_validity`](Self::with_validity) on the result would replace.
    pub fn wrap(
        self,
        compress: Vec<Child>,
        keep: Vec<Child>,
        wrap: impl FnOnce(ArrayRef, &mut EncodedChildren, &mut ExecutionCtx) -> VortexResult<ArrayRef>
        + 'static,
    ) -> Self {
        let Self {
            compress: mut inner_compress,
            keep: mut inner_keep,
            validity,
            ctor,
            uses_compress,
        } = self;
        inner_compress.extend(compress);
        inner_keep.extend(keep);
        Self {
            compress: inner_compress,
            keep: inner_keep,
            validity,
            ctor: Box::new(move |children, exec_ctx| {
                let inner = ctor(children, exec_ctx)?;
                wrap(inner, children, exec_ctx)
            }),
            uses_compress,
        }
    }

    /// Registers the encoding's validity. Its bitmap, if any, is compressed.
    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = Some(validity);
        self
    }

    /// What the default [`Scheme::encode`] returns: the scheme implements only
    /// [`Scheme::compress`], which the compressor calls instead.
    pub(crate) fn uses_compress() -> Self {
        Self {
            uses_compress: true,
            ..Self::leaf(NullArray::new(0).into_array())
        }
    }

    /// Whether the scheme implements only [`Scheme::compress`].
    pub(crate) fn is_uses_compress(&self) -> bool {
        self.uses_compress
    }

    /// Splits the encoding into its children and constructor.
    pub(crate) fn into_parts(self) -> (Vec<Child>, Vec<Child>, Option<Validity>, EncodedCtor) {
        (self.compress, self.keep, self.validity, self.ctor)
    }
}

impl EncodedChildren {
    /// Checks that a constructor used every child `scheme` registered.
    pub(crate) fn ensure_used(&self, scheme: SchemeId) -> VortexResult<()> {
        vortex_ensure!(
            self.compressed.is_empty() && self.kept.is_empty() && self.validity.is_none(),
            "the constructor of scheme {scheme} left {} compressed and {} kept children{} unused",
            self.compressed.len(),
            self.kept.len(),
            if self.validity.is_some() {
                " and its validity"
            } else {
                ""
            },
        );
        Ok(())
    }

    /// The children the compressor produced, ready for the constructor.
    pub(crate) fn from_parts(
        compressed: Vec<ArrayRef>,
        kept: Vec<ArrayRef>,
        validity: Option<Validity>,
    ) -> Self {
        Self {
            compressed,
            kept,
            validity,
        }
    }
}

/// Whether a child is compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChildDecision {
    /// Compress the child.
    Compress,
    /// Keep the child as it is.
    Keep,
}

/// A child the compressor is about to compress or keep, as seen by a [`ChildPolicy`].
#[derive(Debug)]
pub struct ChildRequest<'a> {
    /// How the child would be compressed.
    pub kind: ChildKind,
    /// The child array.
    pub array: &'a ArrayRef,
    /// What the scheme asked for: [`ChildDecision::Compress`] for a child in
    /// [`Encoded::compress`] or a validity bitmap, [`ChildDecision::Keep`] for a child in
    /// [`Encoded::keep`].
    pub requested: ChildDecision,
    /// The context of the scheme that registered the child.
    pub ctx: &'a CompressorContext,
}

/// Lets a compressor override a scheme's decision about a child.
///
/// The compressor asks the policy about every registered child and validity bitmap, and about
/// the validity of leaves it compresses itself. A policy can keep a child a scheme asked to
/// compress, or compress a child a scheme asked to keep. Compressing a kept child is the policy's
/// responsibility: the scheme may rely on it staying canonical.
///
/// The cascade budget still applies: once a chain has used up [`MAX_CASCADE`] levels, its children
/// are kept whatever the policy decides.
///
/// [`MAX_CASCADE`]: crate::scheme::MAX_CASCADE
pub trait ChildPolicy: Debug + Send + Sync {
    /// Decides what happens to the child.
    fn decide(&self, request: &ChildRequest<'_>) -> ChildDecision;
}
