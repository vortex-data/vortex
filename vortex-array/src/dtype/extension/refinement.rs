// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Refinement extension types.
//!
//! A [`RefinementVTable`] defines a refinement type: an [`ExtVTable`] extension type whose logical
//! domain is a subset of some "source" type, carved out by a validation predicate. The source can
//! be either a canonical [`DType`] (via a [`PrimitiveRefinedSource<T>`] marker) or another
//! extension type (via an [`ExtRefinedSource<V>`] marker).
//!
//! The blanket [`impl<R: RefinementVTable> ExtVTable for R`] wires every refinement into the
//! existing [`DType::Extension`] / plugin / registry pipeline, so a [`RefinementVTable`] is *also*
//! an [`ExtVTable`] with no additional work.

use std::any::type_name;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::marker::PhantomData;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::array::ArrayRef;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::dtype::extension::ExtVTable;
use crate::executor::VortexSessionExecute;
use crate::scalar::ScalarValue;

/// A "source" type that a refinement is defined on top of.
///
/// The source determines what the refinement predicate observes: a canonical value (e.g. an `i32`
/// via [`PrimitiveRefinedSource<i32>`]) or another extension type's native value (e.g. a
/// [`uuid::Uuid`] via `ExtRefinedSource<Uuid>`).
///
/// Implementors provide a nullability-agnostic [`matches()`](Self::matches) predicate on the
/// storage [`DType`] and an [`unpack()`](Self::unpack) method that materializes the source's
/// native value from a storage [`ScalarValue`].
pub trait RefinedSource: 'static + Send + Sync {
    /// The native Rust value the refinement predicate receives.
    type Value<'a>;

    /// Returns `true` if `storage_dtype` matches this source. Nullability is ignored.
    fn matches(storage_dtype: &DType) -> bool;

    /// Unpack a storage scalar as this source's native value.
    ///
    /// # Precondition
    ///
    /// [`matches()`](Self::matches) has returned `true` for `storage_dtype`. Implementors may
    /// return an error if the precondition is violated, or may rely on it for safety.
    fn unpack<'a>(
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::Value<'a>>;
}

/// Marker selecting a canonical [`DType::Primitive`] as a refinement source.
///
/// The refinement's predicate observes a native Rust value of type `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimitiveRefinedSource<T: NativePType>(PhantomData<fn() -> T>);

impl<T: NativePType> RefinedSource for PrimitiveRefinedSource<T> {
    type Value<'a> = T;

    fn matches(storage_dtype: &DType) -> bool {
        matches!(storage_dtype, DType::Primitive(p, _) if *p == T::PTYPE)
    }

    fn unpack<'a>(_storage_dtype: &'a DType, storage_value: &'a ScalarValue) -> VortexResult<T> {
        match storage_value {
            ScalarValue::Primitive(pv) => pv.cast::<T>(),
            other => vortex_bail!(
                "PrimitiveRefinedSource<{}> expected ScalarValue::Primitive, got {:?}",
                T::PTYPE,
                other,
            ),
        }
    }
}

/// Marker selecting another extension type `V` as a refinement source.
///
/// The refinement's predicate observes `V`'s native value, produced by the inner
/// [`ExtVTable::unpack_native`]. Because unpacking goes through `V`, refinements *compose*: a
/// refinement over `V` transitively inherits every validation `V` performs.
///
/// `V` does not need to implement [`Default`]. Identifying the expected extension in error
/// messages uses the Rust type name, which works for any [`ExtVTable`] including those whose
/// [`ExtId`] is carried as runtime state (e.g. the internal `ForeignExtDType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtRefinedSource<V: ExtVTable>(PhantomData<fn() -> V>);

impl<V: ExtVTable> RefinedSource for ExtRefinedSource<V> {
    type Value<'a> = V::NativeValue<'a>;

    fn matches(storage_dtype: &DType) -> bool {
        let DType::Extension(ext) = storage_dtype else {
            return false;
        };
        ext.is::<V>()
    }

    fn unpack<'a>(
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<V::NativeValue<'a>> {
        let DType::Extension(ext_ref) = storage_dtype else {
            vortex_bail!(
                "ExtRefinedSource<{}> expected DType::Extension, got {}",
                type_name::<V>(),
                storage_dtype,
            );
        };
        let typed = ext_ref.as_typed::<V>().ok_or_else(|| {
            vortex_err!(
                "ExtRefinedSource<{}> got a differently-typed extension {}",
                type_name::<V>(),
                ext_ref.id(),
            )
        })?;
        V::unpack_native(typed, storage_value)
    }
}

/// Definition of a refinement extension type.
///
/// A refinement is an [`ExtVTable`] expressed as a predicate over a [`RefinedSource`]. The
/// refinement author declares what the source is and what the predicate (plus optional narrowing)
/// looks like; every other [`ExtVTable`] responsibility is wired up by the blanket
/// [`impl<R: RefinementVTable> ExtVTable for R`].
///
/// To implement a refinement, define:
///
/// - [`Source`](Self::Source) names the [`RefinedSource`] being restricted.
/// - [`Metadata`](Self::Metadata) carries refinement-specific state, often
///   [`EmptyMetadata`](crate::extension::EmptyMetadata) when the predicate has no runtime
///   parameters.
/// - [`NativeValue<'a>`](Self::NativeValue) is the refined, narrowed Rust value. Set it to
///   `<Self::Source as RefinedSource>::Value<'a>` when the refinement only restricts the domain
///   and does not narrow the Rust type.
/// - [`id()`](Self::id) returns the extension ID.
/// - [`refine_scalar()`](Self::refine_scalar) implements the per-value predicate and narrowing.
/// - [`serialize_metadata()`](Self::serialize_metadata) and
///   [`deserialize_metadata()`](Self::deserialize_metadata) persist
///   [`Metadata`](Self::Metadata).
///
/// [`validate_array()`](Self::validate_array) has a default scalar-iteration fallback; override it
/// for throughput.
///
/// [`id()`](Self::id), [`serialize_metadata()`](Self::serialize_metadata), and
/// [`deserialize_metadata()`](Self::deserialize_metadata) take `&self` so that refinement vtables
/// carrying runtime state (matching [`ExtVTable`]) can participate. Stateless refinements can
/// ignore `self` in their implementations.
pub trait RefinementVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// The type being refined. This is either a [`PrimitiveRefinedSource<T>`] marker or an
    /// [`ExtRefinedSource<V>`] marker over another extension type.
    type Source: RefinedSource;

    /// Refinement-specific metadata.
    ///
    /// Use [`EmptyMetadata`](crate::extension::EmptyMetadata) for predicates with no runtime
    /// parameters.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// The narrowed native Rust value produced by [`refine_scalar()`](Self::refine_scalar).
    ///
    /// Set to `<Self::Source as RefinedSource>::Value<'a>` if the refinement only restricts the
    /// source's domain. Set to a distinct type (e.g. [`std::num::NonZeroI32`]) to expose a
    /// Rust-level narrowed value.
    type NativeValue<'a>: Display;

    /// Returns the extension ID for this refinement type.
    fn id(&self) -> ExtId;

    /// Validate that `source_value` satisfies the predicate and narrow it into
    /// [`NativeValue`](Self::NativeValue).
    ///
    /// # Errors
    ///
    /// Returns an error if the value does not satisfy the refinement predicate.
    fn refine_scalar<'a>(
        metadata: &'a Self::Metadata,
        source_value: <Self::Source as RefinedSource>::Value<'a>,
    ) -> VortexResult<Self::NativeValue<'a>>;

    /// Validate that every value in `source_array` satisfies the refinement predicate.
    ///
    /// The default implementation iterates scalars via [`ArrayRef::execute_scalar`] and calls
    /// [`refine_scalar()`](Self::refine_scalar) on each one. This is correct but slow: every
    /// iteration pays virtual-dispatch and scalar-materialization overhead.
    ///
    /// Refinement authors that care about throughput should override this with a vectorized
    /// implementation that downcasts `source_array` to a concrete encoding (for example via
    /// `source_array.as_opt::<Primitive>()`) and runs a SIMD-friendly predicate. Delegate back to
    /// [`refine_array_scalar_default`] for any fallback arms that do not specialize.
    fn validate_array(metadata: &Self::Metadata, source_array: &ArrayRef) -> VortexResult<()> {
        refine_array_scalar_default::<Self>(metadata, source_array)
    }

    /// Serialize the refinement-specific metadata into a byte vector.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Deserialize the refinement-specific metadata from a byte slice.
    fn deserialize_metadata(&self, bytes: &[u8]) -> VortexResult<Self::Metadata>;
}

/// Reference scalar-iteration fallback for [`RefinementVTable::validate_array`].
///
/// Exposed so that partial overrides (e.g. a vectorized fast path plus a fallback arm for rarer
/// storage encodings) can delegate back to the default behaviour without re-deriving the loop.
pub fn refine_array_scalar_default<R: RefinementVTable>(
    metadata: &R::Metadata,
    source_array: &ArrayRef,
) -> VortexResult<()> {
    let source_dtype = source_array.dtype();
    let mut ctx = crate::LEGACY_SESSION.create_execution_ctx();
    for i in 0..source_array.len() {
        let scalar = source_array.execute_scalar(i, &mut ctx)?;
        let Some(storage_value) = scalar.value() else {
            continue;
        };
        let source_value = R::Source::unpack(source_dtype, storage_value)?;
        R::refine_scalar(metadata, source_value)?;
    }
    Ok(())
}

impl<R: RefinementVTable> ExtVTable for R {
    type Metadata = R::Metadata;
    type NativeValue<'a> = R::NativeValue<'a>;

    fn id(&self) -> ExtId {
        R::id(self)
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        R::serialize_metadata(self, metadata)
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        R::deserialize_metadata(self, metadata)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        if !R::Source::matches(ext_dtype.storage_dtype()) {
            vortex_bail!(
                "refinement {} got incompatible storage dtype {}",
                ext_dtype.vtable().id(),
                ext_dtype.storage_dtype(),
            );
        }
        Ok(())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>> {
        let source_value = R::Source::unpack(ext_dtype.storage_dtype(), storage_value)?;
        R::refine_scalar(ext_dtype.metadata(), source_value)
    }
}
