// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::expr::Expression;
use crate::scalar_fn::ErasedOptions;
use crate::scalar_fn::ForeignScalarFnOptions;
use crate::scalar_fn::ForeignScalarFnVTable;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnPluginRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::byte_length::ByteLength;
use crate::scalar_fn::fns::case_when::CaseWhen;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::dynamic::DynamicComparison;
use crate::scalar_fn::fns::ext_storage::ExtStorage;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_nan::IsNan;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::PreparedSet;
use crate::scalar_fn::fns::list_length::ListLength;
use crate::scalar_fn::fns::list_sum::ListSum;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::merge::Merge;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::pack::Pack;
use crate::scalar_fn::fns::replace_time_zone::ReplaceTimeZone;
use crate::scalar_fn::fns::select::Select;
use crate::scalar_fn::fns::stat::StatFn;
use crate::scalar_fn::fns::variant_get::VariantGet;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::internal::row_count::RowCount;

/// Registry of scalar function vtables.
pub type ScalarFnRegistry = ArcSwapMap<Id, ScalarFnPluginRef>;

/// Session state for scalar function vtables and rewrite rules.
#[derive(Clone, Debug)]
pub struct ScalarFnSession {
    registry: ScalarFnRegistry,
}

impl ScalarFnSession {
    pub fn registry(&self) -> &ScalarFnRegistry {
        &self.registry
    }

    /// Register a scalar function vtable in the session, replacing any existing vtable with the same ID.
    pub fn register<V: ScalarFnVTable>(&self, vtable: V) {
        self.registry
            .insert(vtable.id(), Arc::new(vtable) as ScalarFnPluginRef);
    }
}

impl Default for ScalarFnSession {
    fn default() -> Self {
        let this = Self {
            registry: ScalarFnRegistry::default(),
        };

        // Register built-in expressions.
        this.register(Between);
        this.register(Binary);
        this.register(ByteLength);
        this.register(CaseWhen);
        this.register(Cast);
        this.register(DynamicComparison);
        this.register(ExtStorage);
        this.register(FillNull);
        this.register(GetItem);
        this.register(IsNan);
        this.register(IsNotNull);
        this.register(IsNull);
        this.register(Like);
        this.register(ListContains);
        this.register(PreparedSet);
        this.register(ListLength);
        this.register(ListSum);
        this.register(Literal);
        this.register(Mask);
        this.register(Merge);
        this.register(Not);
        this.register(Pack);
        this.register(ReplaceTimeZone);
        this.register(Select);
        this.register(StatFn);
        this.register(VariantGet);
        this.register(Zip);
        // Internal functions the scan replaces before execution still have to bind.
        this.register(RowCount);

        this
    }
}

impl SessionVar for ScalarFnSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// An argument a scalar function can be called on: an array or an expression.
pub trait ScalarFnArg: Sized {
    /// Applies `scalar_fn` to `args`.
    fn apply_scalar_fn(
        scalar_fn: ScalarFnRef,
        args: impl IntoIterator<Item = Self>,
    ) -> VortexResult<Self>;
}

impl ScalarFnArg for ArrayRef {
    fn apply_scalar_fn(
        scalar_fn: ScalarFnRef,
        args: impl IntoIterator<Item = Self>,
    ) -> VortexResult<Self> {
        Ok(ScalarFnArray::try_new(scalar_fn, args)?.into_array())
    }
}

impl ScalarFnArg for Expression {
    fn apply_scalar_fn(
        scalar_fn: ScalarFnRef,
        args: impl IntoIterator<Item = Self>,
    ) -> VortexResult<Self> {
        Expression::try_new(scalar_fn, args)
    }
}

/// Extension trait for accessing scalar function session data.
pub trait ScalarFnSessionExt: SessionExt {
    /// Returns the scalar function vtable registry.
    fn scalar_fns(&self) -> SessionGuard<'_, ScalarFnSession> {
        self.get::<ScalarFnSession>()
    }

    /// Builds the scalar function registered under `name` with `options`, which must be the
    /// function's own options type, such as [`EmptyOptions`](crate::scalar_fn::EmptyOptions) for a
    /// function without options.
    ///
    /// # Errors
    ///
    /// Returns an error if no function is registered under `name`, or if `options` is not its
    /// options type.
    fn scalar_fn(&self, name: &str, options: &dyn Any) -> VortexResult<ScalarFnRef> {
        #[expect(
            clippy::disallowed_methods,
            reason = "interning a dynamic function name"
        )]
        let id = ScalarFnId::new(name);
        self.scalar_fn_by_id(id, options)
    }

    /// Builds the scalar function registered under `id` with `options`, like
    /// [`scalar_fn`](Self::scalar_fn) but for an already interned id.
    ///
    /// # Errors
    ///
    /// Returns an error if no function is registered under `id`, or if `options` is not its
    /// options type.
    fn scalar_fn_by_id(&self, id: ScalarFnId, options: &dyn Any) -> VortexResult<ScalarFnRef> {
        self.scalar_fn_plugin(id)?.bind_any(options)
    }

    /// The plugin registered under `id`.
    ///
    /// # Errors
    ///
    /// Returns an error if no function is registered under `id`.
    fn scalar_fn_plugin(&self, id: ScalarFnId) -> VortexResult<ScalarFnPluginRef> {
        self.scalar_fns()
            .registry()
            .get(&id)
            .ok_or_else(|| vortex_err!("unknown scalar function: {id}"))
    }

    /// Builds the scalar function an [`Expression`] node calls: the one registered under `id`,
    /// with `options`.
    ///
    /// Options read for an id the session did not register, by
    /// [`Expression::from_proto`] with unknown ids allowed, give the foreign placeholder
    /// instead, which binds but cannot execute.
    ///
    /// # Errors
    ///
    /// Returns an error if no function is registered under `id`, or if `options` is not its
    /// options type.
    fn resolve_scalar_fn(
        &self,
        id: ScalarFnId,
        options: &ErasedOptions,
    ) -> VortexResult<ScalarFnRef> {
        if let Some(foreign) = options.downcast_ref::<ForeignScalarFnOptions>() {
            return Ok(ForeignScalarFnVTable::new(id).bind(foreign.clone()));
        }
        self.scalar_fn_by_id(id, options.as_any())
    }

    /// Calls the scalar function registered under `name` with `options` on `args`.
    ///
    /// Array arguments give the lazy result array, and expression arguments give the expression.
    ///
    /// # Errors
    ///
    /// Returns an error if no function is registered under `name`, if `options` is not its options
    /// type, or if `args` do not fit its signature.
    fn call<A: ScalarFnArg>(
        &self,
        name: &str,
        options: &dyn Any,
        args: impl IntoIterator<Item = A>,
    ) -> VortexResult<A> {
        A::apply_scalar_fn(self.scalar_fn(name, options)?, args)
    }
}
impl<S: SessionExt> ScalarFnSessionExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::ScalarFnSessionExt;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::not;
    use crate::expr::root;
    use crate::scalar_fn::EmptyOptions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn call_by_name() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let result = SESSION.call("vortex.is_null", &EmptyOptions, [array])?;
        assert_arrays_eq!(
            result,
            BoolArray::from_iter([false, true, false]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn call_with_options() -> VortexResult<()> {
        let target = DType::Primitive(PType::I64, Nullability::NonNullable);
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let result = SESSION.call("vortex.cast", &target, [array])?;
        assert_arrays_eq!(
            result,
            PrimitiveArray::from_iter([1i64, 2, 3]),
            &mut SESSION.create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn wrong_options_type() {
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let err = SESSION
            .call("vortex.cast", &EmptyOptions, [array])
            .unwrap_err();
        assert!(err.to_string().contains("takes options of type"));
    }

    #[test]
    fn call_on_expressions_matches_constructor() -> VortexResult<()> {
        assert_eq!(
            SESSION.call("vortex.not", &EmptyOptions, [root()])?,
            not(root())
        );
        Ok(())
    }

    #[test]
    fn unknown_function() {
        let err = SESSION
            .call("vortex.no_such_function", &EmptyOptions, [root()])
            .unwrap_err();
        assert!(err.to_string().contains("unknown scalar function"));
    }
}
