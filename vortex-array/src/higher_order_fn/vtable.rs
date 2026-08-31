// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::higher_order_fn::HigherOrderFnId;
use crate::higher_order_fn::HigherOrderFnRef;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;

/// The identity and options contract for a higher-order function.
///
/// A higher-order function accepts ordinary arguments and lambda arguments. This core vtable
/// describes their arities and provides type-erased option storage and serialization. Expression
/// binding and execution are layered on top separately.
pub trait HigherOrderFnVTable: 'static + Sized + Clone + Send + Sync {
    /// Per-call options for this higher-order function.
    type Options: 'static + Send + Sync + Clone + Debug + Display + PartialEq + Eq + Hash;

    /// The globally unique identifier for this higher-order function.
    fn id(&self) -> HigherOrderFnId;

    /// Serialize the per-call options.
    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        _ = options;
        Ok(None)
    }

    /// Deserialize per-call options.
    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_bail!("higher-order function {} is not deserializable", self.id());
    }

    /// The arity of the ordinary arguments.
    fn arity(&self, options: &Self::Options) -> Arity;

    /// The number of lambda arguments.
    fn lambda_arity(&self, options: &Self::Options) -> usize;

    /// The name of an ordinary argument.
    fn child_name(&self, options: &Self::Options, child_idx: usize) -> ChildName;
}

/// Empty higher-order-function options.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EmptyOptions;

impl Display for EmptyOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("")
    }
}

/// Factory methods for higher-order function vtables.
pub trait HigherOrderFnVTableExt: HigherOrderFnVTable {
    /// Bind this vtable with the given options into a [`HigherOrderFnRef`].
    fn bind(&self, options: Self::Options) -> HigherOrderFnRef {
        HigherOrderFnRef::new(self.clone(), options)
    }
}

impl<V: HigherOrderFnVTable> HigherOrderFnVTableExt for V {}

#[cfg(test)]
mod tests {
    use std::fmt::Display;
    use std::fmt::Formatter;

    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::registry::CachedId;

    use super::*;
    use crate::array_session;
    use crate::higher_order_fn::session::HigherOrderFnSessionExt;

    #[derive(Clone, Debug)]
    struct TestHigherOrderFn;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct TestOptions(&'static str);

    impl Display for TestOptions {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl HigherOrderFnVTable for TestHigherOrderFn {
        type Options = TestOptions;

        fn id(&self) -> HigherOrderFnId {
            static ID: CachedId = CachedId::new("vortex.test.higher_order");
            *ID
        }

        fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
            Ok(Some(options.0.as_bytes().to_vec()))
        }

        fn deserialize(
            &self,
            metadata: &[u8],
            _session: &VortexSession,
        ) -> VortexResult<Self::Options> {
            match metadata {
                b"strategy=checked" => Ok(TestOptions("strategy=checked")),
                _ => vortex_bail!("unknown test higher-order function options"),
            }
        }

        fn arity(&self, _options: &Self::Options) -> Arity {
            Arity::Exact(1)
        }

        fn lambda_arity(&self, _options: &Self::Options) -> usize {
            1
        }

        fn child_name(&self, _options: &Self::Options, _child_idx: usize) -> ChildName {
            ChildName::from("input")
        }
    }

    #[test]
    fn bound_function_exposes_vtable_metadata_and_options() -> VortexResult<()> {
        let function = TestHigherOrderFn.bind(TestOptions("strategy=checked"));

        assert!(function.is::<TestHigherOrderFn>());
        assert_eq!(
            function.as_::<TestHigherOrderFn>(),
            &TestOptions("strategy=checked")
        );
        assert_eq!(function.arity(), Arity::Exact(1));
        assert_eq!(function.lambda_arity(), 1);
        assert_eq!(function.child_name(0), ChildName::from("input"));
        assert_eq!(function.serialize()?, Some(b"strategy=checked".to_vec()));
        assert_eq!(
            function.to_string(),
            "vortex.test.higher_order(strategy=checked)"
        );
        assert_ne!(
            function,
            TestHigherOrderFn.bind(TestOptions("strategy=unchecked"))
        );
        Ok(())
    }

    #[test]
    fn session_plugin_deserializes_bound_options() -> VortexResult<()> {
        let session = array_session();
        session.higher_order_fns().register(TestHigherOrderFn);
        let plugin = session
            .higher_order_fns()
            .registry()
            .get(&HigherOrderFnVTable::id(&TestHigherOrderFn))
            .vortex_expect("test higher-order function is registered");

        let function = plugin.deserialize(b"strategy=checked", &session)?;
        assert_eq!(
            function.as_::<TestHigherOrderFn>(),
            &TestOptions("strategy=checked")
        );
        Ok(())
    }
}
