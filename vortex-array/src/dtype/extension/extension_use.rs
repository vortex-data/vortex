// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// How a function uses operands whose dtypes contain extension dtypes.
///
/// Extension types are opaque by default: the values of an extension type do not take the equality,
/// the order, or any other semantics of its storage dtype. So [`ScalarFnVTable::extension_use`] and
/// [`AggregateFnVTable::extension_use`] declare how a function uses the extension dtypes in the
/// operands of a call, including those nested in other dtypes, such as the elements of a list.
/// Vortex does not check these declarations yet.
///
/// [`Specific`](Self::Specific) and [`None`](Self::None) differ only in who handles the extension
/// operands of a call. Under `Specific`, the function handles them itself, and its own
/// `return_dtype` decides which extension dtypes it accepts. Under `None`, the function does not
/// handle them. This is the conservative default, and it is the right answer for a function that
/// accepts any dtype without knowing what its values mean, such as a hash of storage bytes.
///
/// [`ScalarFnVTable::extension_use`]: crate::scalar_fn::ScalarFnVTable::extension_use
/// [`AggregateFnVTable::extension_use`]: crate::aggregate_fn::AggregateFnVTable::extension_use
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub enum ExtensionUse {
    /// The function has no semantics for extension values.
    #[default]
    None,
    /// The function moves or selects values without reading them, such as `zip`, `fill_null`, or
    /// `first`.
    Structural,
    /// The function implements the semantics of the extension dtypes that it accepts, such as
    /// `replace_time_zone` for timestamps.
    Specific,
}
