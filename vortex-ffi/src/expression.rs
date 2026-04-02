// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(non_camel_case_types)]

use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr;
use std::slice;
use std::sync::Arc;

use vortex::dtype::FieldName;
use vortex::error::VortexExpect;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::expr::select;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;

use crate::to_field_names;

// Expressions are Arc'ed inside
crate::box_wrapper!(
    /// A node in a Vortex expression tree.
    ///
    /// Expressions represent scalar computations that can be performed on
    /// data. Each expression consists of an encoding (vtable), heap-allocated
    /// metadata, and child expressions.
    ///
    /// Unless stated explicitly, all expressions returned are owned and must
    /// be freed by the caller.
    /// Unless stated explicitly, if an operation on const vx_expression* is
    /// passed NULL, NULL is returned.
    /// Operations on expressions don't take ownership of input values, and so
    /// input values must be freed by the caller.
    Expression,
    vx_expression);

/// Create a root expression. A root expression, applied to an array in
/// vx_array_apply, takes the array itself as opposed to functions like
/// vx_expression_column or vx_expression_select which take the array's parts.
///
/// Example:
///
/// const vx_array* array = ...;
/// vx_expression* root = vx_expression_root();
/// const vx_error* error = NULL;
/// vx_array* applied_array = vx_array_apply(array, root, &error);
/// // array and applied_array are identical
/// vx_array_free(applied_array);
/// vx_expression_free(root);
/// vx_array_free(array);
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_root() -> *mut vx_expression {
    vx_expression::new(Box::new(root()))
}

/// Create an expression that selects (includes) specific fields from a child
/// expression. Child expression must have a DTYPE_STRUCT dtype. Errors in
/// vx_array_apply if the child expression doesn't have a specified field.
///
/// Example:
///
/// vx_expression* root = vx_expression_root();
/// const char* names[] = {"name", "age"};
/// vx_expression* select = vx_expression_select(names, 2, root);
/// vx_expression_free(select);
/// vx_expression_free(root);
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_select(
    names: *const *const c_char,
    len: usize,
    child: *const vx_expression,
) -> *mut vx_expression {
    if child.is_null() {
        return ptr::null_mut();
    }
    let names =
        unsafe { to_field_names(names, len) }.vortex_expect("converting names to field names");
    let expr = select(names, vx_expression::as_ref(child).clone());
    vx_expression::new(Box::new(expr))
}

/// Create an AND expression for multiple child expressions.
/// If there are no input expressions, returns NULL
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_and(
    expressions: *const *const vx_expression,
    len: usize,
) -> *mut vx_expression {
    if expressions.is_null() {
        return ptr::null_mut();
    }
    let slice = unsafe { slice::from_raw_parts(expressions, len) };
    match and_collect(slice.iter().map(|x| vx_expression::as_ref(*x).clone())) {
        Some(expr) => vx_expression::new(expr.into()),
        None => ptr::null_mut(),
    }
}

/// Create an OR disjunction expression for multiple child expressions.
/// If there are no input expressions, returns NULL;
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_or(
    expressions: *const *const vx_expression,
    len: usize,
) -> *mut vx_expression {
    if expressions.is_null() {
        return ptr::null_mut();
    }
    let slice = unsafe { slice::from_raw_parts(expressions, len) };
    match or_collect(slice.iter().map(|x| vx_expression::as_ref(*x).clone())) {
        Some(expr) => vx_expression::new(expr.into()),
        None => ptr::null_mut(),
    }
}

/// Equalities, inequalities, and boolean operations over possibly null values.
/// For most operations, if either side is null, the result is null.
/// VX_OPERATOR_KLEENE_AND, VX_OPERATOR_KLEENE_OR obey Kleene (three-valued)
/// logic
#[repr(C)]
pub enum vx_binary_operator {
    /// Expressions are equal.
    VX_OPERATOR_EQ = 0,
    /// Expressions are not equal.
    VX_OPERATOR_NOT_EQ = 1,
    /// Expression is greater than another
    VX_OPERATOR_GT = 2,
    /// Expression is greater or equal to another
    VX_OPERATOR_GTE = 3,
    /// Expression is less than another
    VX_OPERATOR_LT = 4,
    /// Expression is less or equal to another
    VX_OPERATOR_LTE = 5,
    /// Boolean AND /\.
    VX_OPERATOR_KLEENE_AND = 6,
    /// Boolean OR \/.
    VX_OPERATOR_KLEENE_OR = 7,
    /// The sum of the arguments.
    /// Errors at runtime if the sum would overflow or underflow.
    VX_OPERATOR_ADD = 8,
    /// The difference between the arguments.
    /// Errors at runtime if the sum would overflow or underflow.
    /// The result is null at any index where either input is null.
    VX_OPERATOR_SUB = 9,
    /// Multiply two numbers
    VX_OPERATOR_MUL = 10,
    /// Divide the left side by the right side
    VX_OPERATOR_DIV = 11,
}

impl From<vx_binary_operator> for Operator {
    fn from(operator: vx_binary_operator) -> Self {
        match operator {
            vx_binary_operator::VX_OPERATOR_EQ => Operator::Eq,
            vx_binary_operator::VX_OPERATOR_NOT_EQ => Operator::NotEq,
            vx_binary_operator::VX_OPERATOR_GT => Operator::Gt,
            vx_binary_operator::VX_OPERATOR_GTE => Operator::Gte,
            vx_binary_operator::VX_OPERATOR_LT => Operator::Lt,
            vx_binary_operator::VX_OPERATOR_LTE => Operator::Lte,
            vx_binary_operator::VX_OPERATOR_KLEENE_AND => Operator::And,
            vx_binary_operator::VX_OPERATOR_KLEENE_OR => Operator::Or,
            vx_binary_operator::VX_OPERATOR_ADD => Operator::Add,
            vx_binary_operator::VX_OPERATOR_SUB => Operator::Sub,
            vx_binary_operator::VX_OPERATOR_MUL => Operator::Mul,
            vx_binary_operator::VX_OPERATOR_DIV => Operator::Div,
        }
    }
}

/// Create a binary expression for two expressions of form lhs OP rhs.
/// If either input is NULL, returns NULL.
///
/// Example for a binary sum:
///
/// vx_expression* age = vx_expression_column("age");
/// vx_expression* height = vx_expression_column("height");
/// vx_expression* sum = vx_expression_binary(VX_OPERATOR_ADD, age, height);
/// vx_expression_free(sum);
/// vx_expression_free(height);
/// vx_expression_free(age);
///
/// Example for a binary equality function:
///
/// vx_expression* vx_expression_eq(
///     const vx_expression* lhs,
///     const vx_expression* rhs
/// ) {
///     return vx_expression_binary(VX_OPERATOR_EQ, lhs, rhs);
/// }
///
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_binary(
    operator: vx_binary_operator,
    lhs: *const vx_expression,
    rhs: *const vx_expression,
) -> *mut vx_expression {
    if lhs.is_null() {
        return ptr::null_mut();
    }
    if rhs.is_null() {
        return ptr::null_mut();
    }
    let lhs = vx_expression::as_ref(lhs).clone();
    let rhs = vx_expression::as_ref(rhs).clone();
    vx_expression::new(Box::new(Binary.new_expr(operator.into(), [lhs, rhs])))
}

/// Create a logical NOT of the child expression.
///
/// Returns the logical negation of the input boolean expression.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_not(child: *const vx_expression) -> *const vx_expression {
    if child.is_null() {
        return child;
    }
    vx_expression::new(not(vx_expression::as_ref(child).clone()).into())
}

/// Create an expression that checks for null values.
///
/// Returns a boolean array indicating which positions contain null values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_is_null(child: *const vx_expression) -> *mut vx_expression {
    if child.is_null() {
        return ptr::null_mut();
    }
    vx_expression::new(is_null(vx_expression::as_ref(child).clone()).into())
}

/// Create an expression that extracts a named field from a struct expression.
/// Child expression must have a DTYPE_STRUCT dtype.
/// Errors in vx_array_apply if the root array doesn't have a specified field.
///
/// Accesses the specified field from the result of the child expression.
/// Equivalent to select(&item, 1, child).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_get_item(
    item: *const c_char,
    child: *const vx_expression,
) -> *mut vx_expression {
    if child.is_null() {
        return ptr::null_mut();
    }
    if item.is_null() {
        return ptr::null_mut();
    }
    #[expect(clippy::expect_used)]
    let item = unsafe {
        CStr::from_ptr(item)
            .to_str()
            .expect("converting pointer to str")
    };
    let item: Arc<str> = Arc::from(item);
    let item: FieldName = item.into();
    vx_expression::new(get_item(item, vx_expression::as_ref(child).clone()).into())
}

/// Create an expression that checks if a value is contained in a list.
///
/// Returns a boolean array indicating whether the value appears in each list.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_list_contains(
    list: *const vx_expression,
    value: *const vx_expression,
) -> *mut vx_expression {
    if list.is_null() {
        return ptr::null_mut();
    }
    if value.is_null() {
        return ptr::null_mut();
    }
    let list = vx_expression::as_ref(list).clone();
    let value = vx_expression::as_ref(value).clone();
    vx_expression::new(Box::new(list_contains(list, value)))
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex::array::IntoArray;
    use vortex::array::ToCanonical;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::ListArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::expr::lit;

    use crate::array::vx_array;
    use crate::array::vx_array_apply;
    use crate::array::vx_array_free;
    use crate::error::vx_error_free;
    use crate::expression::vx_binary_operator;
    use crate::expression::vx_expression;
    use crate::expression::vx_expression_and;
    use crate::expression::vx_expression_binary;
    use crate::expression::vx_expression_free;
    use crate::expression::vx_expression_get_item;
    use crate::expression::vx_expression_list_contains;
    use crate::expression::vx_expression_or;
    use crate::expression::vx_expression_root;
    use crate::expression::vx_expression_select;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_root() {
        unsafe {
            let root = vx_expression_root();
            vx_expression_free(root);
        }
    }

    fn struct_array() -> (StructArray, VarBinViewArray, PrimitiveArray) {
        let names_array = VarBinViewArray::from_iter_str(["Alice", "Bob", "Charlie"]);
        let ages_buffer = buffer![30u8, 25u8, 35u8];
        let ages_array = PrimitiveArray::new(ages_buffer, Validity::NonNullable);
        let fields = vec![
            names_array.clone().into_array(),
            ages_array.clone().into_array(),
        ];
        let names = ["name", "age"].into();
        let struct_array = StructArray::try_new(names, fields, 3, Validity::NonNullable);
        (struct_array.unwrap(), names_array, ages_array)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_get_item() {
        let (array, names_array, ages_array) = struct_array();
        unsafe {
            let root = vx_expression_root();
            let column = vx_expression_get_item(c"age".as_ptr(), root);
            assert_ne!(column, ptr::null_mut());

            let array = vx_array::new(array.into_array());
            let mut error = ptr::null_mut();

            let applied_array = vx_array_apply(array, column, &raw mut error);
            assert!(!applied_array.is_null());
            assert!(error.is_null());
            {
                let applied_array = vx_array::as_ref(applied_array);
                let expected: Buffer<u8> = ages_array.to_buffer();
                assert_eq!(applied_array.to_primitive().to_buffer(), expected);
            }
            vx_array_free(applied_array);

            vx_expression_free(column);

            let column = vx_expression_get_item(c"ololo".as_ptr(), root);
            assert_ne!(column, ptr::null_mut());

            let applied_array = vx_array_apply(array, column, &raw mut error);
            assert!(applied_array.is_null());
            assert!(!error.is_null());
            vx_error_free(error);

            let names_array_vx = vx_array::new(names_array.into_array());
            let applied_array = vx_array_apply(names_array_vx, column, &raw mut error);
            assert!(applied_array.is_null());
            assert!(!error.is_null());
            vx_error_free(error);
            vx_array_free(names_array_vx);

            vx_expression_free(column);

            vx_array_free(array);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_select() {
        let (array, ..) = struct_array();
        unsafe {
            let root = vx_expression_root();

            let array = vx_array::new(array.into_array());

            let columns = [c"name".as_ptr(), c"age".as_ptr()];
            let column = vx_expression_select(columns.as_ptr(), 2, root);
            assert_ne!(column, ptr::null_mut());

            let mut error = ptr::null_mut();
            let applied_array = vx_array_apply(array, column, &raw mut error);
            assert!(!applied_array.is_null());
            assert!(error.is_null());
            {
                let array = vx_array::as_ref(array);
                let applied_array = vx_array::as_ref(applied_array);
                assert_eq!(applied_array.dtype(), array.dtype());
            }
            vx_array_free(applied_array);
            vx_expression_free(column);

            let columns = [c"age".as_ptr(), c"ololo".as_ptr()];
            let column = vx_expression_select(columns.as_ptr(), 2, root);
            let applied_array = vx_array_apply(array, column, &raw mut error);
            assert!(applied_array.is_null());
            assert!(!error.is_null());
            vx_error_free(error);
            vx_expression_free(column);

            vx_array_free(array);
            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_and_or() {
        let col1 = BoolArray::from_iter([true, false, true, true]);
        let col2 = BoolArray::from_iter([false, true, true, false]);
        let col3 = BoolArray::from_iter([false, true, true, true]);
        let fields = vec![col1.into_array(), col2.into_array(), col3.into_array()];
        let names = ["col1", "col2", "col3"].into();
        let array = StructArray::try_new(names, fields, 4, Validity::NonNullable);

        unsafe {
            let array = vx_array::new(array.unwrap().into_array());

            let root = vx_expression_root();
            let expression_col1 = vx_expression_get_item(c"col1".as_ptr(), root);
            let expression_col2 = vx_expression_get_item(c"col2".as_ptr(), root);
            let expression_col3 = vx_expression_get_item(c"col3".as_ptr(), root);
            let expression_12 = vx_expression_binary(
                vx_binary_operator::VX_OPERATOR_EQ,
                expression_col1,
                expression_col2,
            );
            let expression_23 = vx_expression_binary(
                vx_binary_operator::VX_OPERATOR_EQ,
                expression_col2,
                expression_col3,
            );

            let expressions = [expression_12, expression_23];

            let mut error = ptr::null_mut();
            let expressions_ptr = expressions.as_ptr() as *const *const vx_expression;
            let expression_and123 = vx_expression_and(expressions_ptr, 2);
            assert!(!expression_and123.is_null());
            let applied_array = vx_array_apply(array, expression_and123, &raw mut error);
            assert!(error.is_null());
            assert!(!applied_array.is_null());
            {
                let array = vx_array::as_ref(applied_array).to_bool();
                let expected = BoolArray::from_iter([false, false, true, false]);
                assert_eq!(array.to_bit_buffer(), expected.to_bit_buffer());
            }
            vx_expression_free(expression_and123);
            vx_array_free(applied_array);

            let expression_or123 = vx_expression_or(expressions_ptr, 2);
            assert!(!expression_or123.is_null());
            let applied_array = vx_array_apply(array, expression_or123, &raw mut error);
            assert!(error.is_null());
            assert!(!applied_array.is_null());
            {
                let array = vx_array::as_ref(applied_array).to_bool();
                let expected = BoolArray::from_iter([true, true, true, false]);
                assert_eq!(array.to_bit_buffer(), expected.to_bit_buffer());
            }
            vx_array_free(applied_array);

            vx_expression_free(expression_or123);

            vx_expression_free(expression_23);
            vx_expression_free(expression_12);
            vx_expression_free(expression_col3);
            vx_expression_free(expression_col2);
            vx_expression_free(expression_col1);
            vx_expression_free(root);

            vx_array_free(array);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_empty_and_or() {
        unsafe {
            let root = vx_expression_root();

            let and_empty = vx_expression_and(ptr::null_mut(), 9);
            assert!(and_empty.is_null());
            let or_empty = vx_expression_or(ptr::null_mut(), 9);
            assert!(or_empty.is_null());

            vx_expression_free(root);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_list_contains() {
        let elements = buffer![1i32, 2, 3, 4, 5].into_array();
        let offsets = buffer![0u32, 2, 5, 5].into_array();
        let array = ListArray::try_new(elements, offsets, Validity::NonNullable).unwrap();

        unsafe {
            let root = vx_expression_root();
            let array = vx_array::new(array.into_array());
            let expression_value = vx_expression::new(Box::new(lit(1)));

            let expression = vx_expression_list_contains(root, expression_value);
            assert!(!expression.is_null());

            let mut error = ptr::null_mut();
            let applied = vx_array_apply(array, expression, &raw mut error);
            assert!(error.is_null());
            assert!(!applied.is_null());
            {
                let applied = vx_array::as_ref(applied).to_bool();
                let expected = BoolArray::from_iter([true, false, false]);
                assert_eq!(applied.to_bit_buffer(), expected.to_bit_buffer());
            }
            vx_array_free(applied);

            vx_expression_free(expression_value);
            vx_expression_free(expression);
            vx_array_free(array);

            vx_expression_free(root);
        }
    }
}
