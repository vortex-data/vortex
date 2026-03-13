use std::ffi::c_char;
use std::ffi::c_int;

use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;

use crate::to_string_vec;

crate::box_wrapper!(Expression, vx_expression);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_root() -> *mut vx_expression {
    vx_expression::new(Box::new(root()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vx_expression_select(
    names: *const *const c_char,
    names_len: usize,
    child: *const vx_expression,
) -> *mut vx_expression {
    // TODO don't allocate, convert to [&str]
    let names = unsafe { to_string_vec(names, names_len as c_int) };
    let expr = select(names, vx_expression::as_ref(child).clone());
    vx_expression::new(Box::new(expr))
}
