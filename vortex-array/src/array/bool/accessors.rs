use vortex_error::VortexResult;

use crate::accessor::ArrayAccessor;
use crate::array::BoolArray;
use crate::validity::Validity;
use crate::{ArrayLen, IntoArrayVariant};

static TRUE: bool = true;
static FALSE: bool = false;

impl ArrayAccessor<bool> for BoolArray {
    fn with_iterator<F, R>(&self, f: F) -> VortexResult<R>
    where
        F: for<'a> FnOnce(&mut dyn Iterator<Item = Option<&'a bool>>) -> R,
    {
        let bools = self.boolean_buffer();
        match self.validity() {
            Validity::NonNullable | Validity::AllValid => Ok(f(&mut bools
                .iter()
                .map(|b| Some(if b { &TRUE } else { &FALSE })))),
            Validity::AllInvalid => Ok(f(&mut (0..self.len()).map(|_| None))),
            Validity::Array(valid) => {
                let valids = valid.into_bool()?.boolean_buffer();
                let mut iter = valids.iter().zip(bools.iter()).map(|(is_valid, value)| {
                    is_valid.then_some(if value { &TRUE } else { &FALSE })
                });

                Ok(f(&mut iter))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::accessor::ArrayAccessor;
    use crate::array::BoolArray;

    #[test]
    fn test_bool_accesor() {
        let original = vec![Some(true), None, Some(false), None];
        let array = BoolArray::from_iter(original.clone());

        let bool_vec: Vec<Option<bool>> =
            ArrayAccessor::<bool>::with_iterator(&array, |values_iter| {
                values_iter
                    .map(|b| b.cloned())
                    .collect::<Vec<Option<bool>>>()
            })
            .unwrap();
        assert_eq!(bool_vec, original);
    }
}
