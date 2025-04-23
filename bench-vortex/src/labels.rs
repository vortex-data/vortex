#![allow(unused_variables)]

use cfg_if::cfg_if;

pub trait FutureCustomLabelsExt: Sized + Future {
    fn with_current_labels(self) -> impl Future<Output = Self::Output> {
        cfg_if! {
            if #[cfg(target_os = "linux")] {
                custom_labels::asynchronous::Label::with_current_labels(self)
            } else {
                self
            }
        }
    }

    fn with_label<K, V>(self, k: K, v: V) -> impl Future<Output = Self::Output>
    where
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        cfg_if! {
            if #[cfg(target_os = "linux")] {
                custom_labels::asynchronous::Label::with_label(self, k, v)
            } else {
                self
            }
        }
    }

    fn with_labels<I, K, V>(self, i: I) -> impl Future<Output = Self::Output>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<[u8]>,
        V: AsRef<[u8]>,
    {
        cfg_if! {
            if #[cfg(target_os = "linux")] {
                custom_labels::asynchronous::Label::with_labels(self, i)
            } else {
                self
            }
        }
    }
}

impl<F: Future> FutureCustomLabelsExt for F {}
