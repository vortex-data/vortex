use jni::JNIEnv;
use vortex::error::VortexError;

pub static ILLEGAL_ARGUMENT_CLASS: &str = "java/lang/IllegalArgumentException";
pub static RUNTIME_EXC_CLASS: &str = "java/lang/RuntimeException";

pub trait Throwable: Sized {
    /// Throw the error with a particular class type.
    fn throw(self, env: &mut JNIEnv, throw_class: &str);

    fn throw_with_msg(self, env: &mut JNIEnv, throw_class: &str, msg: &str);

    fn throw_illegal_argument(self, env: &mut JNIEnv) {
        self.throw(env, ILLEGAL_ARGUMENT_CLASS);
    }

    fn throw_runtime(self, env: &mut JNIEnv, msg: &str) {
        self.throw_with_msg(env, RUNTIME_EXC_CLASS, msg);
    }
}

impl Throwable for VortexError {
    fn throw(self, env: &mut JNIEnv, throw_class: &str) {
        let error_string = self.to_string();
        env.throw_new(throw_class, error_string.as_str())
            .expect("ThrowFailed");
    }

    fn throw_with_msg(self, env: &mut JNIEnv, throw_class: &str, msg: &str) {
        let error_string = format!("{msg}: {self}");
        env.throw_new(throw_class, error_string.as_str())
            .expect("ThrowFailed");
    }
}
