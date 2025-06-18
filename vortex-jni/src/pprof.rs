use axum::http::StatusCode;
use axum::response::IntoResponse;
use jni::JNIEnv;
use jni::objects::JClass;
use tokio::net::TcpListener;

use crate::TOKIO_RUNTIME;

// Extra native method for activating pprof server.
#[allow(clippy::expect_used)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_vortex_jni_NativeLogging_initProf(_env: JNIEnv, _class: JClass) {
    let router =
        axum::Router::new().route("/debug/pprof/heap", axum::routing::get(handle_get_heap));
    let listener = TOKIO_RUNTIME.block_on(async move {
        TcpListener::bind("127.0.0.1:3001")
            .await
            .expect("pprof tcp server bind port 3001")
    });

    TOKIO_RUNTIME.spawn(async move {
        axum::serve(listener, router)
            .await
            .expect("axum::serve port 3001");
    });
}

#[allow(clippy::expect_used)]
pub async fn handle_get_heap() -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut prof_ctl = jemalloc_pprof::PROF_CTL
        .as_ref()
        .expect("PROF_CTL lock")
        .lock()
        .await;
    require_profiling_activated(&prof_ctl)?;
    let pprof = prof_ctl
        .dump_pprof()
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    Ok(pprof)
}

/// Checks whether jemalloc profiling is activated an returns an error response if not.
fn require_profiling_activated(
    prof_ctl: &jemalloc_pprof::JemallocProfCtl,
) -> Result<(), (StatusCode, String)> {
    if prof_ctl.activated() {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "heap profiling not activated".into()))
    }
}
