/* tslint:disable */
/* eslint-disable */

/**
 * A handle to an opened Vortex file, exposing metadata to JavaScript.
 */
export class VortexFileHandle {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * The top-level DType of the file as a string.
     */
    readonly dtype: string;
    /**
     * The total number of rows in the file.
     */
    readonly row_count: bigint;
}

/**
 * Initialize the WASM module (sets up panic hook for better error messages).
 */
export function init(): void;

/**
 * Open a Vortex file from raw bytes and return a handle for exploration.
 *
 * Call this from JavaScript after reading a `.vortex` file via drag-and-drop.
 */
export function open_vortex_file(data: Uint8Array): VortexFileHandle;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly init: () => void;
    readonly open_vortex_file: (a: number, b: number) => [number, number, number];
    readonly __wbg_vortexfilehandle_free: (a: number, b: number) => void;
    readonly vortexfilehandle_row_count: (a: number) => bigint;
    readonly vortexfilehandle_dtype: (a: number) => [number, number];
    readonly wasm_bindgen__closure__destroy__h315b7af3ad8e2911: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hd9aaad206556e4f9: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen__convert__closures_____invoke__hbcb1741f2446bbd2: (a: number, b: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
