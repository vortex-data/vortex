// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Hand-maintained type declarations for the WASM bindings.
// Update this file when the public API in crate/src/wasm.rs changes.

/* eslint-disable */

/// A handle to an opened Vortex file, exposing metadata to JavaScript.
export class VortexFileHandle {
    private constructor();

    free(): void;

    [Symbol.dispose](): void;

    /// The top-level DType of the file as a string.
    readonly dtype: string;
    /// The total number of rows in the file.
    readonly row_count: bigint;
}

/// Initialize the WASM module (sets up panic hook for better error messages).
export function init(): void;

/// Open a Vortex file from raw bytes and return a handle for exploration.
export function open_vortex_file(data: Uint8Array): VortexFileHandle;

export type InitInput =
    | RequestInfo
    | URL
    | Response
    | BufferSource
    | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;

    [key: string]: unknown;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

export function initSync(
    module: { module: SyncInitInput } | SyncInitInput,
): InitOutput;

export default function __wbg_init(
    module_or_path?:
        | { module_or_path: InitInput | Promise<InitInput> }
        | InitInput
        | Promise<InitInput>,
): Promise<InitOutput>;
