import init, { File } from './pkg/vortex_wasm';

let loaded = false;

async function loadInner(module_or_path?: any) {
  await init(module_or_path);

  loaded = true;
}

/**
 * Initialize the Vortex WASM module.
 *
 * Once the promise resolves, Vortex objects are usable.
 *
 * @param module_or_path Optional location to load WASM module (default: load from dist).
 */
export async function vortexLoad(module_or_path?: any) {
  if (!loaded) {
    await loadInner(module_or_path);
  }
}

export default {
  File,
};
