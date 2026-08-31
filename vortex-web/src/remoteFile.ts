// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/** Fetch a remote Vortex file using only browser APIs. Relative URLs resolve against the page. */
export async function fetchRemoteFile(source: string, signal?: AbortSignal): Promise<File> {
  const url = new URL(source, window.location.href);
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`Unsupported file URL protocol: ${url.protocol}`);
  }

  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

  const blob = await response.blob();
  const pathName = url.pathname.split('/').filter(Boolean).pop();
  const fileName = pathName ? decodeURIComponent(pathName) : 'remote.vortex';
  return new File([blob], fileName, { type: blob.type });
}
