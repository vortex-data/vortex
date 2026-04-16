const API_BASE = '';

async function readResponse(response) {
  const contentType = response.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    return response.json();
  }

  const text = await response.text();
  return text ? { error: text } : null;
}

export async function fetchMetadata() {
  const response = await fetch(`${API_BASE}/api/metadata`, { cache: 'no-store' });
  const payload = await readResponse(response);

  if (!response.ok) {
    const message = payload?.lastRefreshError
      ? `Failed to fetch metadata: ${payload.lastRefreshError}`
      : payload?.error
        ? `Failed to fetch metadata: ${payload.error}`
        : `Failed to fetch metadata: ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

export async function fetchChartData(groupName, chartName, options = {}) {
  const { startTimestamp, endTimestamp, last, startIdx, endIdx } = options;
  let url = `${API_BASE}/api/data/${encodeURIComponent(groupName)}/${encodeURIComponent(chartName)}`;
  const params = new URLSearchParams();

  if (last) {
    params.set('last', last);
  } else if (startIdx !== undefined || endIdx !== undefined) {
    // Index-based range
    if (startIdx !== undefined) params.set('startIdx', startIdx);
    if (endIdx !== undefined) params.set('endIdx', endIdx);
  } else {
    // Timestamp-based range
    if (startTimestamp) {
      const ts = typeof startTimestamp === 'number'
        ? startTimestamp
        : new Date(startTimestamp).getTime();
      params.set('start', ts);
    }
    if (endTimestamp) {
      const ts = typeof endTimestamp === 'number'
        ? endTimestamp
        : new Date(endTimestamp).getTime();
      params.set('end', ts);
    }
  }

  if (params.toString()) url += '?' + params.toString();

  const response = await fetch(url, { cache: 'no-store' });
  const payload = await readResponse(response);

  if (!response.ok) {
    const message = payload?.error
      ? `Failed to fetch chart data: ${payload.error}`
      : `Failed to fetch chart data: ${response.status}`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}
