import { useEffect, useState } from 'react';
import { fetchMetadata } from '../api';

const METADATA_RETRY_DELAY_MS = 1000;

const initialState = {
  metadata: null,
  loading: true,
  error: null,
};

export default function useBenchmarkMetadata() {
  const [state, setState] = useState(initialState);

  useEffect(() => {
    let active = true;
    let retryTimer = null;
    let activeController = null;

    const clearRetry = () => {
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
        retryTimer = null;
      }
    };

    const loadMetadata = async () => {
      clearRetry();
      activeController?.abort();
      activeController = new AbortController();

      try {
        const metadata = await fetchMetadata({ signal: activeController.signal });
        if (!active) return;

        setState({
          metadata,
          loading: false,
          error: null,
        });
      } catch (error) {
        if (!active || error.name === 'AbortError') return;

        if (error.status === 503 && error.payload?.status !== 'error') {
          setState((current) => ({
            metadata: current.metadata,
            loading: true,
            error: null,
          }));
          retryTimer = window.setTimeout(
            loadMetadata,
            error.retryAfterMs ?? METADATA_RETRY_DELAY_MS,
          );
          return;
        }

        setState({
          metadata: null,
          loading: false,
          error: error.message,
        });
      }
    };

    loadMetadata();

    return () => {
      active = false;
      clearRetry();
      activeController?.abort();
    };
  }, []);

  return state;
}
