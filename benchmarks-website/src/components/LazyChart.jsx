import { useRef, useState, useEffect } from 'react';

const OFFSCREEN_MARGIN = '200px';

/**
 * Wraps a child component and only mounts it when the container is near
 * the viewport (within OFFSCREEN_MARGIN). Shows a lightweight placeholder
 * until then, avoiding heavy ECharts instances for off-screen charts.
 */
export default function LazyChart({ children, height = 380 }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: OFFSCREEN_MARGIN },
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  if (visible) {
    return children;
  }

  return (
    <div ref={ref} className="chart-container">
      <div className="chart-canvas-placeholder" style={{ height: `${height}px` }}>
        <div className="chart-loading-spinner" />
      </div>
    </div>
  );
}
