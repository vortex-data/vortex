import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import BenchmarkSection from './BenchmarkSection';
import Modal from './Modal';
import { fetchMetadata } from '../lib/api';
import { BENCHMARK_CONFIGS, CATEGORY_TAGS } from '../lib/config';

export default function App() {
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedGroups, setExpandedGroups] = useState(new Set());
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [searchFilter, setSearchFilter] = useState('');
  const [viewMode, setViewMode] = useState('grid');
  const [modalChart, setModalChart] = useState(null);
  const [showBackToTop, setShowBackToTop] = useState(false);
  const metadataFetched = useRef(false);

  useEffect(() => {
    if (metadataFetched.current) return;
    metadataFetched.current = true;

    async function loadMetadata() {
      try {
        const data = await fetchMetadata();
        setMetadata(data);
        const params = new URLSearchParams(window.location.search);
        if (params.get('expanded') === 'true' && data?.groups) {
          setExpandedGroups(new Set(Object.keys(data.groups)));
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadMetadata();
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 400);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (!metadata || loading) return;

    const hash = window.location.hash;
    if (hash && hash.startsWith('#group-')) {
      const groupId = hash.slice(1);

      const matchingGroup = Object.keys(metadata.groups).find(name =>
        name.replace(/\s+/g, '-') === groupId.replace('group-', '')
      );

      if (matchingGroup) {
        setExpandedGroups(prev => new Set([...prev, matchingGroup]));

        setTimeout(() => {
          const element = document.getElementById(groupId);
          if (element) {
            const headerHeight = 72;
            const y = element.getBoundingClientRect().top + window.scrollY - headerHeight - 16;
            window.scrollTo({ top: y, behavior: 'smooth' });
          }
        }, 100);
      }
    }
  }, [metadata, loading]);

  const getBenchmarkConfig = useCallback((groupName) => {
    return BENCHMARK_CONFIGS.find(c => c.name === groupName) || {};
  }, []);

  const filteredGroups = useMemo(() => {
    if (!metadata?.groups) return [];

    return Object.keys(metadata.groups).filter(groupName => {
      if (categoryFilter !== 'all') {
        const tags = CATEGORY_TAGS[groupName] || [];
        if (!tags.includes(categoryFilter)) return false;
      }

      if (searchFilter) {
        const searchLower = searchFilter.toLowerCase();
        const matchesGroup = groupName.toLowerCase().includes(searchLower);
        const groupData = metadata.groups[groupName];
        const charts = groupData?.charts || [];
        const matchesChart = charts.some(c =>
          c.name.toLowerCase().includes(searchLower)
        );
        if (!matchesGroup && !matchesChart) return false;
      }

      return true;
    });
  }, [metadata, categoryFilter, searchFilter]);

  const toggleGroup = useCallback((groupName) => {
    setExpandedGroups(prev => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    if (metadata?.groups) {
      setExpandedGroups(new Set(Object.keys(metadata.groups)));
      const url = new URL(window.location);
      url.searchParams.set('expanded', 'true');
      window.history.replaceState(null, '', url);
    }
  }, [metadata]);

  const collapseAll = useCallback(() => {
    setExpandedGroups(new Set());
    const url = new URL(window.location);
    url.searchParams.delete('expanded');
    window.history.replaceState(null, '', url);
  }, []);

  const scrollToGroup = useCallback((groupName) => {
    const element = document.getElementById(`group-${groupName.replace(/\s+/g, '-')}`);
    if (element) {
      const headerHeight = 72;
      const y = element.getBoundingClientRect().top + window.scrollY - headerHeight - 16;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
    setSidebarOpen(false);
  }, []);

  const scrollToTop = useCallback(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  const clearFilter = useCallback(() => {
    setSearchFilter('');
    setCategoryFilter('all');
  }, []);

  if (loading) {
    return (
      <div className="app">
        <Header
          sidebarOpen={sidebarOpen}
          onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
          categoryFilter={categoryFilter}
          onCategoryChange={setCategoryFilter}
          searchFilter={searchFilter}
          onSearchChange={setSearchFilter}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          onExpandAll={expandAll}
          onCollapseAll={collapseAll}
        />
        <div className="main-container">
          <main className="main-content">
            <div className="loading-indicator">
              <div className="spinner" />
              <p>Loading benchmarks...</p>
            </div>
          </main>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app">
        <Header
          sidebarOpen={sidebarOpen}
          onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
          categoryFilter={categoryFilter}
          onCategoryChange={setCategoryFilter}
          searchFilter={searchFilter}
          onSearchChange={setSearchFilter}
          viewMode={viewMode}
          onViewModeChange={setViewMode}
          onExpandAll={expandAll}
          onCollapseAll={collapseAll}
        />
        <div className="main-container">
          <main className="main-content">
            <div className="error-indicator">
              <p>Error loading benchmarks: {error}</p>
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <Header
        sidebarOpen={sidebarOpen}
        onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
        categoryFilter={categoryFilter}
        onCategoryChange={setCategoryFilter}
        searchFilter={searchFilter}
        onSearchChange={setSearchFilter}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        onExpandAll={expandAll}
        onCollapseAll={collapseAll}
      />

      <div className="main-container">
        <Sidebar
          isOpen={sidebarOpen}
          groups={filteredGroups}
          onClose={() => setSidebarOpen(false)}
          onGroupClick={scrollToGroup}
          onClearFilter={clearFilter}
          showClearFilter={categoryFilter !== 'all' || searchFilter !== ''}
        />

        <div
          className={`sidebar-overlay ${sidebarOpen ? 'active' : ''}`}
          onClick={() => setSidebarOpen(false)}
        />

        <main className="main-content">
          {filteredGroups.map(groupName => {
            const groupData = metadata.groups[groupName] || {};
            const charts = groupData.charts || [];
            const config = getBenchmarkConfig(groupName);
            const isExpanded = expandedGroups.has(groupName);

            if (config.hidden) return null;

            return (
              <BenchmarkSection
                key={groupName}
                groupName={groupName}
                charts={charts}
                config={config}
                isExpanded={isExpanded}
                onToggle={() => toggleGroup(groupName)}
                viewMode={viewMode}
                onFullscreen={(data) => setModalChart(data)}
                commitRange={metadata.totalCommits}
                summary={groupData.summary}
              />
            );
          })}
        </main>
      </div>

      {showBackToTop && (
        <button
          className="back-to-top visible"
          onClick={scrollToTop}
          aria-label="Back to top"
        >
          ↑
        </button>
      )}

      {modalChart && (
        <Modal
          chartData={modalChart}
          onClose={() => setModalChart(null)}
        />
      )}
    </div>
  );
}
