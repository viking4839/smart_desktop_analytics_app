import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { open } from '@tauri-apps/api/dialog';
import {
    BarChart,
    Bar,
    LineChart,
    Line,
    PieChart,
    Pie,
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Cell
} from 'recharts';
import './App.css';

// Enhanced Types
interface Dataset {
    id: string;
    name: string;
    row_count: number;
    column_count: number;
    source_path: string;
    source_format: string;
    loaded_at: string;
    schema: Record<string, ColumnInfo>;
}

interface ColumnInfo {
    name: string;
    data_type: string;
    dtype: string;
    nullable: boolean;
    unique_values: number;
    sample_values: any[];
    stats?: any;
    statistics?: {
        numeric_stats?: {
            min: number;
            max: number;
            mean: number;
            median: number;
            q25: number;
            q75: number;
            std: number;
        };
        categorical_stats?: {
            top_categories: Record<string, number>;
            category_count: number;
        };
    };
}

interface SuggestedQuery {
    id: string;
    name: string;
    description: string;
    query: any;
    category: string;
}

interface QualityScore {
    score: number;
    completeness: number;
    null_percentage: number;
    issues?: string[];
}

// ⭐ NEW: Chart data interface
interface ChartData {
    type: 'bar' | 'line' | 'pie' | 'area';
    title: string;
    x_axis: string;
    y_axis: string;
    data: Array<{
        name: string;
        value: number;
    }>;
}

interface QueryResult {
    summary: Record<string, any>;
    statement: string;
    table?: {
        columns: string[];
        rows: any[][];
        row_count: number;
    };
    chart?: ChartData;
    provenance?: {
        datasets: string[];
        columns: string[];
        operations: string[];
        execution_time: number;
        row_count: number;
    };
    success: boolean;
}

// Add these new interfaces at the top with other interfaces
interface AnalysisResult {
    type: string;
    title: string;
    data: any;
    insights: string[];
    recommendations: string[];
}

interface DataQualityIssue {
    type: 'duplicate' | 'inconsistent' | 'missing' | 'outlier' | 'format';
    column: string;
    count: number;
    examples: any[];
    severity: 'low' | 'medium' | 'high';
    recommendation: string;
}

interface DataQualityReport {
    overall_score: number;
    issues: DataQualityIssue[];
    summary: string;
    row_count: number;
    clean_row_count: number;
}

// Data Explorer Component - Updated Props Interface
interface DataExplorerProps {
    datasetId: string;
    datasetName: string;
    preview: any;
    columns: ColumnInfo[];
    filters: Record<string, any>;
    sortConfig: { column: string, direction: 'asc' | 'desc' } | null;
    selectedRows: number[];
    onFilterChange: (filters: Record<string, any>) => void;
    onSortChange: (column: string, direction: 'asc' | 'desc') => void;
    onRowSelect: (rowIndex: number) => void;
    onSelectAll: () => void;
    onRunAnalysis: (type: string, column?: string) => void;
    onCheckDataQuality: () => void;
    onApplyFilters: () => void;
    onRefreshPreview: () => void;
    onLoadMoreRows: (limit: number) => void;
    analysisLoading: string | null;
    analysisResults: AnalysisResult | null;
    dataQualityReport: DataQualityReport | null;
    availableColumns: ColumnInfo[];
    isLoading: boolean;
}

const DataExplorer: React.FC<DataExplorerProps> = ({
    datasetId,
    datasetName,
    preview,
    columns,
    filters,
    sortConfig,
    selectedRows,
    onFilterChange,
    onSortChange,
    onRowSelect,
    onSelectAll,
    onRunAnalysis,
    onCheckDataQuality,
    onApplyFilters,
    onRefreshPreview,
    onLoadMoreRows,
    analysisLoading,
    analysisResults,
    dataQualityReport,
    availableColumns,
    isLoading
}) => {
    const [showFilters, setShowFilters] = useState(false);
    const [page, setPage] = useState(0);
    const [showAnalysisResults, setShowAnalysisResults] = useState(false);
    const [showDataQuality, setShowDataQuality] = useState(false);
    const [selectedAnalysisColumn, setSelectedAnalysisColumn] = useState<string>('');
    const [localFilters, setLocalFilters] = useState<Record<string, any>>(filters);

    const rowsPerPage = 50;

    // Sync local filters with props
    useEffect(() => {
        setLocalFilters(filters);
    }, [filters]);

    // Reset page when preview changes
    useEffect(() => {
        setPage(0);
    }, [preview]);

    if (isLoading) {
        return (
            <div className="empty-state">
                <div className="loading-spinner"></div>
                <p>Loading dataset preview...</p>
            </div>
        );
    }

    if (!preview || !preview.columns || !preview.rows) {
        return (
            <div className="empty-state">
                <p>No dataset selected for exploration</p>
                <p className="hint">Select a dataset from the left panel to explore</p>
            </div>
        );
    }

    // Calculate total pages
    const totalPages = Math.ceil(preview.rows.length / rowsPerPage);
    const paginatedRows = preview.rows.slice(
        page * rowsPerPage,
        (page + 1) * rowsPerPage
    );

    const handleSort = (column: string) => {
        if (sortConfig?.column === column) {
            onSortChange(column, sortConfig.direction === 'asc' ? 'desc' : 'asc');
        } else {
            onSortChange(column, 'asc');
        }
    };

    const getColumnType = (colName: string): string => {
        const col = columns.find(c => c.name === colName);
        return col?.data_type || col?.dtype || 'unknown';
    };

    const isNumericColumn = (colName: string): boolean => {
        const colType = getColumnType(colName);
        return ['integer', 'float', 'numeric', 'int64', 'float64'].includes(colType);
    };

    const handleApplyFilters = () => {
        onFilterChange(localFilters);
        onApplyFilters();
    };

    const handleClearFilters = () => {
        setLocalFilters({});
        onFilterChange({});
        setPage(0);
    };

    const getColumnStats = (colName: string) => {
        const col = availableColumns.find(c => c.name === colName);
        if (!col) return null;

        return {
            unique: col.unique_values,
            nullable: col.nullable,
            stats: col.statistics || col.stats,
            sample: col.sample_values?.slice(0, 3)
        };
    };

    const renderAnalysisResults = () => {
        if (!analysisResults) return null;

        return (
            <div className="analysis-results">
                <div className="results-header">
                    <h4>{analysisResults.title}</h4>
                    <button onClick={() => setShowAnalysisResults(false)}>✕</button>
                </div>

                {analysisResults.type === 'distribution' && (
                    <div className="distribution-results">
                        <h5>Distribution Analysis</h5>
                        <div className="distribution-chart">
                            <div className="distribution-bars">
                                {analysisResults.data?.buckets?.map((bucket: any, i: number) => (
                                    <div key={i} className="distribution-bar">
                                        <div className="bar-label">{bucket.range}</div>
                                        <div
                                            className="bar-fill"
                                            style={{ width: `${bucket.percentage}%` }}
                                        >
                                            <span>{bucket.count} ({bucket.percentage}%)</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {analysisResults.type === 'summary' && (
                    <div className="summary-results">
                        <h5>Summary Statistics</h5>
                        <div className="stats-grid">
                            {Object.entries(analysisResults.data || {}).map(([key, value]) => (
                                <div key={key} className="stat-item">
                                    <span className="stat-label">{key}</span>
                                    <span className="stat-value">
                                        {typeof value === 'number' ? value.toFixed(2) : String(value)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {analysisResults.insights && analysisResults.insights.length > 0 && (
                    <div className="analysis-insights">
                        <h5>Insights</h5>
                        {analysisResults.insights.map((insight, i) => (
                            <div key={i} className="insight-item">
                                {insight}
                            </div>
                        ))}
                    </div>
                )}

                {analysisResults.recommendations && analysisResults.recommendations.length > 0 && (
                    <div className="analysis-recommendations">
                        <h5>Recommendations</h5>
                        {analysisResults.recommendations.map((rec, i) => (
                            <div key={i} className="recommendation-item">
                                {rec}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    const renderDataQualityReport = () => {
        if (!dataQualityReport) return null;

        return (
            <div className="data-quality-report">
                <div className="quality-header">
                    <h4>Data Quality Report</h4>
                    <span className={`quality-score ${dataQualityReport.overall_score > 80 ? 'good' : dataQualityReport.overall_score > 60 ? 'medium' : 'poor'}`}>
                        Score: {dataQualityReport.overall_score}%
                    </span>
                    <button onClick={() => setShowDataQuality(false)}>✕</button>
                </div>

                <div className="quality-summary">
                    <p>{dataQualityReport.summary}</p>
                    <div className="quality-stats">
                        <span>Rows: {dataQualityReport.row_count}</span>
                        <span>Clean rows: {dataQualityReport.clean_row_count}</span>
                        <span>Issues: {dataQualityReport.issues.length}</span>
                    </div>
                </div>

                {dataQualityReport.issues.length > 0 && (
                    <div className="quality-issues">
                        <h5>Issues Found</h5>
                        {dataQualityReport.issues.map((issue, i) => (
                            <div key={i} className={`issue-item ${issue.severity}`}>
                                <div className="issue-header">
                                    <span className="issue-type">{issue.type.toUpperCase()}</span>
                                    <span className="issue-severity">{issue.severity}</span>
                                    <span className="issue-count">{issue.count} occurrences</span>
                                </div>
                                <div className="issue-details">
                                    <strong>Column:</strong> {issue.column}
                                    <br />
                                    <strong>Recommendation:</strong> {issue.recommendation}
                                </div>
                                {issue.examples.length > 0 && (
                                    <div className="issue-examples">
                                        <strong>Examples:</strong> {issue.examples.slice(0, 3).join(', ')}
                                        {issue.examples.length > 3 && '...'}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="data-explorer">
            {/* Dataset Info Header */}
            <div className="dataset-header">
                <div className="dataset-info">
                    <h3>
                        {datasetName}
                        <span className="dataset-id">ID: {datasetId}</span>
                    </h3>
                    <div className="dataset-stats">
                        <span className="stat-badge">{preview.rows.length} rows</span>
                        <span className="stat-badge">{preview.columns.length} columns</span>
                        <span className="stat-badge">{availableColumns.length} schema columns</span>
                    </div>
                </div>
                <div className="dataset-actions">
                    <button
                        className="action-btn refresh-btn"
                        onClick={onRefreshPreview}
                        disabled={analysisLoading === 'refreshing'}
                    >
                        <span className="btn-icon">↻</span>
                        {analysisLoading === 'refreshing' ? 'Refreshing...' : 'Refresh'}
                    </button>
                    <button
                        className="action-btn load-more-btn"
                        onClick={() => onLoadMoreRows(500)}
                        disabled={analysisLoading === 'loading_more'}
                    >
                        <span className="btn-icon">⬇</span>
                        {analysisLoading === 'loading_more' ? 'Loading...' : 'Load More'}
                    </button>
                </div>
            </div>

            {/* Toolbar */}
            <div className="explorer-toolbar">
                <div className="toolbar-left">
                    <button
                        className="toolbar-btn"
                        onClick={() => setShowFilters(!showFilters)}
                    >
                        <span className="btn-icon">🔍</span> Filters
                        {Object.keys(localFilters).length > 0 && (
                            <span className="filter-count">{Object.keys(localFilters).length}</span>
                        )}
                    </button>
                    <button
                        className="toolbar-btn"
                        onClick={() => {
                            onCheckDataQuality();
                            setShowDataQuality(true);
                        }}
                        disabled={analysisLoading === 'data_quality'}
                    >
                        <span className="btn-icon">✅</span>
                        {analysisLoading === 'data_quality' ? 'Checking...' : 'Data Quality'}
                    </button>
                    <div className="row-count">
                        Showing {preview.rows.length} rows
                        {Object.keys(filters).length > 0 && ` (filtered)`}
                    </div>
                </div>
                <div className="toolbar-right">
                    <div className="pagination">
                        <button
                            className="pagination-btn"
                            onClick={() => setPage(Math.max(0, page - 1))}
                            disabled={page === 0}
                        >
                            ← Prev
                        </button>
                        <span className="page-info">
                            Page {page + 1} of {totalPages}
                        </span>
                        <button
                            className="pagination-btn"
                            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
                            disabled={page >= totalPages - 1}
                        >
                            Next →
                        </button>
                    </div>
                </div>
            </div>

            {/* Filters Panel with Apply Button */}
            {showFilters && (
                <div className="filters-panel">
                    <div className="filters-header">
                        <h4>Column Filters</h4>
                        <div className="filter-actions">
                            <button
                                className="clear-filters-btn"
                                onClick={handleClearFilters}
                            >
                                Clear All
                            </button>
                            <button
                                className="apply-filters-btn"
                                onClick={handleApplyFilters}
                                disabled={analysisLoading === 'filtering'}
                            >
                                {analysisLoading === 'filtering' ? 'Applying...' : 'Apply Filters'}
                            </button>
                        </div>
                    </div>
                    <div className="filter-inputs">
                        {preview.columns.map((col: string, index: number) => (
                            <div key={index} className="filter-input">
                                <label>{col} <span className="column-type-hint">({getColumnType(col)})</span></label>
                                {isNumericColumn(col) ? (
                                    <div className="range-filter">
                                        <input
                                            type="number"
                                            placeholder="Min"
                                            value={localFilters[`${col}_min`] || ''}
                                            onChange={(e) => setLocalFilters({
                                                ...localFilters,
                                                [`${col}_min`]: e.target.value
                                            })}
                                        />
                                        <span>to</span>
                                        <input
                                            type="number"
                                            placeholder="Max"
                                            value={localFilters[`${col}_max`] || ''}
                                            onChange={(e) => setLocalFilters({
                                                ...localFilters,
                                                [`${col}_max`]: e.target.value
                                            })}
                                        />
                                    </div>
                                ) : (
                                    <div className="text-filter">
                                        <input
                                            type="text"
                                            placeholder="Contains..."
                                            value={localFilters[col] || ''}
                                            onChange={(e) => setLocalFilters({
                                                ...localFilters,
                                                [col]: e.target.value
                                            })}
                                        />
                                        <button
                                            className="quick-filter-btn"
                                            onClick={() => {
                                                const colStats = getColumnStats(col);
                                                if (colStats?.sample) {
                                                    setLocalFilters({
                                                        ...localFilters,
                                                        [col]: colStats.sample[0]
                                                    });
                                                }
                                            }}
                                        >
                                            Sample
                                        </button>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Data Grid */}
            <div className="data-grid-container">
                <table className="data-grid">
                    <thead>
                        <tr>
                            <th className="selector-col">
                                <input
                                    type="checkbox"
                                    onChange={onSelectAll}
                                    checked={selectedRows.length === paginatedRows.length && paginatedRows.length > 0}
                                />
                            </th>
                            {preview.columns.map((col: string, index: number) => (
                                <th
                                    key={index}
                                    className={sortConfig?.column === col ? `sorted ${sortConfig.direction}` : ''}
                                    onClick={() => handleSort(col)}
                                >
                                    <div className="column-header">
                                        <span>{col}</span>
                                        <span className="data-type-badge">
                                            {getColumnType(col)}
                                        </span>
                                        {sortConfig?.column === col && (
                                            <span className="sort-indicator">
                                                {sortConfig.direction === 'asc' ? '↑' : '↓'}
                                            </span>
                                        )}
                                    </div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {paginatedRows.map((row: any[], rowIndex: number) => (
                            <tr
                                key={rowIndex}
                                className={selectedRows.includes(rowIndex) ? 'selected' : ''}
                            >
                                <td className="selector-col">
                                    <input
                                        type="checkbox"
                                        checked={selectedRows.includes(rowIndex)}
                                        onChange={() => onRowSelect(rowIndex)}
                                    />
                                </td>
                                {row.map((cell, cellIndex) => (
                                    <td key={cellIndex}>
                                        <div className="cell-content">
                                            {typeof cell === 'number' ?
                                                <span className="numeric-cell">
                                                    {cell.toLocaleString()}
                                                </span> :
                                                <span className="text-cell">
                                                    {String(cell)}
                                                </span>
                                            }
                                        </div>
                                        <div className="cell-actions">
                                            <button
                                                className="cell-action-btn"
                                                title="Filter by this value"
                                                onClick={() => {
                                                    const newFilters = {
                                                        ...localFilters,
                                                        [preview.columns[cellIndex]]: cell
                                                    };
                                                    setLocalFilters(newFilters);
                                                    onFilterChange(newFilters);
                                                }}
                                            >
                                                🔍
                                            </button>
                                        </div>
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Quick Analysis Panel */}
            <div className="quick-analysis">
                <div className="analysis-header">
                    <h4>Quick Analysis</h4>
                    <div className="column-selector">
                        <select
                            value={selectedAnalysisColumn}
                            onChange={(e) => setSelectedAnalysisColumn(e.target.value)}
                        >
                            <option value="">Select column (optional)</option>
                            {preview.columns.map((col: string, index: number) => (
                                <option key={index} value={col}>{col}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <div className="analysis-buttons">
                    <button
                        className="analysis-btn"
                        onClick={() => {
                            onRunAnalysis('distribution', selectedAnalysisColumn);
                            setShowAnalysisResults(true);
                        }}
                        disabled={analysisLoading === 'distribution'}
                    >
                        <span className="analysis-icon">📈</span>
                        {analysisLoading === 'distribution' ? 'Analyzing...' : 'Distribution'}
                    </button>

                    <button
                        className="analysis-btn"
                        onClick={() => {
                            onRunAnalysis('summary_stats', selectedAnalysisColumn);
                            setShowAnalysisResults(true);
                        }}
                        disabled={analysisLoading === 'summary_stats'}
                    >
                        <span className="analysis-icon">📊</span>
                        {analysisLoading === 'summary_stats' ? 'Analyzing...' : 'Summary Stats'}
                    </button>

                    <button
                        className="analysis-btn"
                        onClick={() => {
                            onRunAnalysis('outliers', selectedAnalysisColumn);
                            setShowAnalysisResults(true);
                        }}
                        disabled={analysisLoading === 'outliers'}
                    >
                        <span className="analysis-icon">🔍</span>
                        {analysisLoading === 'outliers' ? 'Finding...' : 'Find Outliers'}
                    </button>

                    <button
                        className="analysis-btn"
                        onClick={() => {
                            onRunAnalysis('duplicates', undefined);
                            setShowAnalysisResults(true);
                        }}
                        disabled={analysisLoading === 'duplicates'}
                    >
                        <span className="analysis-icon">🔄</span>
                        {analysisLoading === 'duplicates' ? 'Finding...' : 'Find Duplicates'}
                    </button>
                </div>
            </div>

            {/* Analysis Results Panel */}
            {(showAnalysisResults && analysisResults) && renderAnalysisResults()}

            {/* Data Quality Report Panel */}
            {(showDataQuality && dataQualityReport) && renderDataQualityReport()}
        </div>
    );
};

function App() {
    // State
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<string>('');
    const [availableColumns, setAvailableColumns] = useState<ColumnInfo[]>([]);
    const [metrics, setMetrics] = useState<string[]>(['sum()']);
    const [groupBy, setGroupBy] = useState<string[]>([]);
    const [chartType, setChartType] = useState<'none' | 'bar' | 'line' | 'pie' | 'area'>('none');
    const [forceChartType, setForceChartType] = useState<boolean>(false);
    const [xAxisColumn, setXAxisColumn] = useState<string>('');
    const [yAxisMetric, setYAxisMetric] = useState<string>('');
    const [chartPreview, setChartPreview] = useState<ChartData | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<QueryResult | null>(null);
    const [error, setError] = useState<string>('');
    const [backendStatus, setBackendStatus] = useState<'unknown' | 'connected' | 'disconnected'>('unknown');
    const [overrideChartType, setOverrideChartType] = useState<string | null>(null);

    // Enhanced state
    const [suggestedQueries, setSuggestedQueries] = useState<SuggestedQuery[]>([]);
    const [quickInsights, setQuickInsights] = useState<string[]>([]);
    const [qualityScore, setQualityScore] = useState<QualityScore | null>(null);
    const [showSuggestions, setShowSuggestions] = useState(false);

    // Data Explorer state - RESET when dataset changes
    const [activeRightTab, setActiveRightTab] = useState<'results' | 'dataExplorer'>('results');
    const [datasetPreview, setDatasetPreview] = useState<any>(null);
    const [explorerFilters, setExplorerFilters] = useState<Record<string, any>>({});
    const [sortConfig, setSortConfig] = useState<{ column: string, direction: 'asc' | 'desc' } | null>(null);
    const [selectedRows, setSelectedRows] = useState<number[]>([]);
    const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null);
    const [dataQualityReport, setDataQualityReport] = useState<DataQualityReport | null>(null);
    const [analysisLoading, setAnalysisLoading] = useState<string | null>(null);
    const [previewLoading, setPreviewLoading] = useState<boolean>(false);

    // Initialize
    useEffect(() => {
        checkBackend();
        loadDatasets();
    }, []);

    // Reset Data Explorer when dataset changes
    const resetDataExplorer = useCallback(() => {
        setDatasetPreview(null);
        setExplorerFilters({});
        setSortConfig(null);
        setSelectedRows([]);
        setAnalysisResults(null);
        setDataQualityReport(null);
        setAnalysisLoading(null);
        setError('');
    }, []);

    // Load everything when dataset changes
    useEffect(() => {
        if (selectedDataset) {
            console.log('Loading schema for dataset:', selectedDataset);
            loadSchema(selectedDataset);
            resetDataExplorer();

            // If we're on the Data Explorer tab, load preview immediately
            if (activeRightTab === 'dataExplorer') {
                loadDatasetPreview();
            }
        } else {
            resetDataExplorer();
        }
    }, [selectedDataset, activeRightTab, resetDataExplorer]);

    // Load dataset preview when switching to Data Explorer tab
    useEffect(() => {
        if (activeRightTab === 'dataExplorer' && selectedDataset && !datasetPreview) {
            console.log('Switched to Data Explorer, loading preview');
            loadDatasetPreview();
        }
    }, [activeRightTab, selectedDataset, datasetPreview]);

    // Backend connection
    const checkBackend = async () => {
        try {
            await invoke('call_python_backend', {
                command: 'ping',
                payload: {}
            });
            setBackendStatus('connected');
        } catch (err) {
            setBackendStatus('disconnected');
            console.error('Backend connection failed:', err);
        }
    };

    // Load datasets
    const loadDatasets = async () => {
        try {
            const response = await invoke('call_python_backend', {
                command: 'list_datasets',
                payload: {}
            }) as { datasets: Dataset[] };
            setDatasets(response.datasets || []);
        } catch (err) {
            console.error('Failed to load datasets:', err);
        }
    };

    // Load schema with enhanced features
    const loadSchema = async (datasetId: string) => {
        try {
            console.log('Loading schema for:', datasetId);
            const response = await invoke('call_python_backend', {
                command: 'get_schema',
                payload: { dataset_id: datasetId }
            }) as {
                schema: Record<string, ColumnInfo>;
                suggested_queries?: SuggestedQuery[];
                quick_insights?: string[];
                quality_score?: QualityScore;
            };

            console.log('Schema response:', response);
            const columns = Object.values(response.schema || {});
            setAvailableColumns(columns);

            setSuggestedQueries(response.suggested_queries || []);
            setQuickInsights(response.quick_insights || []);
            setQualityScore(response.quality_score || null);

        } catch (err) {
            console.error('Failed to load schema:', err);
            setError('Failed to load dataset schema');
        }
    };

    // File upload
    const handleFileUpload = async () => {
        try {
            const selected = await open({
                multiple: false,
                filters: [{
                    name: 'Data Files',
                    extensions: ['csv', 'xlsx', 'xls', 'json', 'parquet']
                }]
            });

            if (selected && !Array.isArray(selected)) {
                setLoading(true);
                setError('');

                const response = await invoke('call_python_backend', {
                    command: 'register_dataset',
                    payload: { file_path: selected }
                }) as { dataset: Dataset };

                await loadDatasets();
                setSelectedDataset(response.dataset.id);
                setActiveRightTab('dataExplorer'); // Switch to Data Explorer tab
                setLoading(false);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to upload file');
            setLoading(false);
        }
    };

    // Load dataset preview
    const loadDatasetPreview = async (limit: number = 100) => {
        if (!selectedDataset) return;

        try {
            setPreviewLoading(true);
            console.log('Loading preview for dataset:', selectedDataset, 'limit:', limit);
            const response = await invoke('call_python_backend', {
                command: 'preview_dataset',
                payload: {
                    dataset_id: selectedDataset,
                    limit: limit
                }
            }) as {
                dataset_id: string;
                dataset_name: string;
                preview: {
                    columns: string[];
                    rows: any[][];
                    types: Record<string, string>;
                };
            };

            console.log('Preview loaded:', response);
            setDatasetPreview(response.preview);
            setPreviewLoading(false);
        } catch (err: any) {
            console.error('Failed to load dataset preview:', err);
            setError('Failed to load dataset preview');
            setPreviewLoading(false);
        }
    };

    // Refresh dataset preview
    const refreshDatasetPreview = async () => {
        if (!selectedDataset) return;
        setAnalysisLoading('refreshing');
        await loadDatasetPreview();
        setAnalysisLoading(null);
    };

    // Load more rows
    const loadMoreRows = async (limit: number) => {
        if (!selectedDataset) return;
        setAnalysisLoading('loading_more');
        await loadDatasetPreview(limit);
        setAnalysisLoading(null);
    };

    // Run analysis on dataset
    const runAnalysis = async (analysisType: string, column?: string) => {
        if (!selectedDataset) {
            setError('Please select a dataset first');
            return;
        }

        setAnalysisLoading(analysisType);
        try {
            const response = await invoke('call_python_backend', {
                command: 'analyze_dataset',
                payload: {
                    dataset_id: selectedDataset,
                    analysis_type: analysisType,
                    column: column || null
                }
            }) as AnalysisResult;

            setAnalysisResults(response);
            setAnalysisLoading(null);
        } catch (err: any) {
            console.error(`Failed to run ${analysisType} analysis:`, err);
            setError(`Failed to run ${analysisType} analysis`);
            setAnalysisLoading(null);
        }
    };

    // Check data quality
    const checkDataQuality = async () => {
        if (!selectedDataset) {
            setError('Please select a dataset first');
            return;
        }

        setAnalysisLoading('data_quality');
        try {
            const response = await invoke('call_python_backend', {
                command: 'check_data_quality',
                payload: { dataset_id: selectedDataset }
            }) as DataQualityReport;

            setDataQualityReport(response);
            setAnalysisLoading(null);
        } catch (err: any) {
            console.error('Failed to check data quality:', err);
            setError('Failed to check data quality');
            setAnalysisLoading(null);
        }
    };

    // Apply filters to dataset
    const applyFilters = async () => {
        if (!selectedDataset || !datasetPreview) return;

        setAnalysisLoading('filtering');
        try {
            const response = await invoke('call_python_backend', {
                command: 'filter_dataset',
                payload: {
                    dataset_id: selectedDataset,
                    filters: explorerFilters
                }
            }) as {
                filtered_preview: any;
                filtered_count: number;
                original_count: number;
            };

            // Update preview with filtered data
            setDatasetPreview({
                ...datasetPreview,
                rows: response.filtered_preview.rows,
                columns: response.filtered_preview.columns
            });

            setError('');

        } catch (err: any) {
            console.error('Failed to apply filters:', err);
            setError('Failed to apply filters');
        } finally {
            setAnalysisLoading(null);
        }
    };

    const previewChart = async () => {
        if (!selectedDataset || chartType === 'none') return;

        setLoading(true);
        try {
            const query = {
                dataset_id: selectedDataset,
                metrics: metrics,
                group_by: groupBy.length > 0 ? groupBy : null,
                chart_type: chartType,
                force_chart_type: forceChartType,
                x_axis: xAxisColumn || null,
                y_axis: yAxisMetric || null,
                query_type: 'aggregation'
            };

            const response = await invoke('call_python_backend', {
                command: 'execute_query',
                payload: { query_dict: query }
            }) as QueryResult;

            // If backend didn't return a chart, create one from the table data
            if (!response.chart && response.table) {
                const chartData = createChartFromTable(response.table, chartType);
                setChartPreview(chartData);
            } else {
                setChartPreview(response.chart || null);
            }
        } catch (err: any) {
            console.error('Failed to generate chart:', err);
        } finally {
            setLoading(false);
        }
    };

    // Helper function to create chart from table data
    const createChartFromTable = (table: any, type: string): ChartData => {
        if (!table || table.rows.length === 0) {
            return {
                type: type as 'bar' | 'line' | 'pie' | 'area',
                title: '',
                x_axis: '',
                y_axis: '',
                data: []
            };
        }

        const chartData: ChartData = {
            type: type as 'bar' | 'line' | 'pie' | 'area',
            title: `Chart of ${table.columns.join(', ')}`,
            x_axis: table.columns[0] || 'Category',
            y_axis: table.columns[1] || 'Value',
            data: []
        };

        // Convert table rows to chart data format
        if (table.columns.length >= 2) {
            chartData.data = table.rows.map((row: any[]) => ({
                name: String(row[0]),
                value: typeof row[1] === 'number' ? row[1] : parseFloat(row[1]) || 0
            })).slice(0, 20);
        }

        return chartData;
    };

    // Execute query
    const executeQuery = async () => {
        if (!selectedDataset) {
            setError('Please select a dataset');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const query = {
                dataset_id: selectedDataset,
                metrics: metrics,
                group_by: groupBy.length > 0 ? groupBy : null,
                chart_type: chartType === 'none' ? null : chartType,
                query_type: 'aggregation'
            };

            console.log('Sending query:', query);

            const response = await invoke('call_python_backend', {
                command: 'execute_query',
                payload: { query_dict: query }
            }) as QueryResult;

            console.log('Received response:', response);

            setResult(response);
            setChartPreview(response.chart || null);
        } catch (err: any) {
            setError(err.message || 'Query execution failed');
        } finally {
            setLoading(false);
        }
    };

    // Run suggested query
    const runSuggestedQuery = async (suggestion: SuggestedQuery) => {
        setMetrics(suggestion.query.metrics || ['count()']);
        setGroupBy(suggestion.query.group_by || []);
        setShowSuggestions(false);
        setTimeout(() => executeQuery(), 100);
    };

    // Metric management
    const addMetric = () => setMetrics([...metrics, 'count()']);

    const updateMetric = (index: number, value: string) => {
        const newMetrics = [...metrics];
        newMetrics[index] = value;
        setMetrics(newMetrics);
    };

    const removeMetric = (index: number) => {
        if (metrics.length > 1) {
            setMetrics(metrics.filter((_, i) => i !== index));
        }
    };

    // Group by management
    const toggleGroupBy = (column: string) => {
        if (groupBy.includes(column)) {
            setGroupBy(groupBy.filter(g => g !== column));
        } else {
            setGroupBy([...groupBy, column]);
        }
    };

    // Helper: Parse metric string
    const parseMetric = (metric: string): [string, string] => {
        const match = metric.match(/^(\w+)\(([^)]*)\)$/);
        return match ? [match[1], match[2]] : ['sum', ''];
    };

    const formatMetric = (func: string, col: string): string => {
        return func === 'count' || !col ? `${func}()` : `${func}(${col})`;
    };

    // Render metric selector
    const renderMetricSelector = (metric: string, onChange: (value: string) => void) => {
        const [func, col] = parseMetric(metric);
        const numericColumns = availableColumns.filter(c =>
            ['integer', 'float', 'currency', 'percentage', 'numeric'].includes(c.data_type || c.dtype)
        );

        return (
            <div className="metric-input">
                <select
                    className="function-select"
                    value={func}
                    onChange={(e) => onChange(formatMetric(e.target.value, col))}
                >
                    <option value="sum">Sum</option>
                    <option value="avg">Average</option>
                    <option value="count">Count</option>
                    <option value="min">Min</option>
                    <option value="max">Max</option>
                    <option value="median">Median</option>
                </select>

                {func !== 'count' && (
                    <select
                        className="column-select"
                        value={col}
                        onChange={(e) => onChange(formatMetric(func, e.target.value))}
                    >
                        <option value="">Select column</option>
                        {numericColumns.map(c => (
                            <option key={c.name} value={c.name}>{c.name}</option>
                        ))}
                    </select>
                )}
            </div>
        );
    };

    // ⭐ NEW: Render chart based on data
    const renderChart = (chartData: ChartData) => {
        if (!chartData || !chartData.data || chartData.data.length === 0) {
            return null;
        }

        const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#30cfd0'];

        const commonProps = {
            data: chartData.data,
            margin: { top: 20, right: 30, left: 20, bottom: 20 }
        };

        switch (chartData.type) {
            case 'bar':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart {...commonProps}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="value" fill="#667eea" name={chartData.y_axis} />
                        </BarChart>
                    </ResponsiveContainer>
                );

            case 'line':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart {...commonProps}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="value" stroke="#667eea" name={chartData.y_axis} />
                        </LineChart>
                    </ResponsiveContainer>
                );

            case 'pie':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie
                                data={chartData.data}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                outerRadius={100}
                                label
                            >
                                {chartData.data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                        </PieChart>
                    </ResponsiveContainer>
                );

            case 'area':
                return (
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart {...commonProps}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Area type="monotone" dataKey="value" stroke="#667eea" fill="#667eea" fillOpacity={0.6} name={chartData.y_axis} />
                        </AreaChart>
                    </ResponsiveContainer>
                );

            default:
                return null;
        }
    };

    // Get selected dataset name
    const getSelectedDatasetName = () => {
        if (!selectedDataset) return '';
        const dataset = datasets.find(d => d.id === selectedDataset);
        return dataset?.name || 'Dataset';
    };

    return (
        <div className="app-container">
            {/* Header */}
            <header className="app-header">
                <h1>Smart Desktop Analytics</h1>
                <div className="status-indicator">
                    <div className={`status-dot ${backendStatus}`} />
                    <span>Backend: {backendStatus}</span>
                    <button onClick={checkBackend} className="refresh-btn">Refresh</button>
                </div>
            </header>

            <div className="app-main">
                {/* Left Panel - Data Management */}
                <div className="left-panel">
                    <div className="panel-section">
                        <h2>Datasets</h2>
                        <button
                            onClick={handleFileUpload}
                            disabled={loading}
                            className="upload-btn"
                        >
                            {loading ? 'Uploading...' : 'Upload File'}
                        </button>

                        {/* Quality Score */}
                        {qualityScore && selectedDataset && (
                            <div className="quality-card">
                                <h3>Data Quality</h3>
                                <div className="quality-bar">
                                    <div
                                        className="quality-fill"
                                        style={{
                                            width: `${qualityScore.score}%`,
                                            background: qualityScore.score > 80 ? '#10b981' : qualityScore.score > 50 ? '#f59e0b' : '#ef4444'
                                        }}
                                    >
                                        {qualityScore.score}%
                                    </div>
                                </div>
                                <div className="quality-details">
                                    <span>Completeness: {qualityScore.completeness}%</span>
                                    <span>Missing: {qualityScore.null_percentage}%</span>
                                </div>
                            </div>
                        )}

                        {/* Datasets List */}
                        <div className="dataset-list">
                            {datasets.length === 0 ? (
                                <div className="empty-state">
                                    <p>No datasets yet. Upload one to get started!</p>
                                </div>
                            ) : (
                                datasets.map(dataset => (
                                    <div
                                        key={dataset.id}
                                        className={`dataset-item ${selectedDataset === dataset.id ? 'selected' : ''}`}
                                        onClick={() => setSelectedDataset(dataset.id)}
                                    >
                                        <div className="dataset-name">{dataset.name}</div>
                                        <div className="dataset-meta">
                                            {dataset.row_count} rows × {dataset.column_count} cols
                                        </div>
                                        <div className="dataset-format">
                                            {dataset.source_format}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Columns */}
                    {selectedDataset && availableColumns.length > 0 && (
                        <div className="panel-section">
                            <h2>Columns</h2>
                            <div className="column-list">
                                {availableColumns.map(column => (
                                    <div key={column.name} className="column-item">
                                        <div className="column-header">
                                            <span className="column-name">{column.name}</span>
                                            <span className={`column-type ${column.data_type || column.dtype}`}>
                                                {column.data_type || column.dtype}
                                            </span>
                                        </div>

                                        {column.statistics?.numeric_stats && (
                                            <div className="column-stats">
                                                <span>Range: {column.statistics.numeric_stats.min.toFixed(0)} - {column.statistics.numeric_stats.max.toFixed(0)}</span>
                                                <span>Median: {column.statistics.numeric_stats.median.toFixed(0)}</span>
                                            </div>
                                        )}

                                        <div className="column-stats">
                                            <span>Unique: {column.unique_values}</span>
                                            <span>Nulls: {column.nullable ? 'Yes' : 'No'}</span>
                                        </div>

                                        <button
                                            onClick={() => toggleGroupBy(column.name)}
                                            className={`group-by-btn ${groupBy.includes(column.name) ? 'active' : ''}`}
                                        >
                                            {groupBy.includes(column.name) ? 'Grouped' : 'Group By'}
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Middle Panel - Query Builder */}
                <div className="middle-panel">
                    <div className="panel-section">
                        <h2>Query Builder</h2>

                        {/* Quick Insights */}
                        {quickInsights.length > 0 && (
                            <div className="insights-container">
                                <h3>Quick Insights</h3>
                                {quickInsights.map((insight, i) => (
                                    <div key={i} className="insight-item">
                                        {insight}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Suggested Queries */}
                        {suggestedQueries.length > 0 && (
                            <div className="suggestions-container">
                                <div className="suggestions-header">
                                    <h3>Suggested Queries</h3>
                                    <button
                                        className="toggle-btn"
                                        onClick={() => setShowSuggestions(!showSuggestions)}
                                    >
                                        {showSuggestions ? 'Hide' : 'Show'}
                                    </button>
                                </div>

                                {showSuggestions && (
                                    <div className="suggestions-list">
                                        {suggestedQueries.slice(0, 5).map((sq) => (
                                            <div key={sq.id} className="suggestion-item">
                                                <div className="suggestion-name">{sq.name}</div>
                                                <div className="suggestion-desc">{sq.description}</div>
                                                <button
                                                    className="run-suggestion-btn"
                                                    onClick={() => runSuggestedQuery(sq)}
                                                >
                                                    Run
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}

                        {selectedDataset ? (
                            <>
                                {/* Metrics */}
                                <div className="query-section">
                                    <h3>Metrics</h3>
                                    {metrics.map((metric, index) => (
                                        <div key={index} className="metric-row">
                                            {renderMetricSelector(metric, (value) => updateMetric(index, value))}
                                            {metrics.length > 1 && (
                                                <button
                                                    onClick={() => removeMetric(index)}
                                                    className="remove-btn"
                                                >
                                                    ×
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                    <button onClick={addMetric} className="add-btn">
                                        + Add Metric
                                    </button>
                                </div>

                                {/* Group By */}
                                <div className="query-section">
                                    <h3>Group By</h3>
                                    {groupBy.length > 0 ? (
                                        <div className="group-by-tags">
                                            {groupBy.map(col => (
                                                <span key={col} className="group-tag">
                                                    {col}
                                                    <button onClick={() => toggleGroupBy(col)}>×</button>
                                                </span>
                                            ))}
                                        </div>
                                    ) : (
                                        <p className="hint">Select columns from the left panel to group by</p>
                                    )}
                                </div>

                                <div className="query-section">
                                    <h3>Chart Visualization</h3>
                                    <div className="chart-controls">
                                        <select
                                            value={chartType}
                                            onChange={(e) => setChartType(e.target.value as any)}
                                            className="function-select"
                                        >
                                            <option value="none">No chart</option>
                                            <option value="bar">Bar Chart</option>
                                            <option value="line">Line Chart</option>
                                            <option value="pie">Pie Chart</option>
                                            <option value="area">Area Chart</option>
                                        </select>

                                        {chartType !== 'none' && (
                                            <div className="chart-options">
                                                <label className="checkbox-label">
                                                    <input
                                                        type="checkbox"
                                                        checked={forceChartType}
                                                        onChange={(e) => setForceChartType(e.target.checked)}
                                                    />
                                                    Force chart type (override auto-detection)
                                                </label>

                                                {forceChartType && chartType === 'pie' && (
                                                    <p className="hint">
                                                        Pie charts work best with fewer than 10 categories
                                                    </p>
                                                )}

                                                {forceChartType && chartType === 'line' && (
                                                    <p className="hint">
                                                        Line charts work best with ordered/numeric X-axis
                                                    </p>
                                                )}
                                            </div>
                                        )}

                                        {chartType !== 'none' && groupBy.length > 0 && (
                                            <div className="chart-axis-selector">
                                                <div className="axis-row">
                                                    <label>X-Axis:</label>
                                                    <select
                                                        value={xAxisColumn || ''}
                                                        onChange={(e) => setXAxisColumn(e.target.value)}
                                                        className="column-select"
                                                    >
                                                        <option value="">Auto (Group By)</option>
                                                        {availableColumns
                                                            .filter(c => !groupBy.includes(c.name) || chartType === 'pie')
                                                            .map(c => (
                                                                <option key={c.name} value={c.name}>{c.name}</option>
                                                            ))}
                                                    </select>
                                                </div>

                                                {chartType !== 'pie' && (
                                                    <div className="axis-row">
                                                        <label>Y-Axis:</label>
                                                        <select
                                                            value={yAxisMetric || ''}
                                                            onChange={(e) => setYAxisMetric(e.target.value)}
                                                            className="function-select"
                                                        >
                                                            <option value="">Auto (First Metric)</option>
                                                            {metrics.map((metric, i) => (
                                                                <option key={i} value={metric}>{metric}</option>
                                                            ))}
                                                        </select>
                                                    </div>
                                                )}

                                                <button
                                                    className="preview-chart-btn"
                                                    onClick={previewChart}
                                                    disabled={!selectedDataset}
                                                >
                                                    Preview Chart
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <button
                                    onClick={executeQuery}
                                    disabled={loading || !selectedDataset}
                                    className="execute-btn"
                                >
                                    {loading ? 'Running...' : 'Run Query'}
                                </button>
                            </>
                        ) : (
                            <div className="empty-state">
                                <p>Select or upload a dataset to start analyzing</p>
                            </div>
                        )}

                        {error && (
                            <div className="error-message">
                                <strong>Error:</strong> {error}
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel with Tabs */}
                <div className="right-panel">
                    <div className="panel-tabs">
                        <button
                            className={`tab-btn ${activeRightTab === 'results' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('results')}
                        >
                            Results
                            {result && (
                                <span className="tab-badge">
                                    {result.table?.row_count || result.summary?.row_count || 0}
                                </span>
                            )}
                        </button>
                        <button
                            className={`tab-btn ${activeRightTab === 'dataExplorer' ? 'active' : ''}`}
                            onClick={() => {
                                setActiveRightTab('dataExplorer');
                                if (selectedDataset) {
                                    loadDatasetPreview();
                                }
                            }}
                        >
                            Data Explorer
                            {selectedDataset && (
                                <span className="tab-badge">
                                    {datasets.find(d => d.id === selectedDataset)?.row_count || 0}
                                </span>
                            )}
                        </button>
                    </div>

                    <div className="tab-content">
                        {activeRightTab === 'results' ? (
                            <div className="panel-section">
                                <h2>Results</h2>
                                {result ? (
                                    <>
                                        {/* Statement */}
                                        <div className="result-statement">
                                            <h3>Summary</h3>
                                            <p>{result.statement}</p>
                                            {result.summary && (
                                                <div className="summary-stats">
                                                    <span>Rows: {result.summary.row_count}</span>
                                                    <span>Time: {result.summary.execution_time?.toFixed(3)}s</span>
                                                </div>
                                            )}
                                        </div>

                                        {chartPreview && (
                                            <div className="result-chart preview">
                                                <h3>Chart Preview</h3>
                                                {renderChart(chartPreview)}
                                                <div className="chart-actions">
                                                    <button onClick={() => setChartPreview(null)} className="remove-btn">
                                                        Close Preview
                                                    </button>
                                                    <button
                                                        onClick={executeQuery}
                                                        className="execute-btn"
                                                        disabled={loading}
                                                    >
                                                        {loading ? 'Running...' : 'Run with Chart'}
                                                    </button>
                                                </div>
                                            </div>
                                        )}

                                        {result.chart && (
                                            <div className="result-chart">
                                                <div className="chart-header">
                                                    <h3>{result.chart.title}</h3>
                                                    <div className="chart-type-badge">{result.chart.type} Chart</div>
                                                </div>
                                                {renderChart(result.chart)}
                                                <div className="chart-details">
                                                    <span>X: {result.chart.x_axis}</span>
                                                    <span>Y: {result.chart.y_axis}</span>
                                                </div>
                                            </div>
                                        )}

                                        {/* Table */}
                                        {result.table && (
                                            <div className="result-table">
                                                <h3>Data ({result.table.row_count} rows)</h3>
                                                <div className="table-container">
                                                    <table>
                                                        <thead>
                                                            <tr>
                                                                {result.table.columns.map((col, i) => (
                                                                    <th key={i}>{col}</th>
                                                                ))}
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {result.table.rows.slice(0, 10).map((row, i) => (
                                                                <tr key={i}>
                                                                    {row.map((cell, j) => (
                                                                        <td key={j}>
                                                                            {typeof cell === 'number' ? cell.toFixed(2) : String(cell)}
                                                                        </td>
                                                                    ))}
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                    {result.table.row_count > 10 && (
                                                        <p className="table-footer">
                                                            Showing first 10 of {result.table.row_count} rows
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        {/* Provenance */}
                                        {result.provenance && (
                                            <div className="result-provenance">
                                                <details>
                                                    <summary>Technical Details</summary>
                                                    <div className="provenance-details">
                                                        <p><strong>Datasets:</strong> {result.provenance.datasets.join(', ')}</p>
                                                        <p><strong>Columns:</strong> {result.provenance.columns.join(', ')}</p>
                                                        <p><strong>Operations:</strong> {result.provenance.operations.join(', ')}</p>
                                                        <p><strong>Execution:</strong> {result.provenance.execution_time.toFixed(3)}s</p>
                                                    </div>
                                                </details>
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <div className="empty-state">
                                        <p>Run a query to see results here</p>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="panel-section">
                                <h2>Data Explorer</h2>
                                <DataExplorer
                                    datasetId={selectedDataset}
                                    datasetName={getSelectedDatasetName()}
                                    preview={datasetPreview}
                                    columns={availableColumns}
                                    filters={explorerFilters}
                                    sortConfig={sortConfig}
                                    selectedRows={selectedRows}
                                    onFilterChange={setExplorerFilters}
                                    onSortChange={(col, dir) => setSortConfig({ column: col, direction: dir })}
                                    onRowSelect={(rowIndex) => {
                                        setSelectedRows(prev =>
                                            prev.includes(rowIndex)
                                                ? prev.filter(i => i !== rowIndex)
                                                : [...prev, rowIndex]
                                        );
                                    }}
                                    onSelectAll={() => {
                                        if (selectedRows.length === datasetPreview?.rows.length) {
                                            setSelectedRows([]);
                                        } else {
                                            setSelectedRows(Array.from({ length: datasetPreview?.rows.length || 0 }, (_, i) => i));
                                        }
                                    }}
                                    onRunAnalysis={runAnalysis}
                                    onCheckDataQuality={checkDataQuality}
                                    onApplyFilters={applyFilters}
                                    onRefreshPreview={refreshDatasetPreview}
                                    onLoadMoreRows={loadMoreRows}
                                    analysisLoading={analysisLoading}
                                    analysisResults={analysisResults}
                                    dataQualityReport={dataQualityReport}
                                    availableColumns={availableColumns}
                                    isLoading={previewLoading}
                                />
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Footer */}
            <footer className="app-footer">
                <div className="footer-stats">
                    {datasets.length > 0 && (
                        <span>{datasets.length} dataset{datasets.length !== 1 ? 's' : ''} loaded</span>
                    )}
                    {selectedDataset && (
                        <span>Selected: {getSelectedDatasetName()}</span>
                    )}
                    {result && result.provenance && (
                        <span>Query: {result.provenance.execution_time.toFixed(3)}s</span>
                    )}
                </div>
                <div className="footer-info">
                    Smart Desktop Analytics v1.0.0
                </div>
            </footer>
        </div>
    );
}

export default App;