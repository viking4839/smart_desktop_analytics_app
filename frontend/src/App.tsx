import React, { useState, useEffect } from 'react';
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
import {
    Upload,
    RefreshCw,
    BarChart2,
    Table,
    Database,
    File,
    Loader,
    Plus,
    X,
    AlertCircle,
    Trash2,
    Activity,
    Settings,
    Check
} from 'lucide-react';

// ----- NEW: Client‑Side DataExplorer -----
import { DataExplorer } from '../components/DataExplorer/DataExplorer';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard/AnalyticsDashboard';
import { AiAssistantWidget } from '../components/common/Apiconnector';
import ERDView from '../components/ERD/ERDView';
import './App.css';

// ----- All existing TypeScript interfaces (unchanged) -----
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

interface ChartData {
    type: 'bar' | 'line' | 'pie' | 'area';
    title: string;
    x_axis: string;
    y_axis: string;
    data: Array<{ name: string; value: number }>;
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

function App() {
    // ----- Core dataset & query builder state -----
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
    const [suggestedQueries, setSuggestedQueries] = useState<SuggestedQuery[]>([]);
    const [quickInsights, setQuickInsights] = useState<string[]>([]);
    const [qualityScore, setQualityScore] = useState<QualityScore | null>(null);
    const [showSuggestions, setShowSuggestions] = useState(false);
    const [activeRightTab, setActiveRightTab] = useState<'results' | 'dataExplorer' | 'analytics' | 'erd'>('results');
    const [columnSearchTerm, setColumnSearchTerm] = useState('');
    const [isAssistantOpen, setIsAssistantOpen] = useState(false);

    // ----- Initialisation -----
    useEffect(() => {
        checkBackend();
        loadDatasets();
    }, []);

    // ----- Reset query state when dataset changes (FIXES "column does not exist" error) -----
    useEffect(() => {
        if (selectedDataset) {
            // Reset query builder state to default values
            setMetrics(['sum()']);
            setGroupBy([]);
            setChartType('none');
            setForceChartType(false);
            setXAxisColumn('');
            setYAxisMetric('');
            setChartPreview(null);
            setResult(null);
            setError('');
        }
    }, [selectedDataset]);

    // ----- Load schema when dataset changes -----
    useEffect(() => {
        if (selectedDataset) {
            loadSchema(selectedDataset);
        }
    }, [selectedDataset]);

    // ----- Backend connection -----
    const checkBackend = async () => {
        try {
            await invoke('call_python_backend', { command: 'ping', payload: {} });
            setBackendStatus('connected');
        } catch (err) {
            setBackendStatus('disconnected');
            console.error('Backend connection failed:', err);
        }
    };

    // ----- Load datasets -----
    const loadDatasets = async () => {
        try {
            const response = await invoke<any>('call_python_backend', {
                command: 'list_datasets',
                payload: {}
            });
            setDatasets(response.datasets || []);
        } catch (err) {
            console.error('Failed to load datasets:', err);
        }
    };
    const handleDeleteDataset = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation(); // Prevents the row click event from firing

        if (!window.confirm("Are you sure you want to delete this dataset?")) {
            return;
        }

        try {
            await invoke('call_python_backend', {
                command: 'remove_dataset',
                payload: { dataset_id: id }
            });


            if (selectedDataset === id) {
                setSelectedDataset('');
            }

            // Refresh the sidebar list
            await loadDatasets();
        } catch (err) {
            console.error("Failed to delete dataset:", err);
            // Optional: you can add a toast or alert here
            alert(`Could not delete dataset: ${err}`);
        }
    };

    const loadSchema = async (datasetId: string) => {
        try {
            const response = await invoke<any>('call_python_backend', {
                command: 'get_schema',
                payload: { dataset_id: datasetId }
            });

            const columns = Object.values(response.schema || {});
            setAvailableColumns(columns as any);
            setSuggestedQueries(response.suggested_queries || []);
            setQuickInsights(response.quick_insights || []);
            setQualityScore(response.quality_score || null);
        } catch (err) {
            console.error('Failed to load schema:', err);
            setError('Failed to load dataset schema');
        }
    };

    // ----- File upload -----
    const handleFileUpload = async () => {
        try {
            const selected = await open({
                multiple: false,
                filters: [{ name: 'Data Files', extensions: ['csv', 'xlsx', 'xls', 'json', 'parquet'] }]
            });

            if (selected && typeof selected === 'string') {
                setLoading(true);
                setError('');
                const response = await invoke<any>('call_python_backend', {
                    command: 'register_dataset',
                    payload: { file_path: selected }
                });
                await loadDatasets();
                setSelectedDataset(response.dataset.id);
                setActiveRightTab('dataExplorer');
                setLoading(false);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to upload file');
            setLoading(false);
        }
    };

    // ----- Execute query (with validation) -----
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

            const response = await invoke<any>('call_python_backend', {
                command: 'execute_query',
                payload: { query_dict: query }
            });

            console.log('Received response:', response);

            setResult(response);
            setChartPreview(response.chart || null);
            setActiveRightTab('results'); // Switch to results tab automatically
        } catch (err: any) {
            setError(err.message || 'Query execution failed');
        } finally {
            setLoading(false);
        }
    };

    // ----- Chart preview -----
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

            const response = await invoke<any>('call_python_backend', {
                command: 'execute_query',
                payload: { query_dict: query }
            });

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

        if (table.columns.length >= 2) {
            chartData.data = table.rows.map((row: any[]) => ({
                name: String(row[0]),
                value: typeof row[1] === 'number' ? row[1] : parseFloat(row[1]) || 0
            })).slice(0, 20);
        }

        return chartData;
    };

    // ----- Run suggested query -----
    const runSuggestedQuery = async (suggestion: SuggestedQuery) => {
        setMetrics(suggestion.query.metrics || ['count()']);
        setGroupBy(suggestion.query.group_by || []);
        setShowSuggestions(false);
        setTimeout(() => executeQuery(), 100);
    };

    // ----- Metric management -----
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

    // ----- Group by management -----
    const toggleGroupBy = (column: string) => {
        if (groupBy.includes(column)) {
            setGroupBy(groupBy.filter(g => g !== column));
        } else {
            setGroupBy([...groupBy, column]);
        }
    };

    // ----- Metric parsing helpers -----
    const parseMetric = (metric: string): [string, string] => {
        const match = metric.match(/^(\w+)\(([^)]*)\)$/);
        return match ? [match[1], match[2]] : ['sum', ''];
    };

    const formatMetric = (func: string, col: string): string => {
        return func === 'count' || !col ? `${func}()` : `${func}(${col})`;
    };

    // ----- Metric selector renderer (with validation against available columns) -----
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
                        value={numericColumns.some(c => c.name === col) ? col : ''}
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

    // ----- Chart renderer -----
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
                                {chartData.data.map((_, index) => (
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

    const getSelectedDatasetName = () => {
        if (!selectedDataset) return '';
        const dataset = datasets.find(d => d.id === selectedDataset);
        return dataset?.name || 'Dataset';
    };

    return (
        <div className="app-container">
            {/* Header */}
{/* ========== MODERN HEADER ========== */}
            <header className="modern-header">
                <div className="header-left">
                    <div className="brand-logo">
                        <Database size={25} color="#cdcbf1" />
                    </div>
                    <h1>Smart Desktop Analytics</h1>
                    
                </div>
                <div className="header-right">
                    <div className="status-badge">
                        <div className={`status-dot ${result ? 'active' : 'idle'}`}></div>
                        <span>Backend Connected</span>
                    </div>
                    <button className="icon-action-btn" title="Settings">
                        <Settings size={18} />
                    </button>
                </div>
            </header>

            <div className="app-main">
                {/* ========== LEFT PANEL – Dark Theme Data Sources ========== */}
                <div className="left-panel dark-theme">
                    {/* Data Management Header */}
                    <div className="data-management-header">
                        <div className="header-title">
                            <Database size={20} />
                            <h2>Data Sources</h2>
                        </div>
                        <button
                            className="refresh-btn-dark"
                            onClick={loadDatasets}
                            title="Refresh dataset list"
                        >
                            <RefreshCw size={16} />
                        </button>
                    </div>

                    {/* Upload Button */}
                    <button
                        className="upload-btn primary"
                        onClick={handleFileUpload}
                        disabled={loading}
                    >
                        {loading ? <Loader size={16} className="spin" /> : <Upload size={16} />}
                        {loading ? 'Uploading...' : 'Upload Dataset'}
                    </button>

                    {/* Data Quality Card */}
                    {qualityScore && selectedDataset && (
                        <div className="quality-card dark">
                            <h3>Data Quality</h3>
                            <div className="quality-bar">
                                <div
                                    className="quality-fill"
                                    style={{
                                        width: `${qualityScore.score}%`,
                                        background: qualityScore.score > 80 ? '#10b981' : qualityScore.score > 50 ? '#f59e0b' : '#ef4444',
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

                    {/* Dataset List */}
                    <div className="dataset-list">
                        {datasets.length === 0 ? (
                            <div className="empty-statee">
                                <Database size={48} strokeWidth={1.5} />
                                <p>No datasets yet. Upload one to get started!</p>
                            </div>
                        ) : (
                            datasets.map(dataset => (
                                <div
                                    key={dataset.id}
                                    className={`dataset-item ${selectedDataset === dataset.id ? 'selected' : ''}`}
                                    onClick={() => setSelectedDataset(dataset.id)}
                                >
                                    <div className="dataset-details">
                                        <div className="dataset-name">
                                            <File size={16} style={{ marginRight: 8 }} />
                                            <span className="truncate">{dataset.name}</span>
                                        </div>
                                        <div className="dataset-meta">
                                            {(dataset.row_count ?? 0).toLocaleString()} rows × {dataset.column_count ?? 0} cols
                                        </div>
                                        <div className="dataset-format badge">{(dataset.source_format ?? 'file').toUpperCase()}</div>
                                    </div>
                                    <button
                                        className="delete-dataset-btn"
                                        onClick={(e) => handleDeleteDataset(e, dataset.id)}
                                        title="Delete Dataset"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Assistant Footer Widget */}
                    <div
                        className={`assistant-footer ${isAssistantOpen ? 'active' : ''}`}
                        onClick={() => setIsAssistantOpen(!isAssistantOpen)}
                    >
                        <div className="assistant-header">
                            <div className="icon-glow">
                                <Activity size={18} />
                            </div>
                            <span>Smart Assistant Active</span>
                        </div>
                        <p className="assistant-message">
                            Select a dataset to unlock AI insights, schema mapping, and automated query suggestions.
                        </p>
                    </div>
                </div>

                {/* ========== MIDDLE PANEL – Config + Results Canvas ========== */}
                <div className="middle-panel modern">
                    {/* Left Config Column */}
                    <div className="config-column">
                        <h2>Query Builder</h2>

                        {/* Quick Insights */}
                        {quickInsights.length > 0 && (
                            <div className="insights-container">
                                <h3>Quick Insights</h3>
                                {quickInsights.map((insight, i) => (
                                    <div key={i} className="insight-item">{insight}</div>
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

                        {/* Available Columns Section */}
                        {selectedDataset && availableColumns.length > 0 && (
                            <div className="available-columns-section">
                                <div className="section-header">
                                    <Database size={18} />
                                    <span>Available Columns</span>
                                </div>
                                <div className="search-bar">
                                    <input
                                        type="text"
                                        placeholder="Search columns..."
                                        value={columnSearchTerm}
                                        onChange={(e) => setColumnSearchTerm(e.target.value)}
                                        className="search-input"
                                    />
                                </div>
                                <div className="columns-grid">
                                    {availableColumns
                                        .filter(col => col.name.toLowerCase().includes(columnSearchTerm.toLowerCase()))
                                        .map(column => (
                                            <div key={column.name} className="column-card">
                                                <div className="column-header">
                                                    <span className="column-name">{column.name}</span>
                                                    <span className={`data-type-badge ${column.data_type || column.dtype}`}>
                                                        {column.data_type || column.dtype}
                                                    </span>
                                                </div>
                                                <div className="column-stats">
                                                    <span>Unique: {column.unique_values}</span>
                                                    <span>Nulls: {column.nullable ? 'Yes' : 'No'}</span>
                                                </div>
                                                {column.statistics?.numeric_stats && (
                                                    <div className="numeric-preview">
                                                        Min: {column.statistics.numeric_stats.min.toFixed(1)} &nbsp;
                                                        Max: {column.statistics.numeric_stats.max.toFixed(1)} &nbsp;
                                                        Median: {column.statistics.numeric_stats.median.toFixed(1)}
                                                    </div>
                                                )}
                                                <button
                                                    className={`group-action-btn ${groupBy.includes(column.name) ? 'active' : ''}`}
                                                    onClick={() => toggleGroupBy(column.name)}
                                                >
                                                    {groupBy.includes(column.name) ? '✓ Grouped' : 'Group By'}
                                                </button>
                                            </div>
                                        ))}
                                </div>
                                {groupBy.length > 0 && (
                                    <div className="selected-summary">
                                        {groupBy.length} column{groupBy.length !== 1 ? 's' : ''} selected for grouping
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Metrics Section */}
                        {selectedDataset ? (
                            <>
                                <div className="metrics-section">
                                    <h3>Metrics</h3>
                                    {metrics.map((metric, index) => (
                                        <div key={index} className="metric-row">
                                            {renderMetricSelector(metric, (value) => updateMetric(index, value))}
                                            {metrics.length > 1 && (
                                                <button onClick={() => removeMetric(index)} className="remove-btn">
                                                    <X size={16} />
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                    <button onClick={addMetric} className="add-btn">
                                        <Plus size={16} /> Add Metric
                                    </button>
                                </div>

                                {/* Selected Groups Tags */}
                                <div className="selected-groups-section">
                                    <h3>Selected Groups</h3>
                                    {groupBy.length > 0 ? (
                                        <div className="group-tags">
                                            {groupBy.map(col => (
                                                <span key={col} className="group-tag">
                                                    {col}
                                                    <button onClick={() => toggleGroupBy(col)}>×</button>
                                                </span>
                                            ))}
                                        </div>
                                    ) : (
                                        <p className="hint">Click "Group By" on any column above</p>
                                    )}
                                </div>

                                {/* Chart Visualization Section */}
                                <div className="chart-section">
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
                                                    <p className="hint">Pie charts work best with fewer than 10 categories</p>
                                                )}
                                                {forceChartType && chartType === 'line' && (
                                                    <p className="hint">Line charts work best with ordered/numeric X-axis</p>
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

                                {/* Run Query Button */}
                                <button
                                    onClick={executeQuery}
                                    disabled={loading || !selectedDataset}
                                    className="run-query-btn"
                                >
                                    {loading ? <Loader size={16} className="spin" /> : <BarChart2 size={16} />}
                                    {loading ? 'Running...' : 'Run Query'}
                                </button>
                            </>
                        ) : (
                            <div className="empty-state">
                                <Database size={48} strokeWidth={1.5} />
                                <p>Select or upload a dataset to start analyzing</p>
                            </div>
                        )}

                        {error && (
                            <div className="error-message">
                                <AlertCircle size={16} />
                                <strong>Error:</strong> {error}
                            </div>
                        )}
                    </div>

                    {/* Right Results Canvas */}

                </div>

                {/* ========== RIGHT PANEL – Results / Data Explorer ========== */}
                {/* ========== RIGHT PANEL – Results / Data Explorer / Analytics ========== */}
                <div className="right-panel">
                    <div className="panel-tabs">
                        <button
                            className={`tab-btn ${activeRightTab === 'results' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('results')}
                        >
                            <BarChart2 size={16} />
                            Results
                            {result && (
                                <span className="tab-badge">
                                    {result.table?.row_count || result.summary?.row_count || 0}
                                </span>
                            )}
                        </button>
                        <button
                            className={`tab-btn ${activeRightTab === 'dataExplorer' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('dataExplorer')}
                        >
                            <Table size={16} />
                            Data Explorer
                            {selectedDataset && (
                                <span className="tab-badge">
                                    {datasets.find(d => d.id === selectedDataset)?.row_count || 0}
                                </span>
                            )}
                        </button>
                        <button
                            className={`tab-btn ${activeRightTab === 'analytics' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('analytics')}
                        >
                            <Activity size={16} />
                            Analytics & Charts
                        </button>
                        <button
                            className={`tab-btn ${activeRightTab === 'erd' ? 'active' : ''}`}
                            onClick={() => setActiveRightTab('erd')}
                        >
                            <Activity size={16} />
                            ERD
                        </button>
                    </div>

                    <div className="tab-content">


                        {activeRightTab === 'results' && (
                            /* ----- Results Panel ----- */
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

                                        {/* Chart Preview */}
                                        {chartPreview && (
                                            <div className="result-chart preview">
                                                <h3>Chart Preview</h3>
                                                {renderChart(chartPreview)}
                                                <div className="chart-actions">
                                                    <button onClick={() => setChartPreview(null)} className="remove-btn">
                                                        Close Preview
                                                    </button>
                                                    <button onClick={executeQuery} className="execute-btn" disabled={loading}>
                                                        {loading ? 'Running...' : 'Run with Chart'}
                                                    </button>
                                                </div>
                                            </div>
                                        )}

                                        {/* Chart */}
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
                                        <BarChart2 size={48} strokeWidth={1.5} />
                                        <p>Run a query to see results here</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {activeRightTab === 'dataExplorer' && (
                            <div className="panel-section data-explorer-wrapper">
                                {selectedDataset ? (
                                    <DataExplorer
                                        datasetId={selectedDataset}
                                    />
                                ) : (
                                    <div className="empty-state">
                                        <Database size={48} strokeWidth={1.5} />
                                        <p>Select a dataset to explore</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {activeRightTab === 'analytics' && (
                            <div className="panel-section analytics-wrapper">
                                {selectedDataset ? (
                                    <AnalyticsDashboard datasetId={selectedDataset} />
                                ) : (
                                    <div className="empty-state">
                                        <Database size={48} strokeWidth={1.5} />
                                        <p>Select a dataset to view analytics</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {activeRightTab === 'erd' && (
                            <div className="panel-section erd-wrapper">
                                <ERDView
                                    datasets={datasets}
                                    selectedDatasetId={selectedDataset}
                                />
                            </div>
                        )}

                    </div>
                </div>
            </div>

            {/* Footer */}
{/* ========== MODERN FOOTER ========== */}
            <footer className="modern-footer">
                <div className="footer-left">
                    <div className="footer-stat">
                        <Database size={14} />
                        <span>{datasets.length} Datasets Loaded</span>
                    </div>
                    {selectedDataset && (
                        <div className="footer-stat active-context">
                            <File size={14} />
                            <span>Context: {getSelectedDatasetName()}</span>
                        </div>
                    )}
                </div>
                <div className="footer-right">
                    {result && result.provenance && (
                        <div className="footer-stat success-text">
                            <Check size={14} />
                            <span>Query: {result.provenance.execution_time.toFixed(3)}s</span>
                        </div>
                    )}
                    <span className="version-text">v1.0.0</span>
                </div>
            </footer>
            <AiAssistantWidget
                datasets={datasets}
                selectedDatasetId={selectedDataset}
                currentResult={result}
                availableColumns={availableColumns}
                quickInsights={quickInsights}
                suggestedQueries={suggestedQueries}
                qualityScore={qualityScore}
                isOpen={isAssistantOpen}                  // <--- Add this
                onClose={() => setIsAssistantOpen(false)}

            />
        </div>
    );
}

export default App;