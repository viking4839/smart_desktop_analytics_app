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
    chart?: ChartData;  // ⭐ NEW: Chart data from backend
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

    // Initialize
    useEffect(() => {
        checkBackend();
        loadDatasets();
    }, []);

    // Load schema when dataset selected
    useEffect(() => {
        if (selectedDataset) {
            loadSchema(selectedDataset);
        }
    }, [selectedDataset]);

    useEffect(() => {
        if (result) {
            setOverrideChartType(null);
        }
    }, [result]);

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
            const response = await invoke('call_python_backend', {
                command: 'get_schema',
                payload: { dataset_id: datasetId }
            }) as {
                schema: Record<string, ColumnInfo>;
                suggested_queries?: SuggestedQuery[];
                quick_insights?: string[];
                quality_score?: QualityScore;
            };

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
                setLoading(false);
            }
        } catch (err: any) {
            setError(err.message || 'Failed to upload file');
            setLoading(false);
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
                force_chart_type: forceChartType,  // Add this flag
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

    // Add this helper function
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
            })).slice(0, 20); // Limit to 20 items for performance
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
                chart_type: chartType,  // Always send chart type, even if 'none'
                query_type: 'aggregation'
            };

            console.log('Sending chart type:', chartType); // Debug log

            const response = await invoke('call_python_backend', {
                command: 'execute_query',
                payload: { query_dict: query }
            }) as QueryResult;

            console.log('Received chart type:', response.chart?.type); // Debug log

            setResult(response);
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

                {/* Right Panel - Results */}
                <div className="right-panel">
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
                                        {chartPreview && renderChart(chartPreview)}
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

                                {result?.chart && (
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
                                {/* ⭐ NEW: Chart Visualization */}
                                {result?.chart && (
                                    <div className="result-chart">
                                        <div className="chart-header">
                                            <h3>{result.chart.title}</h3>
                                            <div className="chart-type-selector">
                                                <label>Change to: </label>
                                                <select
                                                    value={overrideChartType || result.chart.type}
                                                    onChange={(e) => {
                                                        const newType = e.target.value;
                                                        setOverrideChartType(newType);

                                                        // Create a new chart with the selected type
                                                        if (!result.chart) return;

                                                        const newChart = {
                                                            ...result.chart,
                                                            type: newType as 'bar' | 'line' | 'pie' | 'area',
                                                            title: result.chart.title ?? '',
                                                            x_axis: result.chart.x_axis ?? '',
                                                            y_axis: result.chart.y_axis ?? '',
                                                            data: result.chart.data ?? []
                                                        };

                                                        // Update result with new chart type
                                                        setResult({
                                                            ...result,
                                                            chart: newChart
                                                        });
                                                    }}
                                                    className="chart-type-select"
                                                >
                                                    <option value="bar">Bar Chart</option>
                                                    <option value="line">Line Chart</option>
                                                    <option value="pie">Pie Chart</option>
                                                    <option value="area">Area Chart</option>
                                                </select>
                                            </div>
                                        </div>

                                        {/* Render with override type if set */}
                                        {renderChart({
                                            ...result.chart,
                                            type: (overrideChartType || result.chart.type) as any
                                        })}

                                        <div className="chart-details">
                                            <span>X: {result.chart.x_axis}</span>
                                            <span>Y: {result.chart.y_axis}</span>
                                            <span>Type: {overrideChartType || result.chart.type}</span>
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
                </div>
            </div>

            {/* Footer */}
            <footer className="app-footer">
                <div className="footer-stats">
                    {datasets.length > 0 && (
                        <span>{datasets.length} dataset{datasets.length !== 1 ? 's' : ''} loaded</span>
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

function createChartFromTable(table: { columns: string[]; rows: any[][]; row_count: number; }, chartType: string) {
    throw new Error('Function not implemented.');
}
