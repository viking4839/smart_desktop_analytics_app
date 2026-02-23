import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
    ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Activity, TrendingUp, AlertCircle, RefreshCcw, Loader } from 'lucide-react';
import { clientDataEngine } from '../../services/ClientDataEngine';
import './AnalyticsDashboard.css';

interface AnalyticsDashboardProps {
    datasetId: string;
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({ datasetId }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dataReady, setDataReady] = useState(false);

    // Data States
    const [correlationData, setCorrelationData] = useState<any>(null);
    const [regressionData, setRegressionData] = useState<any>(null);

    // Regression Controls
    const [xCol, setXCol] = useState<string>('');
    const [yCol, setYCol] = useState<string>('');

    // Check if data is loaded in the engine
    useEffect(() => {
        if (datasetId) {
            // If engine already has data for this dataset, consider it ready
            const hasData = clientDataEngine.getFilteredData().length > 0;
            setDataReady(hasData);
        } else {
            setDataReady(false);
        }
    }, [datasetId]);

    // Extract numeric columns from ClientDataEngine
    const numericColumns = useMemo(() => {
        if (!dataReady) return [];
        const schema = clientDataEngine.getSchema();
        return Object.values(schema)
            .filter(col => ['number', 'float', 'integer', 'currency', 'percentage'].includes(col.data_type))
            .map(col => col.name);
    }, [dataReady]);

    // 1. Fetch Correlation (Automatic on Load)
    const fetchCorrelation = useCallback(async () => {
        if (!datasetId || numericColumns.length < 2) return;
        setLoading(true);
        setError(null);
        try {
            const res: any = await invoke('call_python_backend', {
                command: 'run_advanced_analytics',
                payload: { dataset_id: datasetId, analysis_type: 'correlation', params: {} }
            });
            setCorrelationData(res);

            // Set default X and Y for regression (first two numeric columns)
            if (!xCol && res.columns.length > 0) setXCol(res.columns[0]);
            if (!yCol && res.columns.length > 1) setYCol(res.columns[1]);

        } catch (err: any) {
            setError(err.message || String(err));
        } finally {
            setLoading(false);
        }
    }, [datasetId, numericColumns.length, xCol, yCol]);

    // 2. Fetch Regression (Runs when X or Y changes)
    useEffect(() => {
        if (!datasetId || !xCol || !yCol || !dataReady) return;

        const fetchRegression = async () => {
            try {
                const res: any = await invoke('call_python_backend', {
                    command: 'run_advanced_analytics',
                    payload: {
                        dataset_id: datasetId,
                        analysis_type: 'regression',
                        params: { x_column: xCol, y_column: yCol }
                    }
                });
                setRegressionData(res);
            } catch (err: any) {
                console.error("Regression Error:", err);
            }
        };

        fetchRegression();
    }, [datasetId, xCol, yCol, dataReady]);

    // Initial load of correlation when dataset is ready
    useEffect(() => {
        if (dataReady && numericColumns.length >= 2) {
            fetchCorrelation();
        }
    }, [dataReady, numericColumns.length, fetchCorrelation]);

    // Helper: Map correlation (-1 to 1) to colors (Red -> White -> Blue)
    const getHeatmapColor = (value: number) => {
        if (value === 1) return 'rgba(37, 99, 235, 1)'; // Perfect match (Diagonal)
        const intensity = Math.abs(value);
        return value > 0
            ? `rgba(37, 99, 235, ${intensity})` // Blue for positive
            : `rgba(239, 68, 68, ${intensity})`; // Red for negative
    };

    // If dataset not yet ready
    if (!dataReady) {
        return (
            <div className="analytics-msg">
                <Loader className="spinner" size={20} />
                <span>Loading dataset...</span>
            </div>
        );
    }

    if (numericColumns.length < 2) {
        return (
            <div className="analytics-msg error">
                <AlertCircle size={20} />
                <span>Need at least 2 numeric columns for analysis.</span>
            </div>
        );
    }

    if (error && !correlationData) {
        return (
            <div className="analytics-msg error">
                <AlertCircle size={20} />
                <span>{error}</span>
            </div>
        );
    }

    return (
        <div className="analytics-container">
            {/* --- TOP: Correlation Heatmap with Refresh --- */}
            <div className="analytics-card">
                <div className="card-header">
                    <Activity size={18} />
                    <h3>Correlation Matrix</h3>
                    <button
                        className="refresh-btn"
                        onClick={fetchCorrelation}
                        disabled={loading}
                        title="Refresh correlation"
                    >
                        <RefreshCcw size={14} className={loading ? 'spinner' : ''} />
                    </button>
                </div>
                <p className="card-desc">
                    Find hidden relationships.
                    <span className="positive">Dark Blue</span> = Strong Positive.
                    <span className="negative">Dark Red</span> = Strong Negative.
                </p>

                {loading && !correlationData ? (
                    <div className="analytics-msg">
                        <Loader className="spinner" size={20} />
                        <span>Calculating correlations...</span>
                    </div>
                ) : correlationData && (
                    <div className="heatmap-wrapper">
                        <div className="heatmap-grid" style={{ gridTemplateColumns: `auto repeat(${correlationData.columns.length}, 1fr)` }}>
                            {/* Top Left Empty Cell */}
                            <div></div>

                            {/* Column Headers */}
                            {correlationData.columns.map((col: string) => (
                                <div key={col} className="heatmap-label-top" title={col}>
                                    {col.length > 10 ? col.substring(0, 10) + '…' : col}
                                </div>
                            ))}

                            {/* Matrix Rows */}
                            {correlationData.columns.map((rowCol: string) => (
                                <React.Fragment key={rowCol}>
                                    <div className="heatmap-label-side" title={rowCol}>
                                        {rowCol.length > 15 ? rowCol.substring(0, 15) + '…' : rowCol}
                                    </div>
                                    {correlationData.columns.map((colCol: string) => {
                                        const cell = correlationData.data.find((d: any) => d.x === colCol && d.y === rowCol);
                                        const val = cell ? cell.value : 0;
                                        return (
                                            <div
                                                key={`${rowCol}-${colCol}`}
                                                className="heatmap-cell"
                                                title={`${rowCol} vs ${colCol}\nCorrelation: ${val.toFixed(3)}`}
                                                style={{
                                                    backgroundColor: getHeatmapColor(val),
                                                    color: Math.abs(val) > 0.5 ? 'white' : 'black'
                                                }}
                                            >
                                                {val.toFixed(2)}
                                            </div>
                                        );
                                    })}
                                </React.Fragment>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            {/* --- BOTTOM: Linear Regression --- */}
            <div className="analytics-card">
                <div className="card-header">
                    <TrendingUp size={18} />
                    <h3>Linear Regression</h3>
                </div>

                {/* Controls */}
                <div className="regression-controls">
                    <div className="control-group">
                        <label>Predict (Y-Axis):</label>
                        <select value={yCol} onChange={e => setYCol(e.target.value)}>
                            {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                    </div>
                    <div className="control-group">
                        <label>Based on (X-Axis):</label>
                        <select value={xCol} onChange={e => setXCol(e.target.value)}>
                            {numericColumns.map(col => <option key={col} value={col}>{col}</option>)}
                        </select>
                    </div>
                </div>

                {/* Chart & Stats */}
                {regressionData ? (
                    <div className="regression-results">
                        <div className="regression-stats">
                            <div className="stat-box">
                                <span className="stat-label">Prediction Equation</span>
                                <span className="stat-value equation">{regressionData.equation}</span>
                            </div>
                            <div className="stat-box">
                                <span className="stat-label">Model Accuracy (R²)</span>
                                <span className="stat-value">{(regressionData.r_squared * 100).toFixed(1)}%</span>
                            </div>
                        </div>

                        <div className="chart-container">
                            <ResponsiveContainer width="100%" height={350}>
                                <ComposedChart data={regressionData.scatter_data} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                    <XAxis dataKey="x" type="number" name={xCol} tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                                    <YAxis dataKey="y" type="number" name={yCol} tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value: number) => value.toLocaleString()} />

                                    {/* The Raw Data Points */}
                                    <Scatter name="Data" dataKey="y" fill="#8884d8" opacity={0.6} />

                                    {/* The Line of Best Fit */}
                                    <Line
                                        data={regressionData.line_points}
                                        type="monotone"
                                        dataKey="y"
                                        stroke="#ef4444"
                                        strokeWidth={3}
                                        dot={false}
                                        activeDot={false}
                                        name="Trendline"
                                    />
                                </ComposedChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                ) : (
                    <div className="analytics-msg">
                        <span>Select X and Y columns to see regression analysis.</span>
                    </div>
                )}
            </div>
        </div>
    );
};