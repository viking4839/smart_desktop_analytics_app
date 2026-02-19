import React, { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
    Search, RefreshCw, AlertCircle, ArrowUpDown,
    ArrowUp, ArrowDown, Database, Loader
} from 'lucide-react';
import { TableVirtuoso } from 'react-virtuoso';
import './DataExplorer.css';

interface DataExplorerProps {
    datasetId: string;
    onLoadComplete?: () => void;
}

interface PreviewData {
    columns: string[];
    rows: any[][];
    types: Record<string, string>;
}

interface DataQualityReport {
    overall_score: number;
    issues: any[];
    summary: string;
    row_count: number;
}

export function DataExplorer({ datasetId, onLoadComplete }: DataExplorerProps) {
    const [preview, setPreview] = useState<PreviewData | null>(null);
    const [qualityReport, setQualityReport] = useState<DataQualityReport | null>(null);
    const [loading, setLoading] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sortCol, setSortCol] = useState<string>('');
    const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
    const [filterText, setFilterText] = useState('');

    useEffect(() => {
        if (datasetId) loadData();
    }, [datasetId]);

    const loadData = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await invoke<any>('call_python_backend', {
                command: 'preview_dataset',
                payload: { dataset_id: datasetId, limit: 5000 } // Up to 5000 rows preview
            });
            if (res.preview) {
                setPreview(res.preview);
            }
            checkQuality();
            onLoadComplete?.();
        } catch (err: any) {
            setError(`Failed to load data: ${err}`);
        } finally {
            setLoading(false);
        }
    };

    const checkQuality = async () => {
        setAnalyzing(true);
        try {
            const res = await invoke<any>('call_python_backend', {
                command: 'check_data_quality',
                payload: { dataset_id: datasetId }
            });
            setQualityReport(res);
        } catch (err) {
            console.warn("Quality check failed", err);
        } finally {
            setAnalyzing(false);
        }
    };

    const getProcessedRows = useCallback(() => {
        if (!preview) return [];
        let rows = [...preview.rows];
        if (filterText) {
            const lowerFilter = filterText.toLowerCase();
            rows = rows.filter(row =>
                row.some(cell => String(cell).toLowerCase().includes(lowerFilter))
            );
        }
        if (sortCol) {
            const colIndex = preview.columns.indexOf(sortCol);
            if (colIndex > -1) {
                rows.sort((a, b) => {
                    const valA = a[colIndex];
                    const valB = b[colIndex];
                    if (valA === valB) return 0;
                    if (valA === null) return 1;
                    if (valB === null) return -1;
                    const comparison = valA < valB ? -1 : 1;
                    return sortDir === 'asc' ? comparison : -comparison;
                });
            }
        }
        return rows;
    }, [preview, filterText, sortCol, sortDir]);

    const handleSort = (col: string) => {
        if (sortCol === col) setSortDir(prev => prev === 'asc' ? 'desc' : 'asc');
        else { setSortCol(col); setSortDir('asc'); }
    };

    if (loading) return (
        <div className="explorer-loading">
            <Loader className="spin" size={32} />
            <p>Loading dataset preview...</p>
        </div>
    );
    if (error) return (
        <div className="explorer-error">
            <AlertCircle size={32} />
            <p>{error}</p>
            <button onClick={loadData}>Retry</button>
        </div>
    );
    if (!preview) return null;

    const processedRows = getProcessedRows();

    const VirtuosoTableComponents = {
        Table: (props: any) => <table {...props} className="data-table" />,
        TableHead: React.forwardRef((props, ref) => <thead {...props} ref={ref} />),
        TableBody: React.forwardRef((props, ref) => <tbody {...props} ref={ref} />),
        TableRow: ({ item: row, ...props }: any) => <tr {...props} />,
    };

    return (
        <div className="data-explorer-container">
            <div className="explorer-toolbar">
                <div className="search-box">
                    <Search size={16} />
                    <input
                        type="text"
                        placeholder="Search data..."
                        value={filterText}
                        onChange={(e) => setFilterText(e.target.value)}
                    />
                </div>
                <div className="toolbar-actions">
                    <span className="row-count">{processedRows.length.toLocaleString()} rows</span>
                    <button className="icon-btn" onClick={loadData} title="Refresh">
                        <RefreshCw size={16} />
                    </button>
                </div>
            </div>

            {qualityReport && (
                <div className={`quality-banner ${qualityReport.overall_score > 80 ? 'good' : 'warning'}`}>
                    <div className="score-circle">{qualityReport.overall_score}</div>
                    <div className="score-info">
                        <h4>Data Quality Score</h4>
                        <p>{qualityReport.summary}</p>
                    </div>
                    {analyzing && <Loader size={16} className="spin" />}
                </div>
            )}

            <div style={{ height: '600px', width: '100%' }}>
                <TableVirtuoso
                    data={processedRows}
                    components={VirtuosoTableComponents}
                    fixedHeaderContent={() => (
                        <tr>
                            {preview.columns.map(col => (
                                <th key={col} onClick={() => handleSort(col)}>
                                    <div className="th-content">
                                        <span>{col}</span>
                                        {sortCol === col ? (
                                            sortDir === 'asc' ? <ArrowUp size={14} /> : <ArrowDown size={14} />
                                        ) : (
                                            <ArrowUpDown size={14} className="sort-hint" />
                                        )}
                                    </div>
                                    <div className="dtype-badge">{preview.types[col]}</div>
                                </th>
                            ))}
                        </tr>
                    )}
                    itemContent={(index, row) => (
                        <>
                            {row.map((cell: any, cellIndex: number) => (
                                <td key={cellIndex}>
                                    {cell === null ? <span className="null-val">null</span> : String(cell)}
                                </td>
                            ))}
                        </>
                    )}
                />
            </div>
            {processedRows.length === 0 && (
                <div className="no-results">
                    <Database size={24} />
                    <p>No matching rows found</p>
                </div>
            )}
        </div>
    );
}