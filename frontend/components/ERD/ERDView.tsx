import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
    Link,
    Loader,
    AlertCircle,
    Eye,
    Code,
    Lightbulb,
    CheckCircle,
    XCircle,
    AlertTriangle,
    GitBranch,
    Layers,
    Database,
    ChevronRight,
    Copy,
    X,
    Network
} from 'lucide-react';
import './ERDView.css';
import { ERDCanvas } from './ERDCanvas';

// Types (same as before, with minor additions)
interface Dataset {
    id: string;
    name: string;
    row_count: number;
    column_count: number;
    columns?: { name: string; type: string }[];
}

interface Relationship {
    anchor_dataset: string;
    anchor_column: string;
    new_dataset: string;
    new_column: string;
    confidence_score: number;
    reasons: string[];
    metadata?: {
        overlap_percentage?: number;
        cardinality?: string;
        matched_values?: number;
        total_values?: number;
    };
    relationship_type?: 'single_column' | 'composite';
    pk_quality_score?: number;
}

interface JoinPreview {
    preview: {
        columns: string[];
        rows: any[];
    };
    quality: {
        total_new_rows: number;
        matched_rows: number;
        orphan_rows: number;
        orphan_percentage: number;
        match_percentage: number;
        anchor_key_unique: boolean;
        anchor_duplicate_keys: number;
        cardinality: string;
    };
    warnings: string[];
}

interface SchemaImprovement {
    type: string;
    column: string;
    reason: string;
    confidence: string;
    action: string;
    metadata?: any;
}

interface Props {
    datasets: Dataset[];
    selectedDatasetId?: string;
}

const ERDView: React.FC<Props> = ({ datasets, selectedDatasetId }) => {
    // Selection state
    const [anchorDatasetId, setAnchorDatasetId] = useState<string>(selectedDatasetId || '');
    const [newDatasetId, setNewDatasetId] = useState<string[]>([]);

    // Results state
    const [relationships, setRelationships] = useState<Relationship[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');

    // Advanced features state
    const [previewData, setPreviewData] = useState<JoinPreview | null>(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [selectedRelationship, setSelectedRelationship] = useState<Relationship | null>(null);

    const [sqlCode, setSqlCode] = useState<string>('');
    const [showSqlModal, setShowSqlModal] = useState(false);

    const [improvements, setImprovements] = useState<SchemaImprovement[]>([]);
    const [improvementsLoading, setImprovementsLoading] = useState(false);

    const [lineagePaths, setLineagePaths] = useState<string[][]>([]);
    const [lineageLoading, setLineageLoading] = useState(false);

    const [includeComposites, setIncludeComposites] = useState(true);
    const [activeTab, setActiveTab] = useState<'visualizer' | 'relationships' | 'improvements' | 'lineage'>('visualizer');

    const [enrichedDatasets, setEnrichedDatasets] = useState<Dataset[]>(datasets);


    useEffect(() => {
        const fetchAllColumns = async () => {
            const updatedDatasets = await Promise.all(datasets.map(async (ds) => {
                try {

                    const response = await invoke<any>('call_python_backend', {
                        command: 'get_schema', // <--- CHANGE THIS to your actual Python command name
                        payload: { dataset_id: ds.id }
                    });

                    // Map the Python response into the {name, type} format React Flow expects
                    const fetchedColumns = response.columns ? response.columns.map((c: any) => ({
                        // Handle both string arrays ["id", "name"] and object arrays [{name: "id", type: "int"}]
                        name: typeof c === 'string' ? c : c.name,
                        type: c.type || 'varchar'
                    })) : [];

                    // Return the dataset with the new columns attached!
                    return { ...ds, columns: fetchedColumns };
                } catch (err) {
                    console.error(`Failed to fetch columns for dataset ${ds.name}:`, err);
                    return ds; // Fallback to the original if it fails
                }
            }));

            setEnrichedDatasets(updatedDatasets);
        };

        if (datasets.length > 0) {
            fetchAllColumns();
        }
    }, [datasets]);

    useEffect(() => {
        setRelationships([]);
        setPreviewData(null);
        setSelectedRelationship(null);
        setSqlCode('');
        setImprovements([]);
        setLineagePaths([]);
    }, [anchorDatasetId, newDatasetId]);

    // ========== Core Functions ==========

    const findRelationships = async () => {
        // 1. Check if the array is empty instead of checking a single string
        if (!anchorDatasetId || newDatasetId.length === 0) {
            setError('Please select an anchor dataset and at least one dataset to compare');
            return;
        }

        setLoading(true);
        setError('');

        try {
            let allFoundRelationships: Relationship[] = [];

            // 2. Loop through every selected dataset ID
            for (const targetId of newDatasetId) {
                const response = await invoke<any>('call_python_backend', {
                    command: 'analyze_erd_relationships',
                    payload: {
                        anchor_dataset_id: anchorDatasetId,
                        new_dataset_id: targetId, // <--- Send ONE string ID at a time
                        include_composites: includeComposites
                    }
                });

                // 3. Merge the new relationships into our master list
                if (response.relationships) {
                    allFoundRelationships = [...allFoundRelationships, ...response.relationships];
                }
            }

            // 4. Update the UI with the combined results
            setRelationships(allFoundRelationships);

        } catch (err: any) {
            setError(err.message || 'Failed to analyze relationships');
        } finally {
            setLoading(false);
        }
    };

    const previewJoin = async (rel: Relationship) => {
        setSelectedRelationship(rel);
        setPreviewLoading(true);
        setPreviewData(null);

        try {
            const response = await invoke<JoinPreview>('call_python_backend', {
                command: 'preview_join',
                payload: {
                    anchor_dataset_id: anchorDatasetId,
                    new_dataset_id: newDatasetId,
                    anchor_column: rel.anchor_column,
                    new_column: rel.new_column,
                    limit: 20
                }
            });
            setPreviewData(response);
        } catch (err: any) {
            setError(err.message || 'Failed to preview join');
        } finally {
            setPreviewLoading(false);
        }
    };

    const generateSQL = async (rels: Relationship[], joinType: string = 'LEFT') => {
        try {
            const response = await invoke<any>('call_python_backend', {
                command: 'generate_join_sql',
                payload: {
                    relationships: rels,
                    join_type: joinType
                }
            });
            setSqlCode(response.sql_query);
            setShowSqlModal(true);
        } catch (err: any) {
            setError(err.message || 'Failed to generate SQL');
        }
    };

    const getSuggestedImprovements = async (datasetId: string) => {
        setImprovementsLoading(true);
        setError('');
        try {
            const response = await invoke<any>('call_python_backend', {
                command: 'suggest_schema_improvements',
                payload: { dataset_id: datasetId }
            });
            setImprovements(response.suggestions || []);
        } catch (err: any) {
            setError(err.message || 'Failed to get suggestions');
        } finally {
            setImprovementsLoading(false);
        }
    };

    const findLineage = async () => {
        if (!anchorDatasetId || !newDatasetId) {
            setError('Please select both datasets');
            return;
        }
        setLineageLoading(true);
        setError('');
        try {
            const response = await invoke<any>('call_python_backend', {
                command: 'find_data_lineage',
                payload: {
                    start_dataset_id: anchorDatasetId,
                    end_dataset_id: newDatasetId
                }
            });
            setLineagePaths(response.paths || []);
        } catch (err: any) {
            setError(err.message || 'Failed to find lineage');
        } finally {
            setLineageLoading(false);
        }
    };

    // ========== UI Helpers ==========

    const getConfidenceColor = (score: number): string => {
        if (score > 80) return '#10b981';
        if (score > 50) return '#f59e0b';
        return '#ef4444';
    };

    const getCardinalityIcon = (cardinality?: string) => {
        switch (cardinality) {
            case '1:1': return '⟷';
            case '1:N': return '→';
            case 'N:1': return '←';
            case 'M:N': return '⟺';
            default: return '→';
        }
    };

    // ========== Render ==========

    return (
        <div className="erd-view">
            {/* Header */}
            <div className="erd-header">
                <div>
                    <h2>
                        <Link size={24} />
                        Entity Relationship Analyzer
                    </h2>
                    <p>Discover relationships, preview joins, and optimize your schema</p>
                </div>
            </div>

            {/* Tab Navigation */}
            <div className="erd-tabs">
                <button
                    className={activeTab === 'relationships' ? 'active' : ''}
                    onClick={() => setActiveTab('relationships')}
                >
                    <Link size={16} />
                    Relationships
                </button>
                <button
                    className={activeTab === 'improvements' ? 'active' : ''}
                    onClick={() => setActiveTab('improvements')}
                >
                    <Lightbulb size={16} />
                    Improvements
                </button>
                <button
                    className={activeTab === 'lineage' ? 'active' : ''}
                    onClick={() => setActiveTab('lineage')}
                >
                    <GitBranch size={16} />
                    Data Lineage
                </button>
                <button
                    className={activeTab === 'visualizer' ? 'active' : ''}
                    onClick={() => setActiveTab('visualizer')}
                >
                    <Network size={16} />
                    Visual ERD
                </button>
            </div>

            {/* Controls */}
            <div className="erd-controls">
                <div className="control-row">
                    <div className="control-group">
                        <label>Anchor Dataset (PK side):</label>
                        <select
                            value={anchorDatasetId}
                            onChange={(e) => setAnchorDatasetId(e.target.value)}
                            className="column-select"
                        >
                            <option value="">Select anchor dataset</option>
                            {datasets.map(d => (
                                <option key={d.id} value={d.id}>
                                    {d.name} ({d.row_count.toLocaleString()} rows)
                                </option>
                            ))}
                        </select>
                    </div>

                    <div className="control-group">
                        <label>Datasets to compare (FK side):</label>

                        {/* Scrollable Checkbox Container */}
                        <div style={{
                            border: '1px solid #cbd5e1',
                            borderRadius: '6px',
                            padding: '10px',
                            maxHeight: '160px',
                            overflowY: 'auto',
                            background: 'white',
                            display: 'flex',
                            flexDirection: 'column',
                            gap: '8px'
                        }}>
                            {datasets
                                .filter(d => d.id !== anchorDatasetId)
                                .map(d => (
                                    <label key={d.id} style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
                                        <input
                                            type="checkbox"
                                            checked={newDatasetId.includes(d.id)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    // Add the ID to the array
                                                    setNewDatasetId([...newDatasetId, d.id]);
                                                } else {
                                                    // Remove the ID from the array
                                                    setNewDatasetId(newDatasetId.filter(id => id !== d.id));
                                                }
                                            }}
                                            style={{ cursor: 'pointer', width: '16px', height: '16px', accentColor: '#4f46e5' }}
                                        />
                                        <span style={{ fontSize: '14px', color: '#334155', fontWeight: 500 }}>
                                            {d.name}
                                            <span style={{ color: '#94a3b8', fontSize: '12px', fontWeight: 400, marginLeft: '6px' }}>
                                                ({d.row_count.toLocaleString()} rows)
                                            </span>
                                        </span>
                                    </label>
                                ))
                            }
                            {/* Fallback if no other datasets exist */}
                            {datasets.filter(d => d.id !== anchorDatasetId).length === 0 && (
                                <div style={{ fontSize: '13px', color: '#94a3b8', fontStyle: 'italic', textAlign: 'center' }}>
                                    Upload more datasets to compare.
                                </div>
                            )}
                        </div>

                        {/* Selection Counter */}
                        <div style={{ fontSize: '12px', color: '#64748b', marginTop: '6px', fontWeight: 500 }}>
                            {newDatasetId.length} dataset(s) selected
                        </div>
                    </div>
                </div>

                <div className="control-actions">
                    <label className="checkbox-label">
                        <input
                            type="checkbox"
                            checked={includeComposites}
                            onChange={(e) => setIncludeComposites(e.target.checked)}
                        />
                        Include composite keys
                    </label>

                    {activeTab === 'relationships' && (
                        <>
                            <button
                                onClick={findRelationships}
                                disabled={loading || !anchorDatasetId || !newDatasetId}
                                className="execute-btn primary"
                            >
                                {loading ? <Loader size={16} className="spin" /> : <Link size={16} />}
                                {loading ? 'Analyzing...' : 'Find Relationships'}
                            </button>

                            {relationships.length > 0 && (
                                <button
                                    onClick={() => generateSQL(relationships, 'LEFT')}
                                    className="execute-btn secondary"
                                >
                                    <Code size={16} /> Generate SQL
                                </button>
                            )}
                        </>
                    )}

                    {activeTab === 'improvements' && anchorDatasetId && (
                        <button
                            onClick={() => getSuggestedImprovements(anchorDatasetId)}
                            disabled={improvementsLoading}
                            className="execute-btn primary"
                        >
                            {improvementsLoading ? <Loader size={16} className="spin" /> : <Lightbulb size={16} />}
                            {improvementsLoading ? 'Analyzing...' : 'Get Suggestions'}
                        </button>
                    )}

                    {activeTab === 'lineage' && (
                        <button
                            onClick={findLineage}
                            disabled={lineageLoading || !anchorDatasetId || !newDatasetId}
                            className="execute-btn primary"
                        >
                            {lineageLoading ? <Loader size={16} className="spin" /> : <GitBranch size={16} />}
                            {lineageLoading ? 'Finding paths...' : 'Find Paths'}
                        </button>
                    )}
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="error-message">
                    <AlertCircle size={16} />
                    <strong>Error:</strong> {error}
                </div>
            )}

            {/* Tab Content */}
            {activeTab === 'relationships' && (
                <div className="tab-content">
                    {loading && (
                        <div className="loading-center">
                            <Loader size={32} className="spin" />
                            <p>Analyzing datasets...</p>
                        </div>
                    )}

                    {!loading && relationships.length > 0 && (
                        <div className="erd-results">
                            <h3>Suggested Relationships ({relationships.length})</h3>
                            <div className="relationships-list">
                                {relationships.map((rel, idx) => (
                                    <div key={idx} className="relationship-item">
                                        <div className="relationship-header">
                                            <div className="relationship-main">
                                                <span className="dataset-badge anchor">
                                                    <Database size={12} />
                                                    {rel.anchor_dataset}
                                                </span>
                                                <span className="column-name">{rel.anchor_column}</span>
                                                <span className="arrow">{getCardinalityIcon(rel.metadata?.cardinality)}</span>
                                                <span className="dataset-badge new">
                                                    <Database size={12} />
                                                    {rel.new_dataset}
                                                </span>
                                                <span className="column-name">{rel.new_column}</span>
                                            </div>

                                            <div className="relationship-actions">
                                                <div
                                                    className="confidence-badge"
                                                    style={{ backgroundColor: getConfidenceColor(rel.confidence_score) }}
                                                >
                                                    {rel.confidence_score}%
                                                </div>

                                                <button
                                                    className="icon-btn"
                                                    onClick={() => previewJoin(rel)}
                                                    title="Preview Join"
                                                >
                                                    <Eye size={16} />
                                                </button>

                                                <button
                                                    className="icon-btn"
                                                    onClick={() => generateSQL([rel], 'LEFT')}
                                                    title="Generate SQL"
                                                >
                                                    <Code size={16} />
                                                </button>
                                            </div>
                                        </div>

                                        <div className="relationship-meta">
                                            {rel.relationship_type === 'composite' && (
                                                <span className="meta-badge composite">Composite Key</span>
                                            )}
                                            {rel.metadata?.cardinality && (
                                                <span className="meta-badge">{rel.metadata.cardinality}</span>
                                            )}
                                            {rel.metadata?.overlap_percentage && (
                                                <span className="meta-badge">
                                                    {rel.metadata.overlap_percentage.toFixed(1)}% overlap
                                                </span>
                                            )}
                                        </div>

                                        <div className="relationship-reasons">
                                            {rel.reasons.map((reason, i) => (
                                                <span key={i} className="reason-tag">{reason}</span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {!loading && relationships.length === 0 && anchorDatasetId && newDatasetId && (
                        <div className="empty-state">
                            <Link size={48} strokeWidth={1.5} />
                            <p>No relationships found between these datasets</p>
                            <small>Try selecting different datasets or enable composite keys</small>
                        </div>
                    )}

                    {!loading && !anchorDatasetId && !newDatasetId && (
                        <div className="empty-state">
                            <Link size={48} strokeWidth={1.5} />
                            <p>Select two datasets to start analyzing relationships</p>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'improvements' && (
                <div className="tab-content">
                    {improvementsLoading && (
                        <div className="loading-center">
                            <Loader size={32} className="spin" />
                            <p>Analyzing schema...</p>
                        </div>
                    )}

                    {!improvementsLoading && improvements.length > 0 && (
                        <div className="improvements-list">
                            <h3>Schema Improvement Suggestions</h3>
                            {improvements.map((imp, idx) => (
                                <div key={idx} className="improvement-item">
                                    <div className="improvement-header">
                                        <Lightbulb size={18} color="#f59e0b" />
                                        <strong>{imp.action}</strong>
                                        <span className={`confidence-tag ${imp.confidence}`}>
                                            {imp.confidence}
                                        </span>
                                    </div>
                                    <p className="improvement-reason">{imp.reason}</p>
                                    <div className="improvement-type">{imp.type}</div>
                                </div>
                            ))}
                        </div>
                    )}

                    {!improvementsLoading && improvements.length === 0 && (
                        <div className="empty-state">
                            <Lightbulb size={48} strokeWidth={1.5} />
                            <p>No improvement suggestions for this dataset</p>
                            <small>Click "Get Suggestions" to analyze</small>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'lineage' && (
                <div className="tab-content">
                    {lineageLoading && (
                        <div className="loading-center">
                            <Loader size={32} className="spin" />
                            <p>Tracing data lineage...</p>
                        </div>
                    )}

                    {!lineageLoading && lineagePaths.length > 0 && (
                        <div className="lineage-results">
                            <h3>Data Lineage Paths ({lineagePaths.length})</h3>
                            {lineagePaths.map((path, idx) => (
                                <div key={idx} className="lineage-path">
                                    <span className="path-number">Path {idx + 1}:</span>
                                    {path.map((node, nodeIdx) => (
                                        <React.Fragment key={nodeIdx}>
                                            <span className="path-node">{node}</span>
                                            {nodeIdx < path.length - 1 && (
                                                <ChevronRight size={16} className="path-arrow" />
                                            )}
                                        </React.Fragment>
                                    ))}
                                </div>
                            ))}
                        </div>
                    )}

                    {!lineageLoading && lineagePaths.length === 0 && anchorDatasetId && newDatasetId && (
                        <div className="empty-state">
                            <GitBranch size={48} strokeWidth={1.5} />
                            <p>No lineage path found between these datasets</p>
                            <small>They may not be related through any common columns</small>
                        </div>
                    )}

                    {!lineageLoading && (!anchorDatasetId || !newDatasetId) && (
                        <div className="empty-state">
                            <GitBranch size={48} strokeWidth={1.5} />
                            <p>Select two datasets to trace lineage</p>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'visualizer' && (
                <div className="tab-content">

                    <div style={{ marginBottom: '16px' }}>
                        <h3>Interactive Schema Visualization</h3>
                        <p style={{ color: '#64748b', fontSize: '13px' }}>
                            Drag tables to organize them. Lines indicate relationships found by the backend.
                            Use the controls to zoom and pan.
                        </p>
                    </div>

                    {/* Render the React Flow Canvas! */}
                    <ERDCanvas datasets={datasets} relationships={relationships} />
                </div>
            )}

            {/* Join Preview Modal */}
            {selectedRelationship && (
                <div className="modal-overlay" onClick={() => setSelectedRelationship(null)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>
                                <Eye size={20} />
                                Join Preview: {selectedRelationship.anchor_column} → {selectedRelationship.new_column}
                            </h3>
                            <button className="modal-close" onClick={() => setSelectedRelationship(null)}>
                                <X size={20} />
                            </button>
                        </div>

                        <div className="modal-body">
                            {previewLoading && (
                                <div className="loading-center">
                                    <Loader size={24} className="spin" />
                                    <p>Analyzing join...</p>
                                </div>
                            )}

                            {previewData && (
                                <>
                                    {/* Quality Metrics */}
                                    <div className="quality-metrics">
                                        <h4>Join Quality</h4>
                                        <div className="metrics-grid">
                                            <div className="metric">
                                                <CheckCircle size={16} color="#10b981" />
                                                <span>Matched: {previewData.quality.matched_rows} / {previewData.quality.total_new_rows}</span>
                                                <strong>{previewData.quality.match_percentage.toFixed(1)}%</strong>
                                            </div>
                                            <div className="metric">
                                                <XCircle size={16} color="#ef4444" />
                                                <span>Orphans: {previewData.quality.orphan_rows}</span>
                                                <strong>{previewData.quality.orphan_percentage.toFixed(1)}%</strong>
                                            </div>
                                            <div className="metric">
                                                <Layers size={16} color="#6366f1" />
                                                <span>Cardinality:</span>
                                                <strong>{previewData.quality.cardinality}</strong>
                                            </div>
                                        </div>

                                        {/* Warnings */}
                                        {previewData.warnings.length > 0 && (
                                            <div className="warnings-section">
                                                {previewData.warnings.map((warning, i) => (
                                                    <div key={i} className="warning-item">
                                                        <AlertTriangle size={14} />
                                                        {warning}
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    {/* Preview Table */}
                                    <div className="preview-table-wrapper">
                                        <h4>Sample Data (first 20 rows)</h4>
                                        <table className="preview-table">
                                            <thead>
                                                <tr>
                                                    {previewData.preview.columns.map((col) => (
                                                        <th key={col}>{col}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {previewData.preview.rows.map((row, i) => (
                                                    <tr key={i}>
                                                        {previewData.preview.columns.map((col) => (
                                                            <td key={col}>
                                                                {row[col] !== null && row[col] !== undefined
                                                                    ? String(row[col])
                                                                    : <span className="null-value">null</span>
                                                                }
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* SQL Modal */}
            {showSqlModal && (
                <div className="modal-overlay" onClick={() => setShowSqlModal(false)}>
                    <div className="modal-content sql-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3><Code size={20} /> Generated SQL</h3>
                            <button className="modal-close" onClick={() => setShowSqlModal(false)}>
                                <X size={20} />
                            </button>
                        </div>
                        <div className="modal-body">
                            <pre className="sql-code">{sqlCode}</pre>
                            <button
                                className="copy-btn"
                                onClick={() => {
                                    navigator.clipboard.writeText(sqlCode);
                                    alert('Copied to clipboard!');
                                }}
                            >
                                <Copy size={14} /> Copy to Clipboard
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ERDView;