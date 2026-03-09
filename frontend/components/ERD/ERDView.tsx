import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { Link, Loader, AlertCircle } from 'lucide-react';
import './ERDView.css';

interface Dataset {
    id: string;
    name: string;
    row_count: number;
    column_count: number;
}

interface Relationship {
    anchor_dataset: string;
    anchor_column: string;
    new_dataset: string;
    new_column: string;
    confidence_score: number;
    reasons: string[];
}

interface Props {
    datasets: Dataset[];
    selectedDatasetId?: string; // optional preselected anchor
}

const ERDView: React.FC<Props> = ({ datasets, selectedDatasetId }) => {
    const [anchorDatasetId, setAnchorDatasetId] = useState<string>(selectedDatasetId || '');
    const [newDatasetId, setNewDatasetId] = useState<string>('');
    const [relationships, setRelationships] = useState<Relationship[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string>('');

    // Clear results when either selection changes
    useEffect(() => {
        setRelationships([]);
    }, [anchorDatasetId, newDatasetId]);

    const findRelationships = async () => {
        if (!anchorDatasetId || !newDatasetId) {
            setError('Please select both anchor and new dataset');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const response = await invoke<any>('call_python_backend', {
                command: 'analyze_erd_relationships',
                payload: {
                    anchor_dataset_id: anchorDatasetId,
                    new_dataset_id: newDatasetId
                }
            });
            setRelationships(response.relationships || []);
        } catch (err: any) {
            setError(err.message || 'Failed to analyze relationships');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="erd-view">
            <h2>Entity Relationship Diagram</h2>
            <p>Select two datasets to find possible Primary Key / Foreign Key links.</p>

            <div className="erd-controls">
                <div className="control-group">
                    <label>Anchor Dataset (PK side):</label>
                    <select
                        value={anchorDatasetId}
                        onChange={(e) => setAnchorDatasetId(e.target.value)}
                        className="column-select"
                    >
                        <option value="">Select anchor dataset</option>
                        {datasets.map(d => (
                            <option key={d.id} value={d.id}>{d.name}</option>
                        ))}
                    </select>
                </div>

                <div className="control-group">
                    <label>Dataset to compare (FK side):</label>
                    <select
                        value={newDatasetId}
                        onChange={(e) => setNewDatasetId(e.target.value)}
                        className="column-select"
                    >
                        <option value="">Select dataset</option>
                        {datasets
                            .filter(d => d.id !== anchorDatasetId)
                            .map(d => (
                                <option key={d.id} value={d.id}>{d.name}</option>
                            ))
                        }
                    </select>
                </div>

                <button
                    onClick={findRelationships}
                    disabled={loading || !anchorDatasetId || !newDatasetId}
                    className="execute-btn"
                >
                    {loading ? <Loader size={16} className="spin" /> : <Link size={16} />}
                    {loading ? 'Analyzing...' : 'Find Relationships'}
                </button>
            </div>

            {error && (
                <div className="error-message">
                    <AlertCircle size={16} />
                    <strong>Error:</strong> {error}
                </div>
            )}

            {relationships.length > 0 && (
                <div className="erd-results">
                    <h3>Suggested Relationships</h3>
                    <div className="relationships-list">
                        {relationships.map((rel, idx) => (
                            <div key={idx} className="relationship-item">
                                <div className="relationship-header">
                                    <span className="anchor-col">{rel.anchor_dataset}.{rel.anchor_column}</span>
                                    <span className="arrow">→</span>
                                    <span className="new-col">{rel.new_dataset}.{rel.new_column}</span>
                                    <div
                                        className="confidence-badge"
                                        style={{
                                            backgroundColor:
                                                rel.confidence_score > 80 ? '#10b981' :
                                                rel.confidence_score > 50 ? '#f59e0b' : '#ef4444'
                                        }}
                                    >
                                        {rel.confidence_score}%
                                    </div>
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

            {relationships.length === 0 && !loading && !error && (
                <div className="empty-state">
                    <Link size={48} strokeWidth={1.5} />
                    <p>No relationships found. Try different datasets.</p>
                </div>
            )}
        </div>
    );
};

export default ERDView;