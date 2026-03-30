import React, { useMemo, useEffect } from 'react';
import ReactFlow, {
    Background,
    Controls,
    MiniMap,
    MarkerType,
    Handle,
    Position,
    useNodesState,
    useEdgesState,
    Node,
    Edge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Database } from 'lucide-react';
import dagre from 'dagre';

// ----- Custom Node: Database Table (unchanged, but ensure columns are rendered) -----
const TableNode = ({ data }: any) => (
    <div style={{
        background: 'white',
        border: '1px solid #cbd5e1',
        borderRadius: '8px',
        minWidth: '220px',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        overflow: 'hidden'
    }}>
        <div style={{
            background: 'linear-gradient(135deg, #4f46e5 0%, #3730a3 100%)',
            color: 'white',
            padding: '10px 12px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            fontWeight: '600'
        }}>
            <Database size={14} /> {data.label}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
            {data.columns && data.columns.map((col: any, index: number) => (
                <div
                    key={col.name}
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '6px 12px',
                        borderBottom: index < data.columns.length - 1 ? '1px solid #f1f5f9' : 'none',
                        position: 'relative',
                        fontSize: '12px',
                        color: '#334155'
                    }}
                >
                    <Handle
                        type="target"
                        position={Position.Left}
                        id={`target-${col.name}`}
                        style={{ top: '50%', background: '#94a3b8', width: '6px', height: '6px', border: 'none' }}
                    />
                    <span style={{ fontWeight: 500 }}>{col.name}</span>
                    <span style={{
                        fontSize: '10px',
                        background: '#f1f5f9',
                        color: '#64748b',
                        padding: '2px 6px',
                        borderRadius: '4px',
                        fontFamily: 'monospace'
                    }}>
                        {col.type || 'varchar'}
                    </span>
                    <Handle
                        type="source"
                        position={Position.Right}
                        id={`source-${col.name}`}
                        style={{ top: '50%', background: '#4f46e5', width: '6px', height: '6px', border: 'none' }}
                    />
                </div>
            ))}
        </div>
        <div style={{
            background: '#f8fafc',
            padding: '6px 12px',
            fontSize: '11px',
            color: '#94a3b8',
            borderTop: '1px solid #e2e8f0',
            textAlign: 'right'
        }}>
            {data.rows?.toLocaleString()} rows
        </div>
    </div>
);

const nodeTypes = { table: TableNode };

// ----- Layout Helper (dagre) -----
const getLayoutedElements = (
    nodes: Node[],
    edges: Edge[],
    direction: 'LR' | 'TB' = 'LR'
): { nodes: Node[]; edges: Edge[] } => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));
    dagreGraph.setGraph({ rankdir: direction, nodesep: 80, ranksep: 120 });

    nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: 220, height: 90 });
    });
    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
    });
    dagre.layout(dagreGraph);

    const layoutedNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        return {
            ...node,
            position: {
                x: nodeWithPosition.x - 110,
                y: nodeWithPosition.y - 45,
            },
        };
    });
    return { nodes: layoutedNodes, edges };
};

// ----- Main Canvas Component -----
interface ERDCanvasProps {
    datasets: any[];
    relationships: any[];
}

export const ERDCanvas: React.FC<ERDCanvasProps> = ({ datasets, relationships }) => {
    // Build nodes with columns (use memo to avoid recalculating on every render)
    const rawNodes: Node[] = useMemo(() => {
        return datasets.map((ds) => {
            const displayColumns = ds.columns && ds.columns.length > 0
                ? ds.columns
                : [
                    { name: 'id', type: 'integer' },
                    { name: 'user_id', type: 'varchar' },
                    { name: 'status', type: 'boolean' }
                ];
            return {
                id: ds.id,
                type: 'table',
                position: { x: 0, y: 0 },
                data: {
                    label: ds.name,
                    rows: ds.row_count,
                    columns: displayColumns
                }
            };
        });
    }, [datasets]);

    // Build edges with proper source/target handles (use memo)
    const rawEdges: Edge[] = useMemo(() => {
        return relationships
            .map((rel, index) => {
                const sourceDataset = datasets.find(d => d.id === rel.anchor_dataset);
                const targetDataset = datasets.find(d => d.id === rel.new_dataset);
                if (!sourceDataset || !targetDataset) return null;
                return {
                    id: `edge-${index}`,
                    source: sourceDataset.id,
                    target: targetDataset.id,
                    sourceHandle: `source-${rel.anchor_column}`,
                    targetHandle: `target-${rel.new_column}`,
                    animated: true,
                    style: { stroke: '#4f46e5', strokeWidth: 2 },
                    markerEnd: { type: MarkerType.ArrowClosed, color: '#4f46e5' },
                };
            })
            .filter(Boolean) as Edge[];
    }, [relationships, datasets]);

    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    // Layout and set nodes/edges when rawNodes/rawEdges change
    useEffect(() => {
        if (rawNodes.length === 0) return;
        const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(rawNodes, rawEdges, 'LR');
        setNodes(layoutedNodes);
        setEdges(layoutedEdges);
    }, [rawNodes, rawEdges, setNodes, setEdges]);

    return (
        <div style={{ width: '100%', height: '600px', border: '1px solid #e2e8f0', borderRadius: '8px', background: '#f8fafc' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                fitView
                attributionPosition="bottom-right"
            >
                <Background color="#cbd5e1" gap={16} />
                <Controls />
                <MiniMap nodeColor="#4f46e5" maskColor="rgba(241, 245, 249, 0.7)" />
            </ReactFlow>
        </div>
    );
};