import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { 
    useReactTable, 
    getCoreRowModel, 
    getSortedRowModel, 
    flexRender,
    createColumnHelper,
    SortingState
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { clientDataEngine } from '../../services/ClientDataEngine';
import './DataExplorer.css';

interface DataExplorerProps {
    datasetId: string;
}

export const DataExplorer: React.FC<DataExplorerProps> = ({ datasetId }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dataLoaded, setDataLoaded] = useState(false);
    
    // We use a simple forceUpdate pattern to refresh when ClientEngine changes
    const [, setForceUpdate] = useState(0); 

    // --- 1. Fetch Data ---
    useEffect(() => {
        if (!datasetId) return;

        let mounted = true;
        const fetchDataset = async () => {
            setLoading(true);
            setError(null);
            setDataLoaded(false);
            try {
                const res: any = await invoke('call_python_backend', {
                    command: 'get_dataset_full',
                    payload: { dataset_id: datasetId, row_limit: 50000 } 
                });

                if (!mounted) return;

                clientDataEngine.loadDataset(
                    res.data,
                    res.schema,
                    res.column_stats,
                    res.total_rows
                );

                setDataLoaded(true);
            } catch (err: any) {
                if (mounted) setError(err.message || String(err));
            } finally {
                if (mounted) setLoading(false);
            }
        };

        fetchDataset();
        return () => { mounted = false; };
    }, [datasetId]);

    // --- 2. Prepare Data & Columns for TanStack Table ---
    
    // Get data from engine
    const data = useMemo(() => {
        return dataLoaded ? clientDataEngine.getFilteredData() : [];
    }, [dataLoaded, clientDataEngine.getFilteredData()]); // Re-run if engine data changes

    // Generate Columns
    const columns = useMemo(() => {
        if (!dataLoaded) return [];
        const schema = clientDataEngine.getSchema();
        const columnHelper = createColumnHelper<any>();

        return Object.values(schema).map(col => 
            columnHelper.accessor(row => row[col.name], {
                id: col.name,
                header: col.name,
                cell: info => info.getValue(),
            })
        );
    }, [dataLoaded]);

    // --- 3. Initialize TanStack Table (v8) ---
    const [sorting, setSorting] = useState<SortingState>([]);

    const table = useReactTable({
        data,
        columns,
        state: {
            sorting,
        },
        onSortingChange: setSorting,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        // This ensures the table doesn't scan deep objects, fixing the JSON crash
        defaultColumn: {
            size: 150, // Default width
        },
    });

    // --- 4. TanStack Virtual (The Scrolling Magic) ---
    const tableContainerRef = useRef<HTMLDivElement>(null);
    const { rows } = table.getRowModel();
    
    const rowVirtualizer = useVirtualizer({
        count: rows.length,
        getScrollElement: () => tableContainerRef.current,
        estimateSize: () => 35, // Estimate row height (35px)
        overscan: 10, // Render 10 rows outside view for smoothness
    });

    // --- Render ---
    if (loading) return <div className="explorer-loading">Loading dataset...</div>;
    if (error) return <div className="explorer-error">Error: {error}</div>;
    if (!dataLoaded) return <div className="explorer-empty">No data selected</div>;

    return (
        <div className="data-explorer-container">
            <div className="explorer-stats">
                <span className="badge">{rows.length.toLocaleString()} rows</span>
                <span className="badge">{columns.length} columns</span>
            </div>

            {/* Scrollable Container */}
            <div 
                className="table-wrapper" 
                ref={tableContainerRef}
                style={{ overflow: 'auto', height: '100%' }}
            >
                {/* The Virtualizer creates a massive div to fake the scrollbar height */}
                <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, width: '100%', position: 'relative' }}>
                    <table className="excel-grid">
                        <thead className="sticky-header">
                            {table.getHeaderGroups().map(headerGroup => (
                                <tr key={headerGroup.id} className="tr header-row">
                                    {headerGroup.headers.map(header => (
                                        <th 
                                            key={header.id} 
                                            className="th"
                                            style={{ width: header.getSize() }}
                                            onClick={header.column.getToggleSortingHandler()}
                                        >
                                            <div className="th-content">
                                                {flexRender(header.column.columnDef.header, header.getContext())}
                                                <span>
                                                    {{
                                                        asc: ' 🔼',
                                                        desc: ' 🔽',
                                                    }[header.column.getIsSorted() as string] ?? null}
                                                </span>
                                            </div>
                                        </th>
                                    ))}
                                </tr>
                            ))}
                        </thead>

                        <tbody>
                            {rowVirtualizer.getVirtualItems().map(virtualRow => {
                                const row = rows[virtualRow.index];
                                return (
                                    <tr 
                                        key={row.id} 
                                        className="tr"
                                        style={{
                                            height: `${virtualRow.size}px`,
                                            transform: `translateY(${virtualRow.start}px)`,
                                            position: 'absolute', // Absolute positioning is key for virtualizer
                                            top: 0,
                                            left: 0,
                                            width: '100%',
                                        }}
                                    >
                                        {row.getVisibleCells().map(cell => (
                                            <td 
                                                key={cell.id} 
                                                className="td"
                                                style={{ width: cell.column.getSize() }}
                                                title={String(cell.getValue())}
                                            >
                                                {cell.getValue() === null ? 
                                                    <span className="null-cell">null</span> : 
                                                    String(cell.getValue())
                                                }
                                            </td>
                                        ))}
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};