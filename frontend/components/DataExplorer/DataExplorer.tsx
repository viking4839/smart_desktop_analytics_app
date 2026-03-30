import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
    useReactTable, getCoreRowModel, getSortedRowModel, getFilteredRowModel,
    flexRender, createColumnHelper, SortingState, VisibilityState, ColumnFiltersState
} from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Search, RotateCcw, Settings, Download, ArrowUp, ArrowDown, Filter, Hash, Type, Calendar, LayoutPanelLeft } from 'lucide-react';
import { clientDataEngine } from '../../services/ClientDataEngine';
import './DataExplorer.css';

interface DataExplorerProps {
    datasetId: string;
}

// ---------- MEMOIZED ROW COMPONENT (prevents re-renders during scroll) ----------
const MemoizedRow = React.memo(({ row, virtualRow, measureElement }: any) => {
    return (
        <tr
            data-index={virtualRow.index}
            ref={measureElement}
            className="tr body-row"
            style={{
                transform: `translateY(${virtualRow.start}px)`,
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                display: 'flex', // flex ensures columns align with header/footer
            }}
        >
            {row.getVisibleCells().map((cell: any) => (
                <td key={cell.id} className="td body-cell" style={{ width: cell.column.getSize() }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
            ))}
        </tr>
    );
}, (prevProps, nextProps) => {
    // Only re-render if the row index changes or the data itself changes
    return prevProps.virtualRow.index === nextProps.virtualRow.index &&
           prevProps.row.original === nextProps.row.original;
});

export const DataExplorer: React.FC<DataExplorerProps> = ({ datasetId }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dataLoaded, setDataLoaded] = useState(false);
    const [globalSearch, setGlobalSearch] = useState('');
    const [showColumnMenu, setShowColumnMenu] = useState(false);

    // --- Fetch Data ---
    useEffect(() => {
        if (!datasetId) return;
        let mounted = true;
        const fetchDataset = async () => {
            setLoading(true); setError(null); setDataLoaded(false); setGlobalSearch('');
            try {
                const res: any = await invoke('call_python_backend', {
                    command: 'get_dataset_full', payload: { dataset_id: datasetId, row_limit: 50000 }
                });
                if (!mounted) return;
                clientDataEngine.loadDataset(res.data, res.schema, res.column_stats, res.total_rows);
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

    // --- Define Columns with Advanced Analytics ---
    const columns = useMemo(() => {
        if (!dataLoaded) return [];
        const schema = clientDataEngine.getSchema();
        const columnHelper = createColumnHelper<any>();

        return Object.values(schema).map(col => {
            const isNumeric = col.data_type === 'number' || col.data_type === 'float' || col.data_type === 'integer';
            const stats = clientDataEngine.getColumnStats(col.name) || {};
            const nullPct = (stats as any).null_percentage || 0;
            const validPct = 100 - nullPct;

            return columnHelper.accessor(row => row[col.name], {
                id: col.name,

                // HEADER: Data Quality Bar & Icon
                header: () => (
                    <div className="th-header-container">
                        <div className="quality-bar" title={`${validPct.toFixed(1)}% Valid, ${nullPct.toFixed(1)}% Null`}>
                            <div className="quality-valid" style={{ width: `${validPct}%` }} />
                            <div className="quality-null" style={{ width: `${nullPct}%` }} />
                        </div>
                        <div className="th-title">
                            {isNumeric ? <Hash size={14} className="type-icon" /> : col.data_type === 'date' ? <Calendar size={14} className="type-icon" /> : <Type size={14} className="type-icon" />}
                            <span className="th-name" title={col.name}>{col.name}</span>
                        </div>
                    </div>
                ),

                // CELL: Conditional Formatting (Data Bars)
                cell: info => {
                    const val = info.getValue();
                    if (val === null || val === undefined) return <span className="null-cell">null</span>;

                    if (isNumeric && stats.max) {
                        const pct = Math.max(0, Math.min(100, (Number(val) / (stats.max || 1)) * 100));
                        return (
                            <div className="data-bar-container">
                                <div className="data-bar-fill" style={{ width: `${pct}%` }} />
                                <span className="data-bar-text">{Number(val).toLocaleString()}</span>
                            </div>
                        );
                    }
                    return String(val);
                },

                // FOOTER: Quick Aggregations based on visible rows
                footer: info => {
                    if (!isNumeric) return <span className="footer-count">Count: {info.table.getFilteredRowModel().rows.length}</span>;

                    const rows = info.table.getFilteredRowModel().rows;
                    const sum = rows.reduce((acc, row) => acc + (Number(row.getValue(col.name)) || 0), 0);
                    const avg = rows.length > 0 ? sum / rows.length : 0;

                    return (
                        <div className="footer-agg">
                            <div>Σ: {sum.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                            <div className="footer-avg">x̄: {avg.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                        </div>
                    );
                },

                filterFn: isNumeric ? 'inNumberRange' : 'includesString',
            });
        });
    }, [dataLoaded]);

    const data = useMemo(() => dataLoaded ? clientDataEngine.getRawData() : [], [dataLoaded]);

    // --- TanStack Table ---
    const [sorting, setSorting] = useState<SortingState>([]);
    const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
    const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});

    const table = useReactTable({
        data,
        columns,
        state: { sorting, columnFilters, columnVisibility, globalFilter: globalSearch },
        onSortingChange: setSorting,
        onColumnFiltersChange: setColumnFilters,
        onColumnVisibilityChange: setColumnVisibility,
        onGlobalFilterChange: setGlobalSearch,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getFilteredRowModel: getFilteredRowModel(),
        defaultColumn: { size: 160 },
    });

    // Export CSV
    const exportToCSV = () => {
        const exportData = table.getFilteredRowModel().rows;
        if (exportData.length === 0) return;
        const visibleCols = table.getVisibleLeafColumns().map(c => c.id);
        const csvRows = [visibleCols.map(c => `"${c}"`).join(',')];

        exportData.forEach(row => {
            const values = visibleCols.map(col => {
                const val = row.getValue(col) === null ? '' : String(row.getValue(col));
                return `"${val.replace(/"/g, '""')}"`;
            });
            csvRows.push(values.join(','));
        });

        const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', `export_${datasetId}.csv`);
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
    };

    // --- Virtualizer ---
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    const { rows } = table.getRowModel();
    const rowVirtualizer = useVirtualizer({
        count: rows.length,
        getScrollElement: () => scrollContainerRef.current,
        estimateSize: () => 35,
        overscan: 10,
        measureElement: (element) => element?.getBoundingClientRect().height ?? 35,
    });

    // --- Drag-to-scroll (horizontal) – kept but now works with a single container ---
    const [isDragging, setIsDragging] = useState(false);

    const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
        const tag = (e.target as HTMLElement).tagName.toLowerCase();
        if (['input', 'button', 'label', 'select', 'textarea'].includes(tag)) return;

        const el = scrollContainerRef.current;
        if (!el) return;

        const startX = e.pageX;
        const startLeft = el.scrollLeft;
        let didDrag = false;

        setIsDragging(true);

        const onMouseMove = (mv: MouseEvent) => {
            const dx = mv.pageX - startX;
            if (Math.abs(dx) > 3) didDrag = true;
            el.scrollLeft = startLeft - dx;
        };

        const onMouseUp = () => {
            setIsDragging(false);
            document.removeEventListener('mousemove', onMouseMove);
            if (didDrag) {
                const suppressClick = (ce: MouseEvent) => ce.stopPropagation();
                document.addEventListener('click', suppressClick, { capture: true, once: true });
            }
        };

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp, { once: true });
    }, []);

    if (loading) return <div className="explorer-message"><RotateCcw className="spinner" /> Loading Data...</div>;
    if (error) return <div className="explorer-message error">⚠️ {error}</div>;
    if (!dataLoaded) return <div className="explorer-message"><LayoutPanelLeft /> Select a dataset</div>;

    return (
        <div className="data-explorer-container">
            {/* Toolbar (unchanged) */}
            <div className="data-toolbar">
                <div className="toolbar-left">
                    <div className="search-box">
                        <Search size={16} className="search-icon" />
                        <input
                            type="text"
                            className="search-input"
                            placeholder="Global search..."
                            value={globalSearch}
                            onChange={(e) => setGlobalSearch(e.target.value)}
                        />
                    </div>
                    <span className="stats-text">
                        Showing {table.getFilteredRowModel().rows.length.toLocaleString()} of {clientDataEngine.getTotalRows().toLocaleString()} rows
                    </span>
                </div>

                <div className="toolbar-right">
                    <button className="toolbar-btn" onClick={() => { setGlobalSearch(''); setColumnFilters([]); setSorting([]); }}>
                        <RotateCcw size={14} /> Reset
                    </button>

                    <div className="dropdown-container">
                        <button className="toolbar-btn" onClick={() => setShowColumnMenu(!showColumnMenu)}>
                            <Settings size={14} /> Columns ({table.getVisibleLeafColumns().length})
                        </button>
                        {showColumnMenu && (
                            <div className="column-dropdown">
                                <label className="col-toggle-all">
                                    <input type="checkbox" checked={table.getIsAllColumnsVisible()} onChange={table.getToggleAllColumnsVisibilityHandler()} />
                                    <strong>Toggle All</strong>
                                </label>
                                <div className="col-divider"></div>
                                {table.getAllLeafColumns().map(column => (
                                    <label key={column.id} className="col-toggle">
                                        <input type="checkbox" checked={column.getIsVisible()} onChange={column.getToggleVisibilityHandler()} />
                                        {column.id}
                                    </label>
                                ))}
                            </div>
                        )}
                    </div>
                    <button className="toolbar-btn primary" onClick={exportToCSV}><Download size={14} /> Export CSV</button>
                </div>
            </div>

            {/* --- SINGLE SCROLL CONTAINER with STICKY HEADER/FOOTER --- */}
            <div
                ref={scrollContainerRef}
                className="grid-scroll-container"
                onMouseDown={handleMouseDown}
                style={{
                    overflow: 'auto',
                    flex: 1,
                    width: '100%',
                    cursor: isDragging ? 'grabbing' : 'grab',
                    position: 'relative',
                }}
            >
                <table className="excel-grid" style={{ width: table.getTotalSize(), display: 'block' }}>
                    {/* Sticky Header */}
                    <thead style={{ position: 'sticky', top: 0, zIndex: 10, background: '#f1f3f4', display: 'block' }}>
                        {table.getHeaderGroups().map(headerGroup => (
                            <tr key={headerGroup.id} className="tr header-row" style={{ display: 'flex' }}>
                                {headerGroup.headers.map(header => (
                                    <th key={header.id} className="th" style={{ width: header.getSize(), display: 'flex', flexDirection: 'column' }}>
                                        <div
                                            className="th-clickable"
                                            onClick={header.column.getToggleSortingHandler()}
                                            style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}
                                        >
                                            {flexRender(header.column.columnDef.header, header.getContext())}
                                            <span className="sort-icon">
                                                {{ asc: <ArrowUp size={12} />, desc: <ArrowDown size={12} /> }[header.column.getIsSorted() as string] ?? null}
                                            </span>
                                        </div>
                                        {header.column.getCanFilter() && (
                                            <div className="th-filter-zone">
                                                {clientDataEngine.getSchema()[header.column.id].data_type === 'number' ? (
                                                    <div className="number-filter">
                                                        <input type="number" placeholder="Min"
                                                            value={(header.column.getFilterValue() as [number, number])?.[0] ?? ''}
                                                            onChange={e => header.column.setFilterValue((old: [number, number]) => [e.target.value, old?.[1]])} />
                                                        <input type="number" placeholder="Max"
                                                            value={(header.column.getFilterValue() as [number, number])?.[1] ?? ''}
                                                            onChange={e => header.column.setFilterValue((old: [number, number]) => [old?.[0], e.target.value])} />
                                                    </div>
                                                ) : (
                                                    <div className="string-filter">
                                                        <Filter size={10} className="filter-icon" />
                                                        <input type="text" placeholder={`Filter ${header.column.id}...`}
                                                            value={(header.column.getFilterValue() ?? '') as string}
                                                            onChange={e => header.column.setFilterValue(e.target.value)} />
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </th>
                                ))}
                            </tr>
                        ))}
                    </thead>

                    {/* Virtualized Body */}
                    <tbody style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: 'relative', display: 'block' }}>
                        {rowVirtualizer.getVirtualItems().map(virtualRow => {
                            const row = rows[virtualRow.index];
                            return (
                                <MemoizedRow
                                    key={row.id}
                                    row={row}
                                    virtualRow={virtualRow}
                                    measureElement={rowVirtualizer.measureElement}
                                />
                            );
                        })}
                    </tbody>

                    {/* Sticky Footer */}
                    <tfoot style={{ position: 'sticky', bottom: 0, zIndex: 10, background: '#f8f9fa', display: 'block', borderTop: '2px solid #ccc' }}>
                        {table.getFooterGroups().map(footerGroup => (
                            <tr key={footerGroup.id} className="tr footer-row" style={{ display: 'flex' }}>
                                {footerGroup.headers.map(header => (
                                    <td key={header.id} className="td footer-cell" style={{ width: header.getSize() }}>
                                        {flexRender(header.column.columnDef.footer, header.getContext())}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tfoot>
                </table>
            </div>
        </div>
    );
};