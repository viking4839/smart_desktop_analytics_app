export interface ColumnStats {
    min?: number;
    max?: number;
    mean?: number;
    median?: number;
    std?: number;
    unique_count?: number;
}

export interface DatasetSchema {
    [columnName: string]: {
        name: string;
        data_type: string;
        dtype: string;
    };
}

export class ClientDataEngine {
    private rawData: any[] = [];
    private filteredData: any[] = [];
    private schema: DatasetSchema = {};
    private columnStats: Record<string, ColumnStats> = {};
    private totalRows: number = 0;
    private loadedRows: number = 0;

    // NEW: Store active filters
    private activeColumnFilters: Record<string, any> = {};
    private activeGlobalFilter: string = '';

    loadDataset(data: any[], schema: DatasetSchema, stats: Record<string, ColumnStats>, totalRows: number) {
        this.rawData = data;
        this.filteredData = [...data];
        this.schema = schema;
        this.columnStats = stats;
        this.totalRows = totalRows;
        this.loadedRows = data.length;
        this.activeColumnFilters = {};
        this.activeGlobalFilter = '';
    }

    // --- NEW: Unified Filtering Logic ---
    setGlobalFilter(query: string) {
        this.activeGlobalFilter = query.toLowerCase().trim();
        this.applyAllFilters();
    }

    setColumnFilters(filters: Record<string, any>) {
        this.activeColumnFilters = filters;
        this.applyAllFilters();
    }

    private applyAllFilters() {
        // Start with raw data
        let result = [...this.rawData];

        // 1. Apply Global Search (if any)
        if (this.activeGlobalFilter) {
            result = result.filter(row => {
                // Check if ANY value in the row contains the search string
                return Object.values(row).some(cellVal =>
                    cellVal !== null &&
                    cellVal !== undefined &&
                    String(cellVal).toLowerCase().includes(this.activeGlobalFilter)
                );
            });
        }

        // 2. Apply Specific Column Filters
        if (Object.keys(this.activeColumnFilters).length > 0) {
            result = result.filter(row => {
                return Object.entries(this.activeColumnFilters).every(([col, filterVal]) => {
                    const cellVal = row[col];
                    if (filterVal === undefined || filterVal === null || filterVal === '') return true;
                    if (typeof cellVal === 'string') {
                        return cellVal.toLowerCase().includes(String(filterVal).toLowerCase());
                    }
                    return cellVal == filterVal;
                });
            });
        }

        this.filteredData = result;
    }

    // --- Sorting ---
    sort(column: string, direction: 'asc' | 'desc') {
        this.filteredData.sort((a, b) => {
            const aVal = a[column];
            const bVal = b[column];
            if (aVal === bVal) return 0;
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;
            return direction === 'asc' ? (aVal < bVal ? -1 : 1) : (aVal > bVal ? -1 : 1);
        });
        return this.filteredData;
    }

    // --- Getters & Utilities ---
    getFilteredData() { return this.filteredData; }
    getRawData() { return this.rawData; }
    getSchema() { return this.schema; }
    getTotalRows() { return this.totalRows; }
    getLoadedRows() { return this.loadedRows; }
    getColumnStats(column: string) { return this.columnStats[column]; }

    reset() {
        this.activeColumnFilters = {};
        this.activeGlobalFilter = '';
        this.filteredData = [...this.rawData];
    }
}

export const clientDataEngine = new ClientDataEngine();