// frontend/src/services/ClientDataEngine.ts

export interface ColumnInfo {
    name: string;
    data_type: string;
    dtype: string;
    nullable: boolean;
    unique_values: number;
    sample_values: any[];
    statistics?: any;
}

export class ClientDataEngine {
    private rawData: any[] = [];
    private filteredData: any[] = [];
    private schema: Record<string, ColumnInfo> = {};
    private columnStats: Record<string, any> = {};
    private totalRows: number = 0;
    private loadedRows: number = 0;

    loadDataset(data: any[], schema: Record<string, ColumnInfo>, stats: Record<string, any>, totalRows: number) {
        this.rawData = data;
        this.filteredData = [...data];
        this.schema = schema;
        this.columnStats = stats;
        this.totalRows = totalRows;
        this.loadedRows = data.length;
    }

    filter(filters: Record<string, any>) {
        if (!filters || Object.keys(filters).length === 0) {
            this.filteredData = [...this.rawData];
            return this.filteredData;
        }
        this.filteredData = this.rawData.filter(row => {
            return Object.entries(filters).every(([col, filterVal]) => {
                const cellVal = row[col];
                if (filterVal === undefined || filterVal === null || filterVal === '') return true;
                if (typeof filterVal === 'string') {
                    return String(cellVal).toLowerCase().includes(filterVal.toLowerCase());
                }
                return cellVal === filterVal;
            });
        });
        return this.filteredData;
    }

    sort(column: string, direction: 'asc' | 'desc') {
        this.filteredData.sort((a, b) => {
            const aVal = a[column];
            const bVal = b[column];
            if (aVal === bVal) return 0;
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;
            const cmp = aVal < bVal ? -1 : 1;
            return direction === 'asc' ? cmp : -cmp;
        });
        return this.filteredData;
    }

    reset() {
        this.filteredData = [...this.rawData];
        return this.filteredData;
    }

    getFilteredData() { return this.filteredData; }
    getRawData() { return this.rawData; }
    getSchema() { return this.schema; }
    getColumnStats(col: string) { return this.columnStats[col]; }
    getTotalRows() { return this.totalRows; }
    getLoadedRows() { return this.loadedRows; }
}

export const clientDataEngine = new ClientDataEngine();