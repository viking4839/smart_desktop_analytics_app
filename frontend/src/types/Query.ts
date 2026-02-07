/**
 * Query Model - defines what analysis to run
 * 
 * This is the contract between frontend and backend.
 * Every user interaction in UI builds one of these objects.
 */

export type QueryType = "SUMMARY" | "COMPARISON" | "TREND";

export interface JoinDefinition {
    targetDatasetId: string;
    leftKey: string;
    rightKey: string;
    relationshipType?: "one-to-one" | "one-to-many";
}

export interface FilterDefinition {
    dateRange?: [string, string];  // ISO date strings
    equals?: Record<string, string | number>;
    notNull?: string[];  // Columns that must not be null
}

export interface Query {
    type: QueryType;
    datasetId: string;  // Primary dataset
    joins?: JoinDefinition[];  // Only one allowed in MVP
    metrics: string[];  // e.g. ["sum(revenue)", "avg(quantity)"]
    groupBy?: string[];  // e.g. ["category", "region"]
    filters?: FilterDefinition;
    limit?: number;
    
    // UI metadata (not sent to backend)
    ui?: {
        name: string;  // User-friendly query name
        description?: string;
    };
}

/**
 * Result Model - backend response
 * 
 * Contains the analysis results in multiple formats
 * plus provenance for explainability.
 */
export interface ColumnStat {
    name: string;
    type: string;
    nullCount: number;
    distinctCount?: number;
}

export interface Provenance {
    datasets: string[];  // Dataset IDs used
    columns: string[];   // Column names used
    operations: string[]; // e.g. ["sum(revenue)", "groupBy(category)"]
    generatedAt: string;
    queryHash: string;   // For caching
}

export interface Result {
    summary: Record<string, number | string>;  // Key metrics
    table?: {
        columns: string[];
        rows: (string | number)[][];
    };
    chart?: {
        type: "line" | "bar" | "pie";
        x: string[];
        series: { name: string; data: number[] }[];
    };
    statement: string;  // Natural language explanation
    provenance: Provenance;
    
    // Quality indicators
    quality?: {
        confidence: "high" | "medium" | "low";
        warnings?: string[];
        notes?: string[];
    };
}