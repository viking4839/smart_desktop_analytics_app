"""
Production-ready main entry point for analytics backend.
Integrates fixed IPC handler and thread-safe registry.
"""
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import signal
import pandas as pd
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ipc_handler import IPCHandler
from core.dataset import Dataset
from core.query_model import Query
from core.registry import DataRegistry
from data_io.ingestor import DataIngestor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log', encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)


class AnalyticsBackend:
    """
    Main backend application with proper async support and resource management.
    """
    
    def __init__(self):
        """Initialize backend components."""
        self.registry = DataRegistry(max_cache_mb=1024)
        self.ingestor = DataIngestor()
        self.ipc = IPCHandler(
            max_concurrent_requests=5,
            request_timeout=30.0
        )
        self.running = False
        
        # Register command handlers
        self._register_handlers()
        
        logger.info("AnalyticsBackend initialized")
    
    def _register_handlers(self):
        """Register all IPC command handlers."""
        handlers = {
            'ping': self.cmd_ping,
            'list_datasets': self.cmd_list_datasets,
            'register_dataset': self.cmd_register_dataset,
            'get_dataset': self.cmd_get_dataset,
            'preview_dataset': self.cmd_preview_dataset,
            'get_schema': self.cmd_get_schema,
            'execute_query': self.cmd_execute_query,
            'remove_dataset': self.cmd_remove_dataset,
            'get_cache_stats': self.cmd_get_cache_stats,
            'analyze_dataset': self.cmd_analyze_dataset,          
            'check_data_quality': self.cmd_check_data_quality,    
            'filter_dataset': self.cmd_filter_dataset,   
            'get_dataset_full': self.cmd_get_dataset_full,
        }
        
        for command, handler in handlers.items():
            self.ipc.register_handler(command, handler)
    
    # ========== Helper Methods ==========
    
    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pandas DataFrame to JSON‑serializable format.
        Handles Timestamps, NaN, Infinity, and categoricals.
        """
        df = df.copy()
        
        # Convert datetime columns to ISO strings
        for col in df.select_dtypes(include=['datetime64', 'datetime', 'datetimetz']):
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert categorical to string (to avoid JSON issues)
        for col in df.select_dtypes(include=['category']):
            df[col] = df[col].astype(str)
        
        # Convert numpy numeric types to Python native types
        # (done automatically when using to_dict(orient='records') with our custom converter)
        
        # Replace NaN, Inf, -Inf with None
        df = df.replace([np.nan, np.inf, -np.inf], None)
        
        return df
    
    def _sanitize_path(self, file_path: str) -> str:
        """Sanitize file path to prevent directory traversal attacks."""
        try:
            path = Path(file_path).resolve()
            path_str = str(path)
            
            if '..' in path_str or path_str.startswith('/etc'):
                raise ValueError("Invalid file path")
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not path.is_file():
                raise ValueError(f"Not a file: {file_path}")
            
            return str(path)
            
        except Exception as e:
            logger.warning(f"Path sanitization failed for '{file_path}': {e}")
            raise ValueError(f"Invalid file path: {file_path}")
    
    def _calculate_dataset_quality(self, dataset) -> Dict[str, Any]:
        """Calculate overall dataset quality metrics."""
        total_cells = dataset.row_count * dataset.column_count
        null_count = sum(
            schema.statistics.null_count 
            for schema in dataset.schema.values() 
            if schema.statistics
        )
        
        null_percentage = (null_count / total_cells * 100) if total_cells > 0 else 0
        
        score = 100
        if null_percentage > 50:
            score -= 40
        elif null_percentage > 20:
            score -= 20
        elif null_percentage > 5:
            score -= 10
        
        return {
            "score": max(0, min(100, score)),
            "completeness": round(100 - null_percentage, 1),
            "null_percentage": round(null_percentage, 1)
        }
    
    # ========== Command Handlers ==========
    
    async def cmd_ping(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "dataset_count": len(self.registry.list_datasets()),
            "cache_stats": self.registry.get_cache_stats()
        }
    
    async def cmd_list_datasets(self) -> Dict[str, Any]:
        """List all registered datasets."""
        datasets = self.registry.list_datasets()
        return {
            "datasets": datasets,
            "count": len(datasets)
        }
    
    async def cmd_register_dataset(self, file_path: str) -> Dict[str, Any]:
        """Register a new dataset from file."""
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        file_path = self._sanitize_path(file_path)
        dataset = await self.registry.register_dataset(file_path, self.ingestor)
        validation = self.ingestor.validate_dataframe(dataset.dataframe)
        
        return {
            "dataset": dataset.to_dict(),
            "validation": validation
        }
    
    async def cmd_get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset metadata."""
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        return dataset.to_dict()
    
    async def cmd_preview_dataset(self, dataset_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get dataset preview (first N rows)."""
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, 100)
        
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        df_preview = dataset.get_dataframe_copy().head(limit)
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "preview": {
                "columns": df_preview.columns.tolist(),
                "rows": df_preview.values.tolist(),
                "types": {col: str(dtype) for col, dtype in df_preview.dtypes.items()}
            }
        }
    
    async def cmd_get_schema(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed schema with statistics and auto-suggestions."""
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        from analysis.suggestions import AnalysisSuggestions
        suggestions = AnalysisSuggestions.generate_suggestions(dataset)
        insights = AnalysisSuggestions.get_quick_insights(dataset)
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "schema": {name: col.to_dict() for name, col in dataset.schema.items()},
            "suggested_queries": suggestions,
            "quick_insights": insights,
            "quality_score": self._calculate_dataset_quality(dataset)
        }
    
    async def cmd_execute_query(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query on a dataset."""
        if not query_dict:
            raise ValueError("query_dict is required")
        
        query = Query.from_dict(query_dict)
        
        from execution.engine import AnalyticsEngine
        engine = AnalyticsEngine(self.registry)
        result = await engine.execute(query)
        
        if isinstance(result, dict):
            return result
        else:
            return result.to_dict()
    
    async def cmd_remove_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Remove a dataset from registry."""
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        removed = self.registry.remove_dataset(dataset_id)
        
        return {
            "success": removed,
            "dataset_id": dataset_id
        }
    
    async def cmd_get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.registry.get_cache_stats()
    
    async def cmd_get_dataset_full(self, dataset_id: str, row_limit: int = 10000) -> Dict[str, Any]:
        """
        Fetch entire dataset (up to limit) with schema and stats.
        Optimized for client-side virtual tables.
        """
        if not dataset_id:
            raise ValueError("dataset_id is required")
            
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # 1. Get Data
        df = dataset.get_dataframe_copy()
        total_rows = len(df)
        
        # Limit rows for safety (sending 1M rows via JSON will crash)
        display_df = df.head(row_limit)
        
        # 2. Sanitize (CRITICAL: Fixes Timestamp/NaN errors)
        display_df = self._sanitize_dataframe(display_df)
        
        # 3. Convert to Records (List of Dicts) for JSON
        data = display_df.to_dict(orient="records")

        # 4. Get Full Schema (using existing Dataset.schema)
        schema = {}
        for col_name, col_schema in dataset.schema.items():
            # Only include columns that are actually in the preview
            # (in case the schema has extra columns from earlier versions)
            if col_name in display_df.columns:
                schema[col_name] = col_schema.to_dict()

        # 5. Pre-compute Basic Stats (vectorised, fast)
        stats = {}
        for col in display_df.columns:
            try:
                col_data = df[col]  # use full column data for stats
                if pd.api.types.is_numeric_dtype(col_data):
                    clean = col_data.dropna()
                    if not clean.empty:
                        stats[col] = {
                            "min": float(clean.min()),
                            "max": float(clean.max()),
                            "mean": float(clean.mean()),
                            "median": float(clean.median()),
                            "q25": float(clean.quantile(0.25)),
                            "q75": float(clean.quantile(0.75)),
                            "std": float(clean.std()),
                        }
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    clean = col_data.dropna()
                    if not clean.empty:
                        stats[col] = {
                            "min": clean.min().strftime('%Y-%m-%d %H:%M:%S'),
                            "max": clean.max().strftime('%Y-%m-%d %H:%M:%S'),
                            "unique_count": int(clean.nunique()),
                        }
                else:
                    # Categorical / text
                    clean = col_data.dropna()
                    if not clean.empty:
                        top = clean.value_counts().head(10).to_dict()
                        stats[col] = {
                            "unique_count": int(clean.nunique()),
                            "top_categories": {str(k): int(v) for k, v in top.items()},
                        }
            except Exception as e:
                logger.warning(f"Failed to compute stats for {col}: {e}")
                stats[col] = {}

        return {
            "dataset_id": dataset.id,
            "name": dataset.name,
            "total_rows": total_rows,
            "loaded_rows": len(display_df),
            "data": data,
            "schema": schema,
            "column_stats": stats,
            "can_load_more": total_rows > row_limit,
        }
    
    async def cmd_analyze_dataset(self, dataset_id: str, analysis_type: str, column: str = None):
        """
        Analyze dataset with various methods.
        
        analysis_type options: 
        - 'distribution': Column distribution analysis
        - 'summary_stats': Summary statistics
        - 'outliers': Find outliers
        - 'date_analysis': Date-specific analysis
        - 'duplicates': Find duplicate rows
        """
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        df = dataset.get_dataframe_copy()
        
        if analysis_type == 'distribution' and column:
            if column not in df.columns:
                raise ValueError(f"Column {column} not found in dataset")
            
            col_data = df[column]
            
            if pd.api.types.is_numeric_dtype(col_data):
                stats = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75))
                }
                
                if len(col_data) > 0:
                    hist, bin_edges = np.histogram(col_data.dropna(), bins=10)
                    buckets = []
                    for i in range(len(hist)):
                        buckets.append({
                            'range': f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                            'count': int(hist[i]),
                            'percentage': round(hist[i] / len(col_data) * 100, 1)
                        })
                    
                    return {
                        'type': 'distribution',
                        'title': f'Distribution of {column}',
                        'data': {
                            'statistics': stats,
                            'buckets': buckets,
                            'total': len(col_data),
                            'missing': int(col_data.isnull().sum())
                        },
                        'insights': [
                            f"Column '{column}' has {len(col_data.dropna().unique())} unique values",
                            f"Missing values: {col_data.isnull().sum()} ({col_data.isnull().mean()*100:.1f}%)"
                        ],
                        'recommendations': [
                            "Consider removing or imputing missing values if they affect analysis",
                            "Check for outliers that might skew the distribution"
                        ]
                    }
        
        elif analysis_type == 'summary_stats':
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            summary = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'total_cells': len(df) * len(df.columns),
                'missing_cells': int(df.isnull().sum().sum()),
                'missing_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 1)
            }
            
            return {
                'type': 'summary',
                'title': 'Dataset Summary Statistics',
                'data': summary,
                'insights': [
                    f"Dataset has {summary['total_rows']} rows and {summary['total_columns']} columns",
                    f"Missing data: {summary['missing_percentage']}% of cells"
                ],
                'recommendations': [
                    f"Consider handling missing values in {df.isnull().sum()[df.isnull().sum() > 0].count()} columns",
                    "Review categorical columns for consistency"
                ]
            }
        
        elif analysis_type == 'outliers' and column:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column {column} is not numeric or not found")
            
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            return {
                'type': 'outliers',
                'title': f'Outliers in {column}',
                'data': {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': len(outliers),
                    'outlier_percentage': round(len(outliers) / len(df) * 100, 2),
                    'sample_outliers': outliers.head(10)[column].tolist()
                },
                'insights': [
                    f"Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)",
                    f"Range: {lower_bound:.2f} to {upper_bound:.2f}"
                ],
                'recommendations': [
                    "Review outliers for data entry errors",
                    "Consider removing outliers if they're not relevant to analysis"
                ]
            }
        
        elif analysis_type == 'duplicates':
            duplicates = df[df.duplicated(keep='first')]
            
            return {
                'type': 'duplicates',
                'title': 'Duplicate Rows Analysis',
                'data': {
                    'duplicate_count': len(duplicates),
                    'duplicate_percentage': round(len(duplicates) / len(df) * 100, 2),
                    'sample_duplicates': duplicates.head(5).values.tolist()
                },
                'insights': [
                    f"Found {len(duplicates)} duplicate rows ({len(duplicates)/len(df)*100:.1f}%)"
                ],
                'recommendations': [
                    "Remove duplicate rows to ensure data integrity",
                    "Check if duplicates are legitimate or data entry errors"
                ]
            }
        
        return {'error': f'Analysis type {analysis_type} not supported'}
    
    async def cmd_check_data_quality(self, dataset_id: str):
        """Check dataset for quality issues."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        df = dataset.get_dataframe_copy()
        
        issues = []
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            
            severity = 'low' if missing_pct < 5 else 'medium' if missing_pct < 20 else 'high'
            
            issues.append({
                'type': 'missing',
                'column': col,
                'count': int(missing_count),
                'examples': [],
                'severity': severity,
                'recommendation': f'Consider imputing missing values or removing rows with >50% missing'
            })
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = duplicate_count / len(df) * 100
            severity = 'low' if duplicate_pct < 2 else 'medium' if duplicate_pct < 10 else 'high'
            
            issues.append({
                'type': 'duplicate',
                'column': 'All columns',
                'count': int(duplicate_count),
                'examples': df[df.duplicated()].head(3).values.tolist(),
                'severity': severity,
                'recommendation': 'Remove duplicate rows to maintain data integrity'
            })
        
        # Check for inconsistent formats in string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 0 and len(unique_values) < 20:
                lower_values = [str(v).lower() for v in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    issues.append({
                        'type': 'inconsistent',
                        'column': col,
                        'count': len(df) - len(df[col].dropna().str.lower().drop_duplicates()),
                        'examples': [],
                        'severity': 'medium',
                        'recommendation': f'Standardize casing in column "{col}"'
                    })
        
        # Calculate overall score
        total_issues = len(issues)
        severity_scores = {'high': 3, 'medium': 2, 'low': 1}
        weighted_score = sum(severity_scores[issue['severity']] for issue in issues)
        
        max_possible_score = total_issues * 3
        if max_possible_score > 0:
            issue_score = (weighted_score / max_possible_score) * 50
        else:
            issue_score = 0
        
        completeness = (df.notnull().sum().sum() / (len(df) * len(df.columns))) * 50
        overall_score = 100 - issue_score + completeness
        overall_score = max(0, min(100, overall_score))
        
        return {
            'overall_score': round(overall_score),
            'issues': issues,
            'summary': f"Found {total_issues} quality issues. Data quality score: {round(overall_score)}/100",
            'row_count': len(df),
            'clean_row_count': len(df.dropna().drop_duplicates())
        }
    
    async def cmd_filter_dataset(self, dataset_id: str, filters: dict):
        """Apply filters to dataset and return filtered preview."""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        df = dataset.get_dataframe_copy()
        
        # Apply filters
        for key, value in filters.items():
            if not value or value == '':
                continue
                
            if key.endswith('_min'):
                col = key.replace('_min', '')
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df = df[df[col] >= float(value)]
            elif key.endswith('_max'):
                col = key.replace('_max', '')
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df = df[df[col] <= float(value)]
            elif key in df.columns:
                df = df[df[key].astype(str).str.contains(str(value), case=False, na=False)]
        
        # Return filtered preview
        preview_df = df.head(100)
        
        return {
            'filtered_preview': {
                'columns': preview_df.columns.tolist(),
                'rows': preview_df.values.tolist(),
                'types': {col: str(dtype) for col, dtype in preview_df.dtypes.items()}
            },
            'filtered_count': len(df),
            'original_count': len(dataset.get_dataframe_copy())
        }
    
    # ========== Lifecycle ==========
    
    async def start(self):
        """Start the backend server."""
        self.running = True
        logger.info("Analytics Backend starting...")
        await self.ipc.start()
    
    async def shutdown(self):
        """Graceful shutdown."""
        if not self.running:
            return
        
        logger.info("Shutting down Analytics Backend...")
        self.running = False
        await self.ipc.shutdown()
        self.registry.clear_all()
        logger.info("Analytics Backend shutdown complete")


async def main():
    """Main entry point."""
    backend = AnalyticsBackend()
    
    # Set up signal handlers (Unix only)
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        
        def signal_handler(signum):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(backend.shutdown())
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    try:
        await backend.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Critical failure: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await backend.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBackend stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"Critical failure: {e}", file=sys.stderr)
        sys.exit(1)