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

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from ipc_handler import IPCHandler
from core.dataset import Dataset
from core.query_model import Query
from core.registry import DataRegistry
from data_io.ingestor import DataIngestor

# Configure logging
# Configure logging (Force UTF-8 for file, safe defaults for stream)
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
    
    Key improvements:
    - Proper async/await throughout
    - Resource cleanup
    - Health monitoring
    - Graceful shutdown
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
        }
        
        for command, handler in handlers.items():
            self.ipc.register_handler(command, handler)
    
    # ========== Command Handlers ==========
    
    async def cmd_ping(self) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns:
            Status information
        """
        return {
            "status": "healthy",
            "version": "1.0.0",
            "dataset_count": len(self.registry.list_datasets()),
            "cache_stats": self.registry.get_cache_stats()
        }
    
    async def cmd_list_datasets(self) -> Dict[str, Any]:
        """
        List all registered datasets.
        
        Returns:
            List of dataset metadata
        """
        datasets = self.registry.list_datasets()
        return {
            "datasets": datasets,
            "count": len(datasets)
        }
    
    async def cmd_register_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Register a new dataset from file.
        
        Args:
            file_path: Path to data file
        
        Returns:
            Dataset metadata and validation results
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        # Validate input
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path must be a non-empty string")
        
        # Sanitize path to prevent traversal attacks
        file_path = self._sanitize_path(file_path)
        
        # Register dataset (uses thread-safe registry)
        dataset = await self.registry.register_dataset(file_path, self.ingestor)
        
        # Validate data quality
        validation = self.ingestor.validate_dataframe(dataset.dataframe)
        
        return {
            "dataset": dataset.to_dict(),
            "validation": validation
        }
    
    async def cmd_get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Dataset metadata
        
        Raises:
            ValueError: If dataset not found
        """
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        return dataset.to_dict()
    
    async def cmd_preview_dataset(self, dataset_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get dataset preview (first N rows).
        
        Args:
            dataset_id: Dataset identifier
            limit: Number of rows to return (default: 10, max: 100)
        
        Returns:
            Dataset preview with columns and rows
        
        Raises:
            ValueError: If dataset not found or invalid limit
        """
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        # Validate and clamp limit
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        limit = min(limit, 100)  # Maximum 100 rows
        
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        
        # Get preview (creates a copy - safe for concurrent access)
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
        
        # ⭐ NEW: Generate auto-suggestions
        from analysis.suggestions import AnalysisSuggestions
        suggestions = AnalysisSuggestions.generate_suggestions(dataset)
        insights = AnalysisSuggestions.get_quick_insights(dataset)
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "schema": {name: col.to_dict() for name, col in dataset.schema.items()},
            "suggested_queries": suggestions,  # ⭐ NEW
            "quick_insights": insights,        # ⭐ NEW
            "quality_score": self._calculate_dataset_quality(dataset)  # ⭐ NEW
        }

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
    
 # File: backend/main.py
# Line: ~251

    async def cmd_execute_query(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not query_dict:
            raise ValueError("query_dict is required")
    
        query = Query.from_dict(query_dict)
    
        from execution.engine import AnalyticsEngine
        engine = AnalyticsEngine(self.registry)
        result = await engine.execute(query)
    
        # FIX: Handle both dict and QueryResult
        if isinstance(result, dict):
            return result  # Already a dict
        else:
            return result.to_dict()  # QueryResult object
    
    async def cmd_remove_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Remove a dataset from registry.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Success status
        """
        if not dataset_id:
            raise ValueError("dataset_id is required")
        
        removed = self.registry.remove_dataset(dataset_id)
        
        return {
            "success": removed,
            "dataset_id": dataset_id
        }
    
    async def cmd_get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache metrics
        """
        return self.registry.get_cache_stats()
    
    # ========== Helper Methods ==========
    
    def _sanitize_path(self, file_path: str) -> str:
        """
        Sanitize file path to prevent directory traversal attacks.
        
        Args:
            file_path: User-provided file path
        
        Returns:
            Sanitized absolute path
        
        Raises:
            ValueError: If path is invalid or suspicious
        """
        try:
            # Convert to absolute path
            path = Path(file_path).resolve()
            
            # Check for suspicious patterns
            path_str = str(path)
            if '..' in path_str or path_str.startswith('/etc'):
                raise ValueError("Invalid file path")
            
            # Ensure file exists and is a regular file
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not path.is_file():
                raise ValueError(f"Not a file: {file_path}")
            
            return str(path)
            
        except Exception as e:
            logger.warning(f"Path sanitization failed for '{file_path}': {e}")
            raise ValueError(f"Invalid file path: {file_path}")
    
    async def start(self):
        """Start the backend server."""
        self.running = True
        logger.info("Analytics Backend starting...")  # Removed emoji
        
        # Start IPC handler
        await self.ipc.start()
    
    async def shutdown(self):
        """Graceful shutdown."""
        if not self.running:
            return
        
        logger.info("Shutting down Analytics Backend...") # Removed emoji
        self.running = False
        
        # Shutdown IPC handler
        await self.ipc.shutdown()
        
        # Clear cache and close connections
        self.registry.clear_all()
        
        logger.info("Analytics Backend shutdown complete") # Removed emoji


async def main():
    """
    Main entry point.
    """
    # Create backend
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
        # Start backend
        await backend.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        # On Windows, we can't print emojis to stderr easily, so keep it simple
        print(f"Critical failure: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await backend.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user", file=sys.stderr)
    except Exception as e:
        print(f"💥 Critical failure: {e}", file=sys.stderr)
        sys.exit(1)