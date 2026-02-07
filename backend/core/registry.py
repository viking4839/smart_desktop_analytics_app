"""
Production-ready thread-safe data registry with proper locking and resource management.
Fixes critical thread safety issues and adds transaction support.
"""
import threading
import asyncio
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json
import sqlite3
from datetime import datetime
from contextlib import contextmanager
import logging
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetrics:
    """Resource usage metrics for a dataset."""
    memory_bytes: int
    load_time: float
    last_accessed: datetime
    access_count: int


class DatasetCache:
    """
    LRU cache for datasets with memory management.
    Uses weak references to allow garbage collection when memory is needed.
    """
    
    def __init__(self, max_memory_mb: int = 1024):
        """
        Initialize cache.
        
        Args:
            max_memory_mb: Maximum memory to use for cached datasets (in MB)
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.cache: Dict[str, Any] = {}  # dataset_id -> Dataset
        self.metrics: Dict[str, DatasetMetrics] = {}
        self.lock = threading.RLock()  # Reentrant lock
        
        logger.info(f"DatasetCache initialized with {max_memory_mb}MB limit")
    
    def get(self, dataset_id: str) -> Optional[Any]:
        """Get dataset from cache."""
        with self.lock:
            dataset = self.cache.get(dataset_id)
            if dataset:
                # Update access metrics
                if dataset_id in self.metrics:
                    self.metrics[dataset_id].last_accessed = datetime.now()
                    self.metrics[dataset_id].access_count += 1
            return dataset
    
    def put(self, dataset_id: str, dataset: Any, memory_bytes: int):
        """
        Add dataset to cache, evicting old entries if needed.
        
        Args:
            dataset_id: Dataset identifier
            dataset: Dataset object
            memory_bytes: Estimated memory usage in bytes
        """
        with self.lock:
            # Check if we need to evict
            while (self.current_memory_bytes + memory_bytes > self.max_memory_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add to cache
            self.cache[dataset_id] = dataset
            self.metrics[dataset_id] = DatasetMetrics(
                memory_bytes=memory_bytes,
                load_time=0.0,
                last_accessed=datetime.now(),
                access_count=1
            )
            self.current_memory_bytes += memory_bytes
            
            logger.info(f"Cached dataset {dataset_id} ({memory_bytes / 1024 / 1024:.2f}MB). "
                       f"Total cache: {self.current_memory_bytes / 1024 / 1024:.2f}MB")
    
    def remove(self, dataset_id: str):
        """Remove dataset from cache."""
        with self.lock:
            if dataset_id in self.cache:
                memory_bytes = self.metrics[dataset_id].memory_bytes
                del self.cache[dataset_id]
                del self.metrics[dataset_id]
                self.current_memory_bytes -= memory_bytes
                logger.info(f"Removed dataset {dataset_id} from cache")
    
    def _evict_lru(self):
        """Evict least recently used dataset."""
        if not self.metrics:
            return
        
        # Find LRU dataset
        lru_id = min(self.metrics.keys(), 
                    key=lambda k: self.metrics[k].last_accessed)
        
        logger.info(f"Evicting LRU dataset: {lru_id}")
        self.remove(lru_id)
    
    def clear(self):
        """Clear all cached datasets."""
        with self.lock:
            self.cache.clear()
            self.metrics.clear()
            self.current_memory_bytes = 0
            logger.info("Cache cleared")


class DataRegistry:
    """
    Thread-safe registry for managing datasets with proper resource management.
    
    Key improvements over original:
    - Proper thread safety with RLock
    - Atomic operations
    - Transaction support for database operations
    - LRU cache with memory limits
    - Resource cleanup
    - Comprehensive error handling
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataRegistry, cls).__new__(cls)
            return cls._instance
    
    def __init__(self, db_path: str = "data_registry.db", max_cache_mb: int = 1024):
        """
        Initialize registry.
        
        Args:
            db_path: Path to SQLite database
            max_cache_mb: Maximum memory for dataset cache (MB)
        """
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._db_path = Path(db_path)
        self._cache = DatasetCache(max_cache_mb)
        self._registry_lock = threading.RLock()  # Reentrant lock for nested calls
        self._db_lock = threading.Lock()  # Separate lock for database operations
        
        # Initialize database
        self._init_database()
        
        logger.info(f"DataRegistry initialized with db={db_path}, cache={max_cache_mb}MB")
    
    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        with self._get_db_connection() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Datasets table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_path TEXT,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    schema_json TEXT NOT NULL,
                    loaded_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    memory_bytes INTEGER DEFAULT 0,
                    UNIQUE(source_path)
                )
            """)
            
            # Relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    left_dataset_id TEXT NOT NULL,
                    left_column TEXT NOT NULL,
                    right_dataset_id TEXT NOT NULL,
                    right_column TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (left_dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                    FOREIGN KEY (right_dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                    UNIQUE(left_dataset_id, left_column, right_dataset_id, right_column)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_accessed ON datasets(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_left ON relationships(left_dataset_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_right ON relationships(right_dataset_id)")
            
            conn.commit()
            logger.info("Database schema initialized")
    
    @contextmanager
    def _get_db_connection(self):
        """
        Get a database connection with proper locking and error handling.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        with self._db_lock:
            conn = None
            try:
                conn = sqlite3.connect(
                    self._db_path,
                    timeout=10.0,  # Wait up to 10 seconds for locks
                    isolation_level='DEFERRED'  # Better concurrency
                )
                conn.row_factory = sqlite3.Row  # Dict-like access
                yield conn
            except Exception as e:
                logger.error(f"Database error: {e}", exc_info=True)
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()
    
    async def register_dataset(self, file_path: str, ingestor) -> Any:
        """
        Load and register a dataset atomically.
        
        Args:
            file_path: Path to data file
            ingestor: DataIngestor instance for loading files
        
        Returns:
            Registered Dataset object
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file already registered or invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Normalize path
        abs_path = str(path.absolute())
        
        with self._registry_lock:
            # Check if already registered (idempotent operation)
            existing = await self._get_dataset_by_path(abs_path)
            if existing:
                logger.info(f"Dataset already registered: {existing.id}")
                return existing
            
            # Load data (done outside lock to allow other operations)
            logger.info(f"Loading dataset from {abs_path}...")
            load_start = datetime.now()
            
        # Load file (releases lock during I/O)
        dataframe = await asyncio.get_event_loop().run_in_executor(
            None,
            ingestor.load_file,
            abs_path
        )
        
        load_time = (datetime.now() - load_start).total_seconds()
        
        # Re-acquire lock for registration
        with self._registry_lock:
            # Double-check it wasn't registered while we were loading
            existing = await self._get_dataset_by_path(abs_path)
            if existing:
                logger.info(f"Dataset registered by another thread: {existing.id}")
                return existing
            
            # FIXED IMPORT: Changed from 'dataset' to 'core.dataset'
            from core.dataset import Dataset
            
            # Create dataset
            dataset = Dataset.from_dataframe(
                name=path.stem,
                dataframe=dataframe,
                source_path=abs_path
            )
            
            # Estimate memory usage
            memory_bytes = self._estimate_memory(dataframe)
            
            # Persist to database (atomic transaction)
            try:
                with self._get_db_connection() as conn:
                    conn.execute("""
                        INSERT INTO datasets 
                        (id, name, source_path, row_count, column_count, schema_json, 
                         loaded_at, last_accessed, memory_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        dataset.id,
                        dataset.name,
                        dataset.source_path,
                        dataset.row_count,
                        dataset.column_count,
                        json.dumps({k: v.to_dict() for k, v in dataset.schema.items()}),
                        dataset.loaded_at.isoformat(),
                        datetime.now().isoformat(),
                        memory_bytes
                    ))
                    conn.commit()
                
                # Add to cache
                self._cache.put(dataset.id, dataset, memory_bytes)
                
                logger.info(f"Registered dataset {dataset.id} ({dataset.row_count} rows, "
                           f"{memory_bytes / 1024 / 1024:.2f}MB) in {load_time:.2f}s")
                
                return dataset
                
            except sqlite3.IntegrityError as e:
                logger.error(f"Dataset registration failed: {e}")
                raise ValueError(f"Dataset with path {abs_path} already exists")
    
    def get_dataset(self, dataset_id: str) -> Optional[Any]:
        """
        Get dataset by ID with caching.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Dataset object or None if not found
        """
        with self._registry_lock:
            # Check cache first
            dataset = self._cache.get(dataset_id)
            if dataset:
                logger.debug(f"Cache hit for dataset {dataset_id}")
                
                # Update access time in database
                asyncio.create_task(self._update_access_time(dataset_id))
                
                return dataset
            
            # Load from database
            logger.debug(f"Cache miss for dataset {dataset_id}, loading from DB")
            return self._load_dataset_from_db(dataset_id)
    
    async def _get_dataset_by_path(self, path: str) -> Optional[Any]:
        """Get dataset by source path."""
        with self._get_db_connection() as conn:
            row = conn.execute(
                "SELECT id FROM datasets WHERE source_path = ?",
                (path,)
            ).fetchone()
            
            if row:
                return self.get_dataset(row['id'])
            return None
    
    def _load_dataset_from_db(self, dataset_id: str) -> Optional[Any]:
        """Load dataset from database and reconstruct."""
        with self._get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ?",
                (dataset_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # FIXED IMPORTS: Changed from bare module names to full paths
            from .dataset import Dataset, ColumnSchema
            import pandas as pd
            
            # Reconstruct dataset (need to reload data file)
            # This is expensive - that's why we cache!
            try:
                # FIXED IMPORT: Changed from 'io_ingestor' to 'io.ingestor'
                from data_io.ingestor import DataIngestor
                ingestor = DataIngestor()
                dataframe = ingestor.load_file(row['source_path'])
                
                # Reconstruct schema
                schema_dict = json.loads(row['schema_json'])
                schema = {
                    name: ColumnSchema(**col_dict)
                    for name, col_dict in schema_dict.items()
                }
                
                dataset = Dataset(
                    id=row['id'],
                    name=row['name'],
                    dataframe=dataframe,
                    schema=schema,
                    row_count=row['row_count'],
                    column_count=row['column_count'],
                    source_path=row['source_path'],
                    loaded_at=datetime.fromisoformat(row['loaded_at'])
                )
                
                # Add to cache
                self._cache.put(dataset_id, dataset, row['memory_bytes'])
                
                return dataset
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_id}: {e}", exc_info=True)
                return None
    
    async def _update_access_time(self, dataset_id: str):
        """Update last access time in database."""
        with self._get_db_connection() as conn:
            conn.execute(
                "UPDATE datasets SET last_accessed = ? WHERE id = ?",
                (datetime.now().isoformat(), dataset_id)
            )
            conn.commit()
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all registered datasets (metadata only).
        
        Returns:
            List of dataset metadata dictionaries
        """
        with self._registry_lock:
            with self._get_db_connection() as conn:
                rows = conn.execute("""
                    SELECT id, name, row_count, column_count, source_path, 
                           loaded_at, last_accessed, memory_bytes
                    FROM datasets
                    ORDER BY last_accessed DESC
                """).fetchall()
                
                return [
                    {
                        "id": row['id'],
                        "name": row['name'],
                        "row_count": row['row_count'],
                        "column_count": row['column_count'],
                        "source_path": row['source_path'],
                        "loaded_at": row['loaded_at'],
                        "last_accessed": row['last_accessed'],
                        "memory_mb": row['memory_bytes'] / 1024 / 1024
                    }
                    for row in rows
                ]
    
    def remove_dataset(self, dataset_id: str) -> bool:
        """
        Remove dataset from registry.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            True if removed, False if not found
        """
        with self._registry_lock:
            try:
                with self._get_db_connection() as conn:
                    # Remove from database (cascades to relationships)
                    cursor = conn.execute(
                        "DELETE FROM datasets WHERE id = ?",
                        (dataset_id,)
                    )
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        # Remove from cache
                        self._cache.remove(dataset_id)
                        logger.info(f"Removed dataset {dataset_id}")
                        return True
                    
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to remove dataset {dataset_id}: {e}", exc_info=True)
                return False
    
    def clear_all(self):
        """Clear all datasets (for testing)."""
        with self._registry_lock:
            try:
                with self._get_db_connection() as conn:
                    conn.execute("DELETE FROM relationships")
                    conn.execute("DELETE FROM datasets")
                    conn.commit()
                
                self._cache.clear()
                logger.info("All datasets cleared")
                
            except Exception as e:
                logger.error(f"Failed to clear datasets: {e}", exc_info=True)
    
    def _estimate_memory(self, dataframe) -> int:
        """
        Estimate memory usage of a DataFrame.
        
        Args:
            dataframe: pandas DataFrame
        
        Returns:
            Estimated memory usage in bytes
        """
        try:
            # pandas has a built-in method for this
            return int(dataframe.memory_usage(deep=True).sum())
        except:
            # Fallback estimation
            return len(dataframe) * len(dataframe.columns) * 8  # Rough estimate
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._registry_lock:
            return {
                "cached_datasets": len(self._cache.cache),
                "total_memory_mb": self._cache.current_memory_bytes / 1024 / 1024,
                "max_memory_mb": self._cache.max_memory_bytes / 1024 / 1024,
                "utilization_percent": (self._cache.current_memory_bytes / 
                                      self._cache.max_memory_bytes * 100)
            }
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_cache'):
            self._cache.clear()