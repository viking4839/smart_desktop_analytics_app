"""
Enhanced dataset model with comprehensive data type support.
Includes currency, percentage, duration detection and rich statistics.
FIXED: Handles currency strings with $ and commas.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np
import re


class DataType(str, Enum):
    """Comprehensive data types for business analytics."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    DURATION = "duration"
    UNKNOWN = "unknown"


def convert_to_json_serializable(value: Any) -> Any:
    """Convert any value to JSON-serializable format."""
    if pd.isna(value):
        return None
    elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, (np.bool_, bool)):
        return bool(value)
    elif isinstance(value, (np.ndarray, pd.Series)):
        return [convert_to_json_serializable(v) for v in value.tolist()]
    elif isinstance(value, pd.Timestamp):
        return value.isoformat()
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, (list, tuple)):
        return [convert_to_json_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {k: convert_to_json_serializable(v) for k, v in value.items()}
    else:
        return value


def clean_numeric_string(value: Any) -> Optional[float]:
    """
    Clean and convert string values with currency symbols, commas, etc. to float.
    
    Args:
        value: String or numeric value to clean
        
    Returns:
        Cleaned float value or None if conversion fails
    """
    if pd.isna(value):
        return None
    
    # If it's already numeric, return as float
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    
    # Convert to string
    str_value = str(value).strip()
    
    # Remove currency symbols, commas, spaces
    cleaned = re.sub(r'[\$,€£¥\s]', '', str_value)
    
    # Handle parentheses for negative numbers (accounting format)
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    # Remove percentage signs and convert to decimal
    is_percentage = False
    if cleaned.endswith('%'):
        cleaned = cleaned[:-1]
        is_percentage = True
    
    try:
        # Convert to float
        result = float(cleaned)
        # Convert percentage to decimal
        if is_percentage:
            result = result / 100.0
        return result
    except (ValueError, TypeError):
        return None


@dataclass
class ColumnStatistics:
    """Comprehensive statistics for any column type."""
    count: int
    null_count: int
    null_percentage: float
    
    # Type-specific stats
    numeric_stats: Optional[Dict] = None
    categorical_stats: Optional[Dict] = None
    datetime_stats: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "count": self.count,
            "null_count": self.null_count,
            "null_percentage": round(self.null_percentage, 2)
        }
        
        if self.numeric_stats:
            result["numeric_stats"] = convert_to_json_serializable(self.numeric_stats)
        if self.categorical_stats:
            result["categorical_stats"] = convert_to_json_serializable(self.categorical_stats)
        if self.datetime_stats:
            result["datetime_stats"] = convert_to_json_serializable(self.datetime_stats)
            
        return result


@dataclass
class ColumnSchema:
    """Enhanced column schema with comprehensive metadata."""
    name: str
    data_type: DataType
    nullable: bool
    unique_values: int
    sample_values: List[Any] = field(default_factory=list)
    statistics: Optional[ColumnStatistics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API/UI with proper type conversion."""
        converted_samples = [convert_to_json_serializable(v) for v in self.sample_values[:10]]
        
        result = {
            "name": self.name,
            "data_type": self.data_type.value,
            "dtype": self.data_type.value,  # Legacy compatibility
            "nullable": convert_to_json_serializable(self.nullable),
            "unique_values": convert_to_json_serializable(self.unique_values),
            "sample_values": converted_samples,
            "metadata": convert_to_json_serializable(self.metadata)
        }
        
        if self.statistics:
            result["stats"] = self.statistics.to_dict()
            result["statistics"] = self.statistics.to_dict()  # Both keys for compatibility
            
        return result
    
    @classmethod
    def infer_from_series(cls, series: pd.Series, name: str) -> 'ColumnSchema':
        """Infer schema from pandas Series with intelligent detection."""
        # Determine data type
        data_type, metadata = cls._detect_data_type(series)
        
        # Get sample values (non-null)
        sample_values = [convert_to_json_serializable(v) 
                        for v in series.dropna().head(10).tolist()]
        
        # Calculate statistics
        statistics = cls._calculate_statistics(series, data_type)
        
        return cls(
            name=name,
            data_type=data_type,
            nullable=bool(series.isna().any()),
            unique_values=int(series.nunique()),
            sample_values=sample_values,
            statistics=statistics,
            metadata=metadata
        )
    
    @staticmethod
    def _detect_data_type(series: pd.Series) -> Tuple[DataType, Dict]:
        """Intelligently detect data type with metadata."""
        metadata = {}
        
        # Handle NaN-only series
        if series.dropna().empty:
            return DataType.UNKNOWN, metadata
        
        non_null = series.dropna()
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME, {"format": "datetime64"}
        
        # Check for timedelta/duration
        if pd.api.types.is_timedelta64_dtype(series):
            return DataType.DURATION, {"unit": "timedelta"}
        
        # Check for boolean
        if pd.api.types.is_bool_dtype(series):
            return DataType.BOOLEAN, {}
        
        # Check for numeric or numeric-like strings
        # First, try to clean and convert a sample
        sample_values = non_null.head(100).tolist()
        numeric_values = []
        
        for val in sample_values:
            cleaned = clean_numeric_string(val)
            if cleaned is not None:
                numeric_values.append(cleaned)
        
        # If we can convert most values to numeric
        if len(numeric_values) > 0 and len(numeric_values) / len(sample_values) > 0.8:
            # Create a cleaned series for analysis
            cleaned_series = non_null.apply(clean_numeric_string)
            cleaned_series = cleaned_series.dropna()
            
            if len(cleaned_series) > 0:
                # Check for integer vs float
                if cleaned_series.apply(lambda x: float(x).is_integer()).all():
                    return DataType.INTEGER, {}
                
                # Check for percentage (values between 0 and 1)
                if cleaned_series.between(0, 1).all():
                    return DataType.PERCENTAGE, {"range": "0-1"}
                
                # Check for currency (by looking at original string values)
                str_series = non_null.astype(str)
                if str_series.str.contains(r'[\$€£¥]', na=False, regex=True).any():
                    return DataType.CURRENCY, {"symbol_detected": True}
                
                # Default to float
                return DataType.FLOAT, {}
        
        # String columns - check for patterns
        if series.dtype == 'object':
            str_series = series.astype(str)
            
            # Check for currency symbols (but not numeric)
            if str_series.str.contains(r'[\$€£¥]', na=False, regex=True).any():
                # Try to clean and convert to verify it's actually currency
                try:
                    # Sample a few values
                    sample = str_series.head(10).tolist()
                    cleaned_samples = [clean_numeric_string(v) for v in sample]
                    valid_samples = [v for v in cleaned_samples if v is not None]
                    
                    if len(valid_samples) / len(sample) > 0.7:
                        return DataType.CURRENCY, {"symbol_detected": True}
                except:
                    pass
            
            # Check for percentage symbols
            if str_series.str.contains(r'%', na=False).any():
                # Try to clean and convert
                try:
                    sample = str_series.head(10).tolist()
                    cleaned_samples = [clean_numeric_string(v) for v in sample]
                    valid_samples = [v for v in cleaned_samples if v is not None]
                    
                    if len(valid_samples) / len(sample) > 0.7:
                        return DataType.PERCENTAGE, {"symbol_detected": True}
                except:
                    pass
            
            # Check for categorical (low cardinality)
            unique_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0
            if unique_ratio < 0.3 or non_null.nunique() < 20:
                return DataType.CATEGORICAL, {
                    "cardinality": "low",
                    "unique_ratio": float(unique_ratio)
                }
        
        # Default to string
        return DataType.STRING, {"cardinality": "high"}
    
    @staticmethod
    def _calculate_statistics(series: pd.Series, data_type: DataType) -> ColumnStatistics:
        """Calculate type-specific statistics with proper numeric conversion."""
        non_null = series.dropna()
        count = len(series)
        null_count = series.isna().sum()
        null_percentage = (null_count / count) * 100 if count > 0 else 0
        
        stats = ColumnStatistics(
            count=count,
            null_count=int(null_count),
            null_percentage=float(null_percentage)
        )
        
        # Type-specific statistics
        if data_type in [DataType.INTEGER, DataType.FLOAT, DataType.PERCENTAGE, DataType.CURRENCY]:
            # Clean and convert to numeric for statistics
            try:
                # Clean the non-null values
                cleaned_series = non_null.apply(clean_numeric_string)
                cleaned_series = cleaned_series.dropna()
                
                if not cleaned_series.empty and len(cleaned_series) > 0:
                    # Convert to numpy array for faster operations
                    values = cleaned_series.values.astype(float)
                    
                    stats.numeric_stats = {
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "std": float(np.std(values)) if len(values) > 1 else 0,
                        "q25": float(np.percentile(values, 25)),
                        "q75": float(np.percentile(values, 75))
                    }
            except Exception as e:
                # If conversion fails, skip numeric stats
                print(f"Warning: Could not calculate statistics for {series.name}: {e}")
        
        elif data_type == DataType.CATEGORICAL:
            value_counts = non_null.value_counts()
            stats.categorical_stats = {
                "top_categories": {str(k): int(v) for k, v in value_counts.head(5).items()},
                "category_count": int(len(value_counts))
            }
        
        elif data_type == DataType.DATETIME:
            if not non_null.empty:
                try:
                    # Convert to datetime
                    datetime_series = pd.to_datetime(non_null, errors='coerce')
                    datetime_series = datetime_series.dropna()
                    
                    if not datetime_series.empty:
                        min_date = datetime_series.min()
                        max_date = datetime_series.max()
                        stats.datetime_stats = {
                            "min": min_date.isoformat(),
                            "max": max_date.isoformat(),
                            "range_days": (max_date - min_date).days
                        }
                except Exception as e:
                    print(f"Warning: Could not calculate datetime statistics: {e}")
        
        return stats


@dataclass
class Dataset:
    """
    Enhanced dataset container with comprehensive metadata.
    """
    id: str
    name: str
    dataframe: pd.DataFrame
    schema: Dict[str, ColumnSchema]
    row_count: int
    column_count: int
    source_path: str
    source_format: str
    loaded_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate dataset integrity."""
        if self.row_count != len(self.dataframe):
            raise ValueError(f"Row count mismatch: {self.row_count} vs {len(self.dataframe)}")
        if self.column_count != len(self.dataframe.columns):
            raise ValueError(f"Column count mismatch: {self.column_count} vs {len(self.dataframe.columns)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize dataset metadata."""
        return {
            "id": self.id,
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "loaded_at": self.loaded_at.isoformat(),
            "description": self.description,
            "tags": self.tags,
            "schema": {name: col.to_dict() for name, col in self.schema.items()}
        }
    
    def get_column(self, column_name: str) -> pd.Series:
        """Get a copy of a column series."""
        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset '{self.name}'")
        return self.dataframe[column_name].copy()
    
    def get_dataframe_copy(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Get a copy of the dataframe."""
        if columns:
            missing = set(columns) - set(self.dataframe.columns)
            if missing:
                raise ValueError(f"Columns not found: {missing}")
            return self.dataframe[columns].copy()
        return self.dataframe.copy()
    
    def get_preview(self, rows: int = 10) -> Dict[str, Any]:
        """Get dataset preview with proper type handling."""
        preview_df = self.dataframe.head(rows)
        
        rows_data = []
        for _, row in preview_df.iterrows():
            row_dict = {col: convert_to_json_serializable(value) 
                       for col, value in row.items()}
            rows_data.append(row_dict)
        
        return {
            "columns": list(self.dataframe.columns),
            "rows": rows_data,
            "types": {col: schema.data_type.value for col, schema in self.schema.items()}
        }
    
    def validate_for_query(self, required_columns: List[str]) -> Tuple[bool, Optional[str]]:
        """Check if dataset has required columns."""
        missing = set(required_columns) - set(self.dataframe.columns)
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        return True, None
    
    @classmethod
    def from_dataframe(cls, name: str, dataframe: pd.DataFrame, source_path: str = "") -> 'Dataset':
        """Factory method to create Dataset from pandas DataFrame."""
        data_hash = hash(pd.util.hash_pandas_object(dataframe).sum())
        dataset_id = f"ds_{abs(data_hash) % 1000000:06d}"
        
        source_format = Path(source_path).suffix.lower().replace('.', '') if source_path else "unknown"
        
        schema = {}
        for col in dataframe.columns:
            schema[col] = ColumnSchema.infer_from_series(dataframe[col], col)
        
        return cls(
            id=dataset_id,
            name=name,
            dataframe=dataframe.copy(),
            schema=schema,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            source_path=source_path,
            source_format=source_format
        )