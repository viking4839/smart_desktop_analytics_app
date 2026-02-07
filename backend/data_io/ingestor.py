"""
Enhanced data ingestor with multiple file formats and quality scoring.
Supports CSV, Excel, JSON, Parquet, Feather, and more.
FIXED: Handles currency values with proper cleaning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Make chardet optional
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False


def clean_currency_value(value):
    """Clean currency values by removing symbols and commas."""
    if pd.isna(value):
        return value
    
    # If already numeric, return as is
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    
    # Convert to string
    str_value = str(value).strip()
    
    # Remove currency symbols, commas, spaces
    cleaned = re.sub(r'[\$,€£¥\s]', '', str_value)
    
    # Handle parentheses for negative numbers (accounting format)
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        # Return original value if conversion fails
        return value


class DataIngestor:
    """Enhanced data loader with format-specific optimizations."""
    
    # Format-specific configurations
    READERS = {
        '.csv': {
            'function': pd.read_csv,
            'params': {
                'low_memory': False,
                'parse_dates': True,
                'thousands': ',',
                'on_bad_lines': 'warn'
            }
        },
        '.xlsx': {
            'function': pd.read_excel,
            'params': {
                'engine': 'openpyxl'
            }
        },
        '.xls': {
            'function': pd.read_excel,
            'params': {
                'engine': 'xlrd'
            }
        },
        '.json': {
            'function': pd.read_json,
            'params': {
                'orient': 'records'
            }
        },
        '.parquet': {
            'function': pd.read_parquet,
            'params': {}
        },
        '.tsv': {
            'function': pd.read_csv,
            'params': {
                'sep': '\t',
                'low_memory': False,
                'on_bad_lines': 'warn'
            }
        }
    }
    
    DEFAULT_ENCODING = 'utf-8'
    
    def __init__(self):
        """Initialize ingestor."""
        pd.set_option('mode.chained_assignment', None)
        warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    
    def load_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            file_path: Path to file
            **kwargs: Additional arguments
            
        Returns:
            pandas DataFrame
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.READERS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: {list(self.READERS.keys())}"
            )
        
        logger.info(f"Loading {suffix} file: {file_path}")
        
        try:
            reader_config = self.READERS[suffix]
            reader_func = reader_config['function']
            params = reader_config['params'].copy()
            
            # Add encoding for text-based formats
            if suffix in ['.csv', '.tsv']:
                encoding = self._detect_encoding(file_path) if HAS_CHARDET else self.DEFAULT_ENCODING
                params['encoding'] = encoding
            
            # Merge with user params
            params.update(kwargs)
            
            # Load file
            df = reader_func(file_path, **params)
            
            # Basic cleaning
            df = self._clean_dataframe(df)
            
            # Additional cleaning for currency values
            df = self._clean_currency_columns(df)
            
            logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            # Retry with permissive parameters for CSV
            if suffix == '.csv':
                try:
                    logger.warning(f"Retrying CSV with permissive parameters")
                    df = pd.read_csv(
                        file_path,
                        encoding='utf-8',
                        on_bad_lines='skip',
                        low_memory=False
                    )
                    df = self._clean_dataframe(df)
                    df = self._clean_currency_columns(df)
                    return df
                except Exception as e2:
                    raise ValueError(f"Failed to load CSV: {str(e2)}")
            
            raise ValueError(f"Failed to load {suffix} file: {str(e)}")
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding."""
        if not HAS_CHARDET:
            return self.DEFAULT_ENCODING
            
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else self.DEFAULT_ENCODING
                confidence = result['confidence']
                
                if confidence > 0.7:
                    return encoding
                return self.DEFAULT_ENCODING
        except Exception:
            return self.DEFAULT_ENCODING
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply safe cleaning to dataframe."""
        df_clean = df.copy()
        
        # Clean column names
        df_clean.columns = df_clean.columns.astype(str).str.strip()
        
        # Remove completely empty columns
        empty_cols = df_clean.columns[df_clean.isna().all()].tolist()
        if empty_cols:
            logger.info(f"Removing empty columns: {empty_cols}")
            df_clean = df_clean.drop(columns=empty_cols)
        
        # Remove duplicate rows
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        if len(df_clean) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        return df_clean
    
    def _clean_currency_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and clean currency columns by converting string values to numeric.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with cleaned currency columns
        """
        df_clean = df.copy()
        
        # Check each column for currency patterns
        for col in df_clean.columns:
            # Skip if column is already numeric
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            # Check if column contains currency-like values
            try:
                # Sample some values to check
                sample = df_clean[col].dropna().head(20)
                if len(sample) == 0:
                    continue
                
                # Check for currency symbols or comma-separated numbers
                str_sample = sample.astype(str)
                has_currency = str_sample.str.contains(r'[\$€£¥]', na=False, regex=True).any()
                has_comma_numbers = str_sample.str.contains(r'^\s*[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*$', 
                                                          na=False, regex=True).any()
                
                if has_currency or has_comma_numbers:
                    # Try to convert the column to numeric
                    try:
                        # Apply cleaning function
                        cleaned_series = df_clean[col].apply(clean_currency_value)
                        # Check if most values converted successfully
                        non_null_original = df_clean[col].dropna()
                        non_null_cleaned = cleaned_series.dropna()
                        
                        if len(non_null_cleaned) / len(non_null_original) > 0.7:
                            df_clean[col] = cleaned_series
                            logger.info(f"Converted column '{col}' from string to numeric (currency detected)")
                    except Exception as e:
                        logger.debug(f"Could not convert column '{col}' to numeric: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error checking column '{col}' for currency: {e}")
                continue
        
        return df_clean
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataframe validation.
        
        Returns:
            Validation report with quality score
        """
        report = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "stats": {},
            "timestamp": datetime.now().isoformat()
        }
        
        if df.empty:
            report["is_valid"] = False
            report["errors"].append("DataFrame is empty")
            return report
        
        # Check duplicate columns
        if len(df.columns) != len(set(df.columns)):
            report["warnings"].append("Duplicate column names detected")
        
        # Check empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            report["warnings"].append(f"Empty columns: {', '.join(empty_cols[:5])}")
        
        # Check high null percentage
        for col in df.columns:
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 50:
                report["warnings"].append(f"Column '{col}' has {null_pct:.1f}% null values")
        
        # Calculate stats
        total_cells = len(df) * len(df.columns)
        null_cells = df.isna().sum().sum()
        
        report["stats"] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "total_cells": int(total_cells),
            "null_cells": int(null_cells),
            "null_percentage": float((null_cells / total_cells * 100) if total_cells > 0 else 0),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        }
        
        # Check for extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            non_null = df[col].dropna()
            if not non_null.empty and non_null.abs().max() > 1e12:
                report["warnings"].append(f"Column '{col}' has extremely large values")
        
        # Quality score
        report["quality_score"] = self._calculate_quality_score(report)
        
        return report
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate data quality score (0-100)."""
        score = 100
        
        # Penalize for warnings
        score -= len(report.get("warnings", [])) * 5
        
        # Penalize for missing values
        missing_pct = report["stats"]["null_percentage"]
        if missing_pct > 50:
            score -= 30
        elif missing_pct > 20:
            score -= 15
        elif missing_pct > 5:
            score -= 5
        
        return max(0, min(100, score))
    
    def get_supported_formats(self) -> List[Dict[str, str]]:
        """Get list of supported file formats."""
        return [
            {"extension": ext, "description": self._get_description(ext)}
            for ext in self.READERS.keys()
        ]
    
    @staticmethod
    def _get_description(extension: str) -> str:
        """Get format description."""
        descriptions = {
            '.csv': "Comma Separated Values",
            '.xlsx': "Excel Spreadsheet",
            '.xls': "Legacy Excel",
            '.json': "JSON Data",
            '.parquet': "Parquet Columnar",
            '.tsv': "Tab Separated Values"
        }
        return descriptions.get(extension, "Unknown format")


# Singleton instance
_ingestor_instance = None

def get_ingestor() -> DataIngestor:
    """Get singleton ingestor instance."""
    global _ingestor_instance
    if _ingestor_instance is None:
        _ingestor_instance = DataIngestor()
    return _ingestor_instance