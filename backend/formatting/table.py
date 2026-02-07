"""
Table formatting for UI display.
"""
import pandas as pd
from typing import Dict, Any, List


class TableFormatter:
    """Formats pandas DataFrames for UI table display."""
    
    def format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Format DataFrame for table display.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with columns and rows for UI
        """
        if df.empty:
            return {
                "columns": [],
                "rows": [],
                "row_count": 0
            }
        
        # Convert DataFrame to UI-friendly format
        columns = df.columns.tolist()
        
        # Convert rows to list of lists, handling different data types
        rows = []
        for _, row in df.iterrows():
            row_values = []
            for val in row:
                # Handle different types for JSON serialization
                if pd.isna(val):
                    row_values.append(None)
                elif isinstance(val, (pd.Timestamp, pd.DatetimeIndex)):
                    row_values.append(val.isoformat())
                elif isinstance(val, float):
                    # Format floats nicely
                    row_values.append(round(val, 4))
                else:
                    row_values.append(str(val))
            rows.append(row_values)
        
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows)
        }