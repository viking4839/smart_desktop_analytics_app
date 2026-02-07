"""
Chart data formatter.
Formats query results for UI charting libraries (Recharts).
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ChartFormatter:
    """Formats query results for UI charting libraries."""
    
    def format(self, result_df: pd.DataFrame, x_column: str = None, y_column: str = None, chart_type: str = "bar") -> Optional[Dict[str, Any]]:
        """
        Format dataframe for charting.
        
        Args:
            result_df: Input dataframe
            x_column: Column for X axis (categories). Auto-detected if None.
            y_column: Column for Y axis (values). Auto-detected if None.
            chart_type: Type of chart (bar, line, pie, area)
            
        Returns:
            Chart configuration dictionary or None if can't create chart
        """
        if result_df.empty:
            return None
        
        # Auto-detect columns if not specified
        if x_column is None or y_column is None:
            x_column, y_column = self._auto_detect_columns(result_df)
        
        if not x_column or not y_column:
            return None
        
        # Validate columns exist
        if x_column not in result_df.columns or y_column not in result_df.columns:
            return None
        
        # Limit data points for performance (UI can't render millions of bars)
        df_chart = result_df.head(50).copy()
        
        # Ensure proper types and handle nulls
        data_points = []
        for _, row in df_chart.iterrows():
            x_val = row[x_column]
            y_val = row[y_column]
            
            # Convert to JSON-safe types
            if pd.isna(x_val):
                x_str = "NULL"
            elif isinstance(x_val, (np.integer, np.int64, np.int32)):
                x_str = str(int(x_val))
            elif isinstance(x_val, (np.floating, np.float64, np.float32)):
                x_str = str(float(x_val))
            elif isinstance(x_val, pd.Timestamp):
                x_str = x_val.strftime('%Y-%m-%d')
            else:
                x_str = str(x_val)
            
            if pd.isna(y_val):
                y_num = 0
            elif isinstance(y_val, (np.integer, np.int64, np.int32)):
                y_num = int(y_val)
            elif isinstance(y_val, (np.floating, np.float64, np.float32)):
                y_num = float(y_val)
            else:
                try:
                    y_num = float(y_val)
                except:
                    y_num = 0
            
            data_points.append({
                "name": x_str,      # X-axis label
                "value": y_num      # Y-axis value
            })
        
        return {
            "type": chart_type,
            "title": f"{y_column} by {x_column}",
            "x_axis": x_column,
            "y_axis": y_column,
            "data": data_points
        }
    
    def _auto_detect_columns(self, df: pd.DataFrame) -> tuple:
        """
        Auto-detect which columns to use for X and Y axes.
        
        Returns:
            Tuple of (x_column, y_column)
        """
        # Get numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        x_column = None
        y_column = None
        
        # Strategy: Use first non-numeric for X, first numeric for Y
        if non_numeric_cols and numeric_cols:
            x_column = non_numeric_cols[0]
            y_column = numeric_cols[0]
        
        # Fallback: Use first two columns
        elif len(df.columns) >= 2:
            x_column = df.columns[0]
            y_column = df.columns[1]
        
        # Fallback: Use first column for both
        elif len(df.columns) == 1:
            x_column = df.columns[0]
            y_column = df.columns[0]
        
        return x_column, y_column
    
    def suggest_chart_type(self, result_df: pd.DataFrame, x_column: str, y_column: str) -> str:
        """
        Suggest the best chart type based on data characteristics.
        
        Returns:
            Chart type string: "bar", "line", "pie", "area"
        """
        if result_df.empty:
            return "bar"
        
        # Get unique count
        unique_count = result_df[x_column].nunique()
        
        # Pie chart: Few categories (2-8)
        if 2 <= unique_count <= 8:
            return "pie"
        
        # Line chart: Many data points (suggesting time series)
        if unique_count > 15:
            return "line"
        
        # Bar chart: Default for categorical data
        return "bar"