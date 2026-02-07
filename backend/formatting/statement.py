"""
Natural language statement generation for query results.
"""
from typing import Dict, Any, List
import pandas as pd
from core.query_model import Query


class StatementFormatter:
    """Formats query results into human-readable statements."""
    
    async def format(self, result_df: pd.DataFrame, query: Query, provenance: Dict[str, Any]) -> str:
        """
        Generate a natural language statement from results.
        
        Args:
            result_df: Result dataframe
            query: Original query
            provenance: Query provenance
            
        Returns:
            Human-readable statement
        """
        if result_df.empty:
            return "No results found for the given query."
        
        try:
            if query.group_by:
                return self._format_grouped_result(result_df, query)
            else:
                return self._format_aggregate_result(result_df, query)
        except Exception:
            # Fallback to simple statement
            return f"Query executed successfully. Returned {len(result_df)} rows."
    
    def _format_aggregate_result(self, result_df: pd.DataFrame, query: Query) -> str:
        """Format aggregated result (no grouping)."""
        statements = []
        
        for metric in query.metrics:
            func, col = query._parse_metric(metric)
            value = result_df[metric].iloc[0]
            
            # Format value based on type
            if pd.api.types.is_numeric_dtype(result_df[metric]):
                if abs(value) > 1000:
                    value_str = f"{value:,.0f}"
                else:
                    value_str = f"{value:,.2f}"
            else:
                value_str = str(value)
            
            if func == "count" and not col:
                statements.append(f"Total count: {value_str}")
            elif col:
                statements.append(f"{func.capitalize()} of {col}: {value_str}")
            else:
                statements.append(f"{func.capitalize()}: {value_str}")
        
        return " | ".join(statements)
    
    def _format_grouped_result(self, result_df: pd.DataFrame, query: Query) -> str:
        """Format grouped result."""
        group_col = query.group_by[0] if query.group_by else None
        metric = query.metrics[0] if query.metrics else None
        
        if not group_col or not metric:
            return f"Grouped analysis with {len(result_df)} categories."
        
        # Find top 3 values
        result_df = result_df.sort_values(metric, ascending=False)
        top_values = result_df.head(3)
        
        func, col = query._parse_metric(metric)
        
        parts = []
        for _, row in top_values.iterrows():
            group_val = row[group_col]
            metric_val = row[metric]
            
            if pd.api.types.is_numeric_dtype(type(metric_val)):
                metric_str = f"{metric_val:,.0f}" if abs(metric_val) > 1000 else f"{metric_val:,.2f}"
            else:
                metric_str = str(metric_val)
            
            parts.append(f"{group_val}: {metric_str}")
        
        if len(parts) == 1:
            return f"{func.capitalize()} of {col} for {group_col} = {parts[0]}"
        else:
            return f"Top {len(parts)} {group_col} by {func} of {col}: " + "; ".join(parts)