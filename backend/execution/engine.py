"""
Analytics query execution engine.
"""
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from core.query_model import Query
from core.registry import DataRegistry
from core.dataset import Dataset
from formatting.statement import StatementFormatter
from formatting.table import TableFormatter
from formatting.chart import ChartFormatter


class AnalyticsEngine:
    """
    Executes analytics queries and formats results.
    Thread-safe for concurrent execution.
    """
    
    def __init__(self, registry: DataRegistry):
        self.registry = registry
        self.statement_formatter = StatementFormatter()
        self.table_formatter = TableFormatter()
        self.chart_formatter = ChartFormatter()
    
    async def execute(self, query: Query) -> Dict[str, Any]:
        """
        Execute a query and return formatted results.
        
        Args:
            query: Query to execute
            
        Returns:
            Complete result with summary, table, chart, and statement
        """
        # Start timing
        start_time = datetime.now()
        
        try:
            # Get primary dataset
            primary_dataset = self.registry.get_dataset(query.dataset_id)
            if not primary_dataset:
                raise ValueError(f"Dataset not found: {query.dataset_id}")
            
            # Apply joins if specified
            working_df = await self._apply_joins(primary_dataset, query.joins)
            
            # Apply filters
            working_df = self._apply_filters(working_df, query.filters)
            
            # Execute aggregation
            result_df = self._execute_aggregation(working_df, query.metrics, query.group_by)
            
            # Apply limit if specified
            if query.limit and len(result_df) > query.limit:
                result_df = result_df.head(query.limit)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Format results
            return await self._format_results(
                result_df=result_df,
                query=query,
                execution_time=execution_time,
                datasets_used=[query.dataset_id] + ([j.target_dataset_id for j in query.joins] if query.joins else [])
            )
            
        except Exception as e:
            # Re-raise with context
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    async def _apply_joins(self, primary_dataset: Dataset, joins: Optional[List]) -> pd.DataFrame:
        """
        Apply joins to create working dataframe.
        
        Returns:
            Joined dataframe (copy, original datasets untouched)
        """
        if not joins:
            return primary_dataset.get_dataframe_copy()
        
        # MVP: Only support one join
        if len(joins) > 1:
            raise ValueError("MVP supports only one join per query")
        
        join_def = joins[0]
        
        # Get secondary dataset
        secondary_dataset = self.registry.get_dataset(join_def.target_dataset_id)
        if not secondary_dataset:
            raise ValueError(f"Join dataset not found: {join_def.target_dataset_id}")
        
        # Perform join
        primary_df = primary_dataset.get_dataframe_copy()
        secondary_df = secondary_dataset.get_dataframe_copy()
        
        # Ensure join columns exist
        if join_def.left_key not in primary_df.columns:
            raise ValueError(f"Join column '{join_def.left_key}' not found in primary dataset")
        if join_def.right_key not in secondary_df.columns:
            raise ValueError(f"Join column '{join_def.right_key}' not found in secondary dataset")
        
        # Perform the join
        joined_df = pd.merge(
            left=primary_df,
            right=secondary_df,
            left_on=join_def.left_key,
            right_on=join_def.right_key,
            how=join_def.join_type,
            suffixes=('', '_right')
        )
        
        return joined_df
    
    def _apply_filters(self, df: pd.DataFrame, filters: Optional[List]) -> pd.DataFrame:
        """Apply filters to dataframe."""
        if not filters:
            return df
        
        filtered_df = df.copy()
        
        for filter_cond in filters:
            col = filter_cond.column
            op = filter_cond.operator
            val = filter_cond.value
            
            if col not in filtered_df.columns:
                raise ValueError(f"Filter column '{col}' not found")
            
            if op == "=":
                filtered_df = filtered_df[filtered_df[col] == val]
            elif op == "!=":
                filtered_df = filtered_df[filtered_df[col] != val]
            elif op == ">":
                filtered_df = filtered_df[filtered_df[col] > val]
            elif op == "<":
                filtered_df = filtered_df[filtered_df[col] < val]
            elif op == ">=":
                filtered_df = filtered_df[filtered_df[col] >= val]
            elif op == "<=":
                filtered_df = filtered_df[filtered_df[col] <= val]
            elif op == "in":
                filtered_df = filtered_df[filtered_df[col].isin(val)]
            elif op == "not_in":
                filtered_df = filtered_df[~filtered_df[col].isin(val)]
            elif op == "contains":
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val))]
            elif op == "is_null":
                filtered_df = filtered_df[filtered_df[col].isna()]
            elif op == "not_null":
                filtered_df = filtered_df[~filtered_df[col].isna()]
        
        return filtered_df
    
    def _execute_aggregation(self, df: pd.DataFrame, metrics: List[str], group_by: Optional[List[str]]) -> pd.DataFrame:
        """
        Execute aggregation on dataframe.
        
        Returns:
            Aggregated dataframe
        """
        from core.query_model import Query
        
        if df.empty:
            # Return empty result with proper columns
            result_cols = []
            if group_by:
                result_cols.extend(group_by)
            for metric in metrics:
                func, col = Query("temp", [metric], None)._parse_metric(metric)
                result_cols.append(metric)
            return pd.DataFrame(columns=result_cols)
        
        # Parse metrics
# Parse metrics
        aggregations = {}
        for metric in metrics:
            func, col = Query("temp", [metric], None)._parse_metric(metric)
            
            # --- FIX START: Map 'avg' to 'mean' for Pandas ---
            if func == "avg":
                func = "mean"
            # -------------------------------------------------

            if func == "count":
                if col:  # count(column) - count non-null
                    aggregations[metric] = pd.NamedAgg(column=col, aggfunc='count')
                else:    # count() - count rows
                    # Add a dummy column for counting
                    df['_count'] = 1
                    aggregations[metric] = pd.NamedAgg(column='_count', aggfunc='count')
            else:
                aggregations[metric] = pd.NamedAgg(column=col, aggfunc=func)
        
        # Execute aggregation
        if group_by:
            # Group by specified columns
            result_df = df.groupby(group_by, as_index=False).agg(**aggregations)
        else:
            # Overall aggregation (no group by)
            # Create a dummy column for grouping
            df['_group'] = 1
            result_df = df.groupby('_group', as_index=False).agg(**aggregations)
            result_df = result_df.drop('_group', axis=1)
        
        return result_df
    
    async def _format_results(self, result_df: pd.DataFrame, query: Query, 
                            execution_time: float, datasets_used: List[str]) -> Dict[str, Any]:
        """Format query results into UI-ready format."""
        
        # Build provenance
        provenance = {
            "datasets": datasets_used,
            "columns": self._extract_columns_used(query),
            "operations": query.metrics,
            "group_by": query.group_by or [],
            "execution_time": execution_time,
            "row_count": len(result_df),
            "generated_at": datetime.now().isoformat()
        }
        
        # Generate statement
        statement = await self.statement_formatter.format(
            result_df=result_df,
            query=query,
            provenance=provenance
        )
        
        # Generate table view
        table_data = self.table_formatter.format(result_df)
        
        # Generate chart data (if applicable)
        chart_data = None
        if query.group_by and len(query.group_by) == 1 and len(query.metrics) == 1:
            chart_data = self.chart_formatter.format(
                result_df=result_df,
                x_column=query.group_by[0],
                y_column=query.metrics[0],
                chart_type="bar"  # Default for grouped queries
            )
        
        # Build summary statistics
        summary = self._build_summary(result_df, query)
        
        return {
            "summary": summary,
            "statement": statement,
            "table": table_data,
            "chart": chart_data,
            "provenance": provenance,
            "query_id": query.id,
            "success": True
        }
    
    def _extract_columns_used(self, query: Query) -> List[str]:
        """Extract all columns used in query."""
        columns = set()
        
        # From metrics
        for metric in query.metrics:
            _, col = query._parse_metric(metric)
            if col:
                columns.add(col)
        
        # From group by
        if query.group_by:
            columns.update(query.group_by)
        
        # From filters
        if query.filters:
            for filter_cond in query.filters:
                columns.add(filter_cond.column)
        
        # From joins
        if query.joins:
            for join in query.joins:
                columns.add(join.left_key)
                # Right key is from other dataset, but track it
                columns.add(join.right_key)
        
        return list(columns)
    
    def _build_summary(self, result_df: pd.DataFrame, query: Query) -> Dict[str, Any]:
        """Build summary statistics from results."""
        if result_df.empty:
            return {"message": "No results", "row_count": 0}
        
        summary = {
            "row_count": len(result_df),
            "metric_count": len(query.metrics),
            "has_grouping": bool(query.group_by)
        }
        
        # Add first few values for quick glance
        if not result_df.empty:
            first_row = result_df.iloc[0]
            for i, col in enumerate(result_df.columns):
                if i < 5:  # First 5 columns
                    summary[f"first_{col}"] = str(first_row[col])
        
        return summary