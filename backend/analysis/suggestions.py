"""
Auto-suggest analytics queries based on dataset characteristics.
Provides intelligent query recommendations for users.
"""
from typing import List, Dict, Any
from core.dataset import Dataset, DataType


class AnalysisSuggestions:
    """Generates suggested queries based on dataset schema."""
    
    @staticmethod
    def generate_suggestions(dataset: Dataset) -> List[Dict[str, Any]]:
        """
        Generate suggested analytics queries for a dataset.
        
        Args:
            dataset: Dataset object
            
        Returns:
            List of suggested query definitions
        """
        suggestions = []
        
        # Basic overview
        suggestions.append({
            "id": "overview",
            "name": "📊 Dataset Overview",
            "description": "Basic statistics about the dataset",
            "query": {
                "dataset_id": dataset.id,
                "metrics": ["count()"],
                "query_type": "aggregation"
            },
            "category": "overview",
            "auto_generated": True
        })
        
        # Analyze numeric columns
        numeric_columns = [
            col for col, schema in dataset.schema.items() 
            if schema.data_type in [DataType.INTEGER, DataType.FLOAT, 
                                   DataType.CURRENCY, DataType.PERCENTAGE]
        ]
        
        for col in numeric_columns[:3]:  # First 3 numeric columns
            suggestions.append({
                "id": f"stats_{col}",
                "name": f"📈 Statistics for {col}",
                "description": f"Summary statistics for {col}",
                "query": {
                    "dataset_id": dataset.id,
                    "metrics": [
                        f"sum({col})",
                        f"avg({col})",
                        f"min({col})",
                        f"max({col})"
                    ],
                    "query_type": "aggregation"
                },
                "category": "statistics",
                "auto_generated": True
            })
        
        # Analyze categorical columns
        categorical_columns = [
            col for col, schema in dataset.schema.items()
            if schema.data_type == DataType.CATEGORICAL and schema.unique_values < 20
        ]
        
        for col in categorical_columns[:3]:
            suggestions.append({
                "id": f"count_by_{col}",
                "name": f"📊 Count by {col}",
                "description": f"Count records grouped by {col}",
                "query": {
                    "dataset_id": dataset.id,
                    "metrics": ["count()"],
                    "group_by": [col],
                    "query_type": "aggregation"
                },
                "category": "distribution",
                "auto_generated": True
            })
        
        # Cross-analysis between numeric and categorical
        if numeric_columns and categorical_columns:
            num_col = numeric_columns[0]
            cat_col = categorical_columns[0]
            
            suggestions.append({
                "id": f"cross_{num_col}_{cat_col}",
                "name": f"🔄 {num_col} by {cat_col}",
                "description": f"Analyze {num_col} across different {cat_col} values",
                "query": {
                    "dataset_id": dataset.id,
                    "metrics": [
                        f"sum({num_col})",
                        f"avg({num_col})",
                        "count()"
                    ],
                    "group_by": [cat_col],
                    "query_type": "aggregation"
                },
                "category": "cross_analysis",
                "auto_generated": True
            })
        
        # Currency-specific analysis
        currency_columns = [
            col for col, schema in dataset.schema.items()
            if schema.data_type == DataType.CURRENCY
        ]
        
        if currency_columns and categorical_columns:
            curr_col = currency_columns[0]
            cat_col = categorical_columns[0]
            
            suggestions.append({
                "id": f"revenue_{curr_col}_{cat_col}",
                "name": f"💰 Revenue Analysis",
                "description": f"Total and average {curr_col} by {cat_col}",
                "query": {
                    "dataset_id": dataset.id,
                    "metrics": [
                        f"sum({curr_col})",
                        f"avg({curr_col})"
                    ],
                    "group_by": [cat_col],
                    "query_type": "aggregation"
                },
                "category": "financial",
                "auto_generated": True
            })
        
        # Top N analysis
        if numeric_columns:
            num_col = numeric_columns[0]
            suggestions.append({
                "id": "top_10",
                "name": f"🏆 Top 10 by {num_col}",
                "description": f"Top 10 records sorted by {num_col}",
                "query": {
                    "dataset_id": dataset.id,
                    "metrics": [num_col],
                    "limit": 10,
                    "query_type": "aggregation"
                },
                "category": "top_n",
                "auto_generated": True
            })
        
        return suggestions
    
    @staticmethod
    def get_suggestions_by_category(dataset: Dataset) -> Dict[str, List[Dict]]:
        """
        Get suggestions organized by category.
        
        Returns:
            Dict mapping category name to list of suggestions
        """
        all_suggestions = AnalysisSuggestions.generate_suggestions(dataset)
        
        categorized = {}
        for suggestion in all_suggestions:
            category = suggestion.get("category", "other")
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(suggestion)
        
        return categorized
    
    @staticmethod
    def get_quick_insights(dataset: Dataset) -> List[str]:
        """
        Generate quick insights about the dataset.
        
        Returns:
            List of insight strings
        """
        insights = []
        
        # Data quality
        total_cells = dataset.row_count * dataset.column_count
        null_count = sum(
            schema.statistics.null_count 
            for schema in dataset.schema.values() 
            if schema.statistics
        )
        
        if null_count > 0:
            null_pct = (null_count / total_cells) * 100
            if null_pct > 20:
                insights.append(f"⚠️ {null_pct:.1f}% of data is missing")
            else:
                insights.append(f"✅ Only {null_pct:.1f}% missing data")
        else:
            insights.append("✅ No missing data - dataset is complete!")
        
        # Data types
        type_counts = {}
        for schema in dataset.schema.values():
            dt = schema.data_type.value
            type_counts[dt] = type_counts.get(dt, 0) + 1
        
        if type_counts.get('currency', 0) > 0:
            insights.append(f"💰 Contains {type_counts['currency']} currency column(s)")
        
        if type_counts.get('datetime', 0) > 0:
            insights.append(f"📅 Contains {type_counts['datetime']} date column(s)")
        
        if type_counts.get('categorical', 0) > 0:
            insights.append(f"🏷️ Contains {type_counts['categorical']} categorical column(s)")
        
        # Size
        if dataset.row_count > 10000:
            insights.append(f"📊 Large dataset with {dataset.row_count:,} rows")
        elif dataset.row_count < 100:
            insights.append(f"📊 Small dataset with {dataset.row_count} rows")
        
        return insights