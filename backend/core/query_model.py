"""
Query data models for analytics operations.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


@dataclass
class JoinDefinition:
    """Definition of a dataset join."""
    target_dataset_id: str
    left_key: str
    right_key: str
    join_type: str = "inner"  # inner, left, right, outer
    
    def validate(self):
        """Validate join definition."""
        if self.join_type not in ["inner", "left", "right", "outer"]:
            raise ValueError(f"Invalid join type: {self.join_type}")
        if not self.left_key or not self.right_key:
            raise ValueError("Join keys cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_dataset_id": self.target_dataset_id,
            "left_key": self.left_key,
            "right_key": self.right_key,
            "join_type": self.join_type
        }


@dataclass
class FilterCondition:
    """Single filter condition."""
    column: str
    operator: str  # =, !=, >, <, >=, <=, in, not_in, contains
    value: Any
    
    def validate(self):
        """Validate filter condition."""
        valid_operators = ["=", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "is_null", "not_null"]
        if self.operator not in valid_operators:
            raise ValueError(f"Invalid operator: {self.operator}. Valid: {valid_operators}")


@dataclass
class Query:
    """
    Analytics query definition.
    This is the main object that gets serialized between frontend and backend.
    
    IMPORTANT: Fields are ordered - required fields first, then optional with defaults!
    """
    # ========== REQUIRED FIELDS (no defaults) ==========
    dataset_id: str
    metrics: List[str]  # e.g., ["sum(revenue)", "avg(price)", "count()"]
    
    # ========== OPTIONAL FIELDS (with defaults) ==========
    id: str = field(default_factory=lambda: f"qry_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    group_by: Optional[List[str]] = None
    filters: Optional[List[FilterCondition]] = None
    joins: Optional[List[JoinDefinition]] = None
    limit: Optional[int] = None
    query_type: str = "aggregation"  # aggregation, trend, comparison
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate query after initialization."""
        if not self.metrics:
            raise ValueError("Query must have at least one metric")
        
        if self.joins:
            for join in self.joins:
                join.validate()
        
        if self.filters:
            for filter_cond in self.filters:
                filter_cond.validate()
        
        # Parse metrics to validate syntax
        for metric in self.metrics:
            self._parse_metric(metric)
    
    def _parse_metric(self, metric: str) -> tuple:
        """Parse metric string into (function, column)."""
        # Patterns: sum(revenue), avg(price), count(), revenue (implicit sum)
        pattern = r'^(\w+)\(([^)]*)\)$|^(\w+)$'
        match = re.match(pattern, metric)
        
        if not match:
            raise ValueError(f"Invalid metric format: {metric}")
        
        func, col_within, col_only = match.groups()
        
        if func:  # Function style: sum(revenue)
            valid_funcs = ["sum", "avg", "mean", "count", "min", "max", "std", "var"]
            if func not in valid_funcs:
                raise ValueError(f"Invalid function: {func}. Valid: {valid_funcs}")
            return func, col_within if col_within else None
        else:  # Column only: revenue (implies sum)
            return "sum", col_only
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize query to dictionary."""
        result = {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "metrics": self.metrics,
            "group_by": self.group_by or [],
            "filters": [{"column": f.column, "operator": f.operator, "value": f.value} 
                       for f in (self.filters or [])],
            "joins": [j.to_dict() for j in (self.joins or [])],
            "limit": self.limit,
            "query_type": self.query_type,
            "created_at": self.created_at.isoformat(),
            "description": self.description
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """Create Query from dictionary."""
        # Handle filters
        filters = None
        if data.get("filters"):
            filters = [FilterCondition(**f) for f in data["filters"]]
        
        # Handle joins
        joins = None
        if data.get("joins"):
            joins = [JoinDefinition(**j) for j in data["joins"]]
        
        return cls(
            dataset_id=data["dataset_id"],  # Required field first
            metrics=data["metrics"],          # Required field second
            id=data.get("id"),                # Optional fields after
            group_by=data.get("group_by"),
            filters=filters,
            joins=joins,
            limit=data.get("limit"),
            query_type=data.get("query_type", "aggregation"),
            description=data.get("description")
        )