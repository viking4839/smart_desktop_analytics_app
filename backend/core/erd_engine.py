"""
Enhanced ERD Engine with Advanced Features
- Smart relationship discovery
- Join preview with quality metrics
- Composite key detection
- Data lineage tracking
- Schema improvement suggestions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ERDEngine:
    """Advanced ERD analysis engine"""
    
    def __init__(self):
        self.relationship_cache = {}
    
    def analyze_relationships(self, anchor_df: pd.DataFrame, new_df: pd.DataFrame, 
                            anchor_name: str, new_name: str,
                            include_composites: bool = True) -> List[Dict[str, Any]]:
        """
        Analyzes two dataframes and returns potential relationships with confidence scores.
        
        Features:
        - Single column FK detection
        - Composite key detection (optional)
        - Data overlap analysis
        - Cardinality detection
        """
        relationships = []
        
        # 1. Find single-column relationships
        single_col_rels = self._find_single_column_relationships(
            anchor_df, new_df, anchor_name, new_name
        )
        relationships.extend(single_col_rels)
        
        # 2. Find composite key relationships (if enabled)
        if include_composites:
            composite_rels = self._find_composite_relationships(
                anchor_df, new_df, anchor_name, new_name
            )
            relationships.extend(composite_rels)
        
        # 3. Sort by confidence
        return sorted(relationships, key=lambda x: x['confidence_score'], reverse=True)
    
    def _find_single_column_relationships(self, anchor_df: pd.DataFrame, 
                                         new_df: pd.DataFrame,
                                         anchor_name: str, 
                                         new_name: str) -> List[Dict[str, Any]]:
        """Find single-column FK relationships"""
        relationships = []
        
        # Get ID-like columns
        anchor_cols = self._get_id_columns(anchor_df)
        new_cols = self._get_id_columns(new_df)
        
        for a_col in anchor_cols:
            # Check if anchor column is a good PK candidate
            pk_score = self._evaluate_primary_key(anchor_df, a_col)
            
            # Skip poor PK candidates
            if pk_score < 20:
                continue
            
            anchor_set = set(anchor_df[a_col].dropna().astype(str))
            
            for n_col in new_cols:
                confidence = 0
                reasons = []
                metadata = {}
                
                # 1. Name similarity
                name_score, name_reason = self._compare_column_names(a_col, n_col)
                confidence += name_score
                if name_reason:
                    reasons.append(name_reason)
                
                # 2. Data overlap
                new_set = set(new_df[n_col].dropna().astype(str))
                if len(new_set) > 0:
                    intersection = new_set.intersection(anchor_set)
                    overlap_pct = (len(intersection) / len(new_set)) * 100
                    
                    if overlap_pct > 10:
                        overlap_score = min((overlap_pct / 100) * 60, 60)
                        confidence += overlap_score
                        reasons.append(f"{overlap_pct:.1f}% data overlap")
                        
                        # Calculate additional metrics
                        metadata['overlap_percentage'] = round(overlap_pct, 2)
                        metadata['matched_values'] = len(intersection)
                        metadata['total_values'] = len(new_set)
                
                # 3. Cardinality detection
                cardinality = self._detect_cardinality(anchor_df, new_df, a_col, n_col)
                metadata['cardinality'] = cardinality
                reasons.append(f"{cardinality} relationship")
                
                # 4. Data type compatibility
                if self._are_types_compatible(anchor_df[a_col], new_df[n_col]):
                    confidence += 10
                    reasons.append("Compatible data types")
                
                # Only return high-confidence relationships
                if confidence >= 40:
                    relationships.append({
                        "anchor_dataset": anchor_name,
                        "anchor_column": a_col,
                        "new_dataset": new_name,
                        "new_column": n_col,
                        "confidence_score": min(int(confidence), 100),
                        "reasons": reasons,
                        "metadata": metadata,
                        "relationship_type": "single_column",
                        "pk_quality_score": pk_score
                    })
        
        return relationships
    
    def _find_composite_relationships(self, anchor_df: pd.DataFrame, 
                                     new_df: pd.DataFrame,
                                     anchor_name: str, 
                                     new_name: str) -> List[Dict[str, Any]]:
        """Find composite (multi-column) key relationships"""
        relationships = []
        
        anchor_cols = self._get_id_columns(anchor_df)
        new_cols = self._get_id_columns(new_df)
        
        # Test pairs of columns
        for i, a_col1 in enumerate(anchor_cols):
            for a_col2 in anchor_cols[i+1:]:
                # Check if combination is unique enough
                anchor_combined = anchor_df[[a_col1, a_col2]].fillna('NULL')
                anchor_combined_unique = len(anchor_combined.drop_duplicates())
                
                if anchor_combined_unique / len(anchor_df) < 0.7:
                    continue  # Not unique enough
                
                # Create composite key
                anchor_composite = anchor_combined.apply(
                    lambda row: f"{row[a_col1]}|{row[a_col2]}", axis=1
                )
                anchor_set = set(anchor_composite)
                
                # Look for matching pairs in new dataset
                for j, n_col1 in enumerate(new_cols):
                    for n_col2 in new_cols[j+1:]:
                        new_combined = new_df[[n_col1, n_col2]].fillna('NULL')
                        new_composite = new_combined.apply(
                            lambda row: f"{row[n_col1]}|{row[n_col2]}", axis=1
                        )
                        new_set = set(new_composite)
                        
                        # Calculate overlap
                        intersection = new_set.intersection(anchor_set)
                        if len(new_set) == 0:
                            continue
                        
                        overlap_pct = (len(intersection) / len(new_set)) * 100
                        
                        if overlap_pct > 30:  # Higher threshold for composites
                            confidence = min(50 + (overlap_pct / 100) * 50, 100)
                            
                            relationships.append({
                                "anchor_dataset": anchor_name,
                                "anchor_column": f"{a_col1} + {a_col2}",
                                "new_dataset": new_name,
                                "new_column": f"{n_col1} + {n_col2}",
                                "confidence_score": int(confidence),
                                "reasons": [
                                    "Composite key",
                                    f"{overlap_pct:.1f}% composite match"
                                ],
                                "metadata": {
                                    "overlap_percentage": round(overlap_pct, 2),
                                    "composite_columns_anchor": [a_col1, a_col2],
                                    "composite_columns_new": [n_col1, n_col2]
                                },
                                "relationship_type": "composite",
                                "pk_quality_score": 90
                            })
        
        return relationships
    
    def preview_join(self, anchor_df: pd.DataFrame, new_df: pd.DataFrame,
                    anchor_col: str, new_col: str,
                    anchor_name: str, new_name: str,
                    limit: int = 20) -> Dict[str, Any]:
        """
        Preview a join with quality metrics
        """
        # Prepare join keys
        anchor_df = anchor_df.copy()
        new_df = new_df.copy()
        
        anchor_df['_join_key'] = anchor_df[anchor_col].astype(str)
        new_df['_join_key'] = new_df[new_col].astype(str)
        
        # Perform left join (new -> anchor)
        merged = new_df.merge(
            anchor_df, 
            how='left', 
            on='_join_key',
            suffixes=('', '_anchor')
        )
        
        # Calculate quality metrics
        total_new = len(new_df)
        matched = merged[f'{anchor_col}_anchor'].notna().sum()
        orphan_count = total_new - matched
        orphan_pct = (orphan_count / total_new * 100) if total_new else 0
        
        # Check anchor key uniqueness
        anchor_unique_count = anchor_df['_join_key'].nunique()
        anchor_total = len(anchor_df)
        is_unique = anchor_unique_count == anchor_total
        duplicate_count = anchor_total - anchor_unique_count
        
        # Detect cardinality
        cardinality = self._detect_cardinality(anchor_df, new_df, anchor_col, new_col)
        
        # Get sample data
        preview_df = merged.drop(columns=['_join_key'], errors='ignore').head(limit)
        preview_df = self._sanitize_for_json(preview_df)
        
        return {
            "anchor_dataset": anchor_name,
            "new_dataset": new_name,
            "anchor_column": anchor_col,
            "new_column": new_col,
            "preview": {
                "columns": preview_df.columns.tolist(),
                "rows": preview_df.to_dict(orient='records')
            },
            "quality": {
                "total_new_rows": total_new,
                "matched_rows": int(matched),
                "orphan_rows": int(orphan_count),
                "orphan_percentage": round(orphan_pct, 2),
                "match_percentage": round(100 - orphan_pct, 2),
                "anchor_key_unique": is_unique,
                "anchor_duplicate_keys": int(duplicate_count),
                "cardinality": cardinality
            },
            "warnings": self._generate_warnings(
                orphan_pct, is_unique, duplicate_count, cardinality
            )
        }
    
    def suggest_schema_improvements(self, df: pd.DataFrame, 
                                   dataset_name: str) -> List[Dict[str, Any]]:
        """
        Suggest schema improvements like normalization, PK promotion, etc.
        """
        suggestions = []
        
        for col in df.columns:
            # 1. Suggest PK promotion
            uniqueness = df[col].nunique() / len(df) * 100
            if uniqueness > 95 and df[col].notna().sum() == len(df):
                suggestions.append({
                    "type": "promote_to_pk",
                    "column": col,
                    "reason": f"Column is {uniqueness:.1f}% unique with no nulls",
                    "confidence": "high",
                    "action": f"Use '{col}' as primary key"
                })
            
            # 2. Suggest normalization
            if df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                if len(value_counts) < len(df) * 0.1:  # Low cardinality
                    repetition_pct = (len(df) - len(value_counts)) / len(df) * 100
                    if repetition_pct > 50:
                        suggestions.append({
                            "type": "extract_lookup_table",
                            "column": col,
                            "reason": f"Column has high repetition ({repetition_pct:.0f}%)",
                            "confidence": "medium",
                            "action": f"Extract '{col}' to separate lookup table",
                            "metadata": {
                                "unique_values": len(value_counts),
                                "total_rows": len(df)
                            }
                        })
        
        return suggestions
    
    def find_data_lineage(self, datasets: Dict[str, pd.DataFrame],
                         start_dataset: str, end_dataset: str) -> List[List[str]]:
        """
        Find all possible paths from start_dataset to end_dataset
        Returns list of paths, where each path is a list of dataset names
        """
        # Build relationship graph
        graph = {}
        for anchor_name, anchor_df in datasets.items():
            graph[anchor_name] = []
            for other_name, other_df in datasets.items():
                if anchor_name != other_name:
                    rels = self.analyze_relationships(
                        anchor_df, other_df, anchor_name, other_name,
                        include_composites=False
                    )
                    if rels:  # If any relationship exists
                        graph[anchor_name].append(other_name)
        
        # Find paths using DFS
        paths = []
        self._dfs_find_paths(graph, start_dataset, end_dataset, [], paths)
        return paths
    
    def _dfs_find_paths(self, graph: Dict, current: str, target: str,
                       path: List[str], all_paths: List[List[str]]):
        """DFS helper for lineage tracking"""
        path = path + [current]
        
        if current == target:
            all_paths.append(path)
            return
        
        if current not in graph:
            return
        
        for node in graph[current]:
            if node not in path:  # Avoid cycles
                self._dfs_find_paths(graph, node, target, path, all_paths)
    
    # ========== Helper Methods ==========
    
    def _get_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that could be IDs"""
        id_cols = []
        for col in df.columns:
            # Check data type
            if df[col].dtype in ['int64', 'object', 'string', 'Int64']:
                # Check cardinality
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.1:  # At least 10% unique
                    id_cols.append(col)
        return id_cols
    
    def _evaluate_primary_key(self, df: pd.DataFrame, col: str) -> int:
        """Score how good a column is as a PK (0-100)"""
        score = 0
        
        # Uniqueness
        uniqueness = df[col].nunique() / len(df) * 100
        score += min(uniqueness, 70)
        
        # No nulls
        null_pct = df[col].isna().sum() / len(df) * 100
        if null_pct == 0:
            score += 20
        elif null_pct < 5:
            score += 10
        
        # Name contains 'id'
        if 'id' in col.lower():
            score += 10
        
        return min(int(score), 100)
    
    def _compare_column_names(self, col1: str, col2: str) -> Tuple[int, str]:
        """Compare column names and return score + reason"""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        if col1_lower == col2_lower:
            return 40, "Exact column name match"
        
        if 'id' in col1_lower and 'id' in col2_lower:
            # Check for patterns like "customer_id" vs "id"
            if col1_lower.replace('_', '') in col2_lower.replace('_', '') or \
               col2_lower.replace('_', '') in col1_lower.replace('_', ''):
                return 30, "Similar ID column names"
            return 15, "Both contain 'id'"
        
        return 0, ""
    
    def _detect_cardinality(self, anchor_df: pd.DataFrame, new_df: pd.DataFrame,
                          anchor_col: str, new_col: str) -> str:
        """Detect relationship cardinality"""
        anchor_unique = anchor_df[anchor_col].nunique()
        new_unique = new_df[new_col].nunique()
        
        anchor_total = len(anchor_df)
        new_total = len(new_df)
        
        anchor_is_unique = anchor_unique == anchor_total
        new_is_unique = new_unique == new_total
        
        if anchor_is_unique and new_is_unique:
            return "1:1"
        elif anchor_is_unique and not new_is_unique:
            return "1:N"
        elif not anchor_is_unique and new_is_unique:
            return "N:1"
        else:
            return "M:N"
    
    def _are_types_compatible(self, series1: pd.Series, series2: pd.Series) -> bool:
        """Check if two series have compatible data types"""
        type1 = series1.dtype
        type2 = series2.dtype
        
        # Numeric types are compatible with each other
        if pd.api.types.is_numeric_dtype(type1) and pd.api.types.is_numeric_dtype(type2):
            return True
        
        # String types are compatible
        if type1 == 'object' and type2 == 'object':
            return True
        
        # Same type
        if type1 == type2:
            return True
        
        return False
    
    def _generate_warnings(self, orphan_pct: float, is_unique: bool,
                          duplicate_count: int, cardinality: str) -> List[str]:
        """Generate warnings based on join quality"""
        warnings = []
        
        if orphan_pct > 20:
            warnings.append(
                f"⚠️ High orphan rate: {orphan_pct:.1f}% of new records don't match anchor"
            )
        
        if not is_unique:
            warnings.append(
                f"⚠️ Anchor column has {duplicate_count} duplicates - not a valid PK"
            )
        
        if cardinality == "M:N":
            warnings.append(
                "⚠️ Many-to-many relationship detected - may cause data explosion"
            )
        
        return warnings
    
    def _sanitize_for_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize DataFrame for JSON serialization"""
        df = df.copy()
        
        # Convert datetime
        for col in df.select_dtypes(include=['datetime64']):
            df[col] = df[col].astype(str)
        
        # Replace NaN/Inf
        df = df.replace([np.inf, -np.inf], None)
        df = df.where(pd.notna(df), None)
        
        return df


# Singleton instance
_erd_engine = None

def get_erd_engine() -> ERDEngine:
    """Get singleton ERD engine instance"""
    global _erd_engine
    if _erd_engine is None:
        _erd_engine = ERDEngine()
    return _erd_engine


# Legacy function for backward compatibility
def analyze_relationships(anchor_df: pd.DataFrame, new_df: pd.DataFrame,
                         anchor_name: str, new_name: str) -> List[Dict[str, Any]]:
    """Legacy wrapper for analyze_relationships"""
    engine = get_erd_engine()
    return engine.analyze_relationships(anchor_df, new_df, anchor_name, new_name)