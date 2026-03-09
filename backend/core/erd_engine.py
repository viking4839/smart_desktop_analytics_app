import pandas as pd

def analyze_relationships(anchor_df, new_df, anchor_name, new_name):
    """
    Analyzes two dataframes and returns potential relationships with a confidence score.
    """
    relationships = []
    
    # Get columns that are likely IDs (strings or integers)
    anchor_cols = [c for c in anchor_df.columns if anchor_df[c].dtype in ['int64', 'object', 'string']]
    new_cols = [c for c in new_df.columns if new_df[c].dtype in ['int64', 'object', 'string']]

    for a_col in anchor_cols:
        # Check if Anchor column could be a Primary Key (high uniqueness)
        a_unique_count = anchor_df[a_col].nunique()
        a_total_count = len(anchor_df[a_col].dropna())
        
        # Skip columns that are clearly not IDs (e.g., a "Gender" column with only 2 unique values)
        if a_total_count == 0 or (a_unique_count / a_total_count) < 0.1:
            continue
            
        # Convert anchor column to a fast lookup set
        anchor_set = set(anchor_df[a_col].dropna().astype(str))

        for n_col in new_cols:
            confidence = 0
            reasons = []

            # 1. Name Heuristic
            if a_col.lower() == n_col.lower():
                confidence += 40
                reasons.append("Exact column name match")
            elif "id" in a_col.lower() and "id" in n_col.lower():
                confidence += 15
                reasons.append("Both columns contain 'id'")

            # 2. Data Overlap (The real proof)
            new_set = set(new_df[n_col].dropna().astype(str))
            
            if len(new_set) > 0:
                # Find how many items in the new column exist in the anchor column
                intersection = new_set.intersection(anchor_set)
                overlap_percentage = len(intersection) / len(new_set) * 100

                if overlap_percentage > 10: # Only care if at least 10% match
                    # Scale overlap to a max of 60 confidence points
                    overlap_score = (overlap_percentage / 100) * 60
                    confidence += overlap_score
                    reasons.append(f"{overlap_percentage:.1f}% data overlap")

            # 3. Compile the Result if confidence is high enough
            if confidence >= 40: # Threshold to show to the user
                # Cap at 100%
                final_confidence = min(round(confidence), 100)
                
                relationships.append({
                    "anchor_dataset": anchor_name,
                    "anchor_column": a_col,
                    "new_dataset": new_name,
                    "new_column": n_col,
                    "confidence_score": final_confidence,
                    "reasons": reasons
                })

    # Sort by highest confidence first
    return sorted(relationships, key=lambda x: x['confidence_score'], reverse=True)