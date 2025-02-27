def get_hybrid_recommendations(user_id, top_n=5, weight=0.6):
    collab_recs = get_collaborative_recommendations(user_id, top_n=10)
    content_recs = get_similar_products(collab_recs[0], top_n=10) if collab_recs else []

    # Adjust weight (higher means more collaborative influence)
    collab_weight = weight
    content_weight = 1 - weight

    final_recs = list(set(collab_recs[:int(top_n * collab_weight)] + content_recs[:int(top_n * content_weight)]))
    
    return final_recs[:top_n]
