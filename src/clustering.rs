use anyhow::Result;
use std::collections::HashMap;

/// Cluster result containing cluster assignments
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ClusterResult {
    /// Map from chunk_id to cluster_id
    pub assignments: HashMap<i64, i64>,
    /// Number of clusters found
    pub num_clusters: usize,
    /// Noise/outlier points (cluster_id = -1)
    pub noise_points: Vec<i64>,
}

/// Simple k-means clustering on embeddings
pub fn cluster_embeddings(
    embeddings: &[(i64, Vec<f32>)], // (chunk_id, vector)
    k: usize,
) -> Result<ClusterResult> {
    if embeddings.is_empty() {
        return Ok(ClusterResult {
            assignments: HashMap::new(),
            num_clusters: 0,
            noise_points: Vec::new(),
        });
    }

    let dims = embeddings[0].1.len();
    let n = embeddings.len();

    // Use fewer clusters for small datasets
    let k = k.min(n / 3).max(1);

    // Initialize centroids using k-means++ strategy
    let mut centroids = kmeans_plus_plus_init(embeddings, k);

    // Run k-means iterations
    let max_iters = 100;
    let mut assignments = vec![0; n];

    for _ in 0..max_iters {
        let mut changed = false;

        // Assignment step: assign each point to nearest centroid
        for (i, (_chunk_id, embedding)) in embeddings.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best_cluster = 0;

            for (c_idx, centroid) in centroids.iter().enumerate() {
                let dist = cosine_distance(embedding, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c_idx;
                }
            }

            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recalculate centroids
        let mut new_centroids = vec![vec![0.0; dims]; k];
        let mut counts = vec![0; k];

        for (i, (_chunk_id, embedding)) in embeddings.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for d in 0..dims {
                new_centroids[cluster][d] += embedding[d];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dims {
                    new_centroids[c][d] /= counts[c] as f32;
                }
                // Normalize
                let norm: f32 = new_centroids[c].iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for d in 0..dims {
                        new_centroids[c][d] /= norm;
                    }
                }
            } else {
                // Empty cluster - reinitialize to a random point
                if !embeddings.is_empty() {
                    let idx = c % embeddings.len();
                    new_centroids[c] = embeddings[idx].1.clone();
                }
            }
        }

        centroids = new_centroids;
    }

    // Build result
    let mut result_map = HashMap::new();
    for (i, (chunk_id, _)) in embeddings.iter().enumerate() {
        result_map.insert(*chunk_id, assignments[i] as i64);
    }

    Ok(ClusterResult {
        assignments: result_map,
        num_clusters: k,
        noise_points: Vec::new(),
    })
}

/// K-means++ initialization for better starting centroids
fn kmeans_plus_plus_init(embeddings: &[(i64, Vec<f32>)], k: usize) -> Vec<Vec<f32>> {
    let mut centroids = Vec::new();
    let n = embeddings.len();

    if n == 0 || k == 0 {
        return centroids;
    }

    // Choose first centroid randomly
    centroids.push(embeddings[0].1.clone());

    // Choose remaining centroids
    for _ in 1..k {
        let mut distances = vec![f32::MAX; n];

        // Calculate distance to nearest existing centroid
        for (i, (_chunk_id, embedding)) in embeddings.iter().enumerate() {
            for centroid in &centroids {
                let dist = cosine_distance(embedding, centroid);
                distances[i] = distances[i].min(dist);
            }
        }

        // Choose next centroid proportional to squared distance
        let sum_dist: f32 = distances.iter().sum();
        if sum_dist <= 0.0 {
            // Fallback to next available point
            if centroids.len() < n {
                centroids.push(embeddings[centroids.len()].1.clone());
            }
            continue;
        }

        let mut cumulative = 0.0;
        let threshold = sum_dist * 0.5; // Pick median distance point
        let mut chosen_idx = 0;

        for (i, &dist) in distances.iter().enumerate() {
            cumulative += dist;
            if cumulative >= threshold {
                chosen_idx = i;
                break;
            }
        }

        centroids.push(embeddings[chosen_idx].1.clone());
    }

    centroids
}

/// Calculate cosine distance between two vectors (1 - cosine_similarity)
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // Assuming normalized vectors, cosine similarity = dot product
    // Cosine distance = 1 - similarity
    1.0 - dot.max(-1.0).min(1.0)
}

/// Get cluster sizes
pub fn get_cluster_sizes(result: &ClusterResult) -> HashMap<i64, usize> {
    let mut sizes = HashMap::new();
    for &cluster_id in result.assignments.values() {
        *sizes.entry(cluster_id).or_insert(0) += 1;
    }
    sizes
}

/// Get chunks in a specific cluster
#[allow(dead_code)]
pub fn get_cluster_members(result: &ClusterResult, cluster_id: i64) -> Vec<i64> {
    result
        .assignments
        .iter()
        .filter(|&(_, &cid)| cid == cluster_id)
        .map(|(&chunk_id, _)| chunk_id)
        .collect()
}
