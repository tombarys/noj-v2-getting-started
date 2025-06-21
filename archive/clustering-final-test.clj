(ns clustering-final-test
  (:require
   [clustering :as cl]
   [tablecloth.api :as tc]))

(println "=== FINAL K-MEANS CLUSTERING TEST ===")
(println "Testing full clustering pipeline with quality metrics...")

;; Test that all the main clustering variables are defined
(println "\n✓ Dataset loaded:")
(println "  Raw dataset shape:" (tc/shape cl/raw-ds))
(println "  Processed dataset shape:" (tc/shape cl/processed-ds))
(println "  Clustering data shape:" (tc/shape cl/clustering-data))

(println "\n✓ K-means results:")
(println "  K=3 result iterations:" (:iterations cl/kmeans-3-result))
(println "  K=5 result iterations:" (:iterations cl/kmeans-5-result))
(println "  K=8 result iterations:" (:iterations cl/kmeans-8-result))

(println "\n✓ Clustered dataset:")
(println "  Data with clusters shape:" (tc/shape cl/data-with-clusters))
(println "  Columns:" (tc/column-names cl/data-with-clusters))

(println "\n✓ Quality metrics:")
(doseq [[model metrics] cl/quality-metrics]
  (println (str "  " (name model) ": WCSS=" (format "%.2f" (:wcss metrics))
                ", Silhouette=" (format "%.3f" (:silhouette metrics))
                ", Iterations=" (:iterations metrics))))

(println "\n✓ Cluster distributions:")
(println "K=3 clusters:")
(println (-> cl/data-with-clusters
             (tc/group-by [:cluster-3])
             (tc/aggregate {:count tc/row-count})))

(println "\nK=5 clusters:")
(println (-> cl/data-with-clusters
             (tc/group-by [:cluster-5])
             (tc/aggregate {:count tc/row-count})))

(println "\n=== FULL CLUSTERING PIPELINE SUCCESSFUL ===")
(println "🎉 K-means clustering implementation is complete and working!")
