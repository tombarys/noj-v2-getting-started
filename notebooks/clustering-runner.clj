(ns clustering-runner
  (:require
   [clustering :as cl]
   [tablecloth.api :as tc]))

(println "=== CLUSTERING RUNNER ===")
(println "Dataset loaded successfully!")
(println "Processed dataset shape:" (tc/shape cl/processed-ds))
(println "Clustering data shape:" (tc/shape cl/clustering-data))

;; Test the k-means algorithm with simple data
(println "\n=== TESTING K-MEANS ALGORITHM ===")

;; Run k-means with 2 clusters
(def test-kmeans-result
  (cl/k-means cl/clustering-data 2 20))

(println "K-means completed!")
(println "Number of iterations:" (:iterations test-kmeans-result))
(println "Cluster labels:" (:labels test-kmeans-result))
(println "Number of centroids:" (count (:centroids test-kmeans-result)))

;; Add cluster labels to dataset
(def test-data-with-clusters
  (tc/add-column cl/clustering-data :cluster (:labels test-kmeans-result)))

(println "\nCluster distribution:")
(println (-> test-data-with-clusters
             (tc/group-by [:cluster])
             (tc/aggregate {:count tc/row-count})))

(println "\n=== K-MEANS TEST COMPLETED ===")
