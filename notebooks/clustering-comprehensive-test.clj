(ns clustering-comprehensive-test
  (:require
   [clustering :as cl]
   [tablecloth.api :as tc]))

(println "=== COMPREHENSIVE K-MEANS CLUSTERING TEST ===")

;; Test with different numbers of clusters
(defn test-multiple-clusters [data max-k]
  (println (str "\nTesting k-means with k=1 to k=" max-k))
  (doseq [k (range 1 (inc max-k))]
    (let [result (cl/k-means data k 50)
          data-points (cl/dataset-to-points data)
          wcss (cl/calculate-wcss data-points (:centroids result) (:labels result))]
      (println (str "k=" k ": iterations=" (:iterations result) 
                    ", WCSS=" (format "%.2f" wcss))))))

;; Test quality metrics
(defn test-quality-metrics [data k]
  (println (str "\n=== QUALITY METRICS FOR k=" k " ==="))
  (let [result (cl/k-means data k 50)
        data-points (cl/dataset-to-points data)
        wcss (cl/calculate-wcss data-points (:centroids result) (:labels result))
        silhouette (cl/average-silhouette-score data-points (:labels result))]
    (println (str "WCSS: " (format "%.2f" wcss)))
    (println (str "Silhouette Score: " (format "%.3f" silhouette)))
    (println (str "Iterations: " (:iterations result)))
    result))

;; Run comprehensive tests
(println "Dataset info:")
(println "Shape:" (tc/shape cl/clustering-data))
(println "Columns:" (tc/column-names cl/clustering-data))

;; Test multiple cluster numbers
(test-multiple-clusters cl/clustering-data 5)

;; Test quality metrics for k=2 and k=3
(def result-2 (test-quality-metrics cl/clustering-data 2))
(def result-3 (test-quality-metrics cl/clustering-data 3))

;; Show cluster distributions
(println "\n=== CLUSTER DISTRIBUTIONS ===")
(let [data-with-clusters-2 (tc/add-column cl/clustering-data :cluster-2 (:labels result-2))
      data-with-clusters-3 (tc/add-column cl/clustering-data :cluster-3 (:labels result-3))]
  
  (println "\nk=2 distribution:")
  (println (-> data-with-clusters-2
               (tc/group-by [:cluster-2])
               (tc/aggregate {:count tc/row-count})))
  
  (println "\nk=3 distribution:")
  (println (-> data-with-clusters-3
               (tc/group-by [:cluster-3])
               (tc/aggregate {:count tc/row-count}))))

(println "\n=== COMPREHENSIVE TEST COMPLETED ===")
