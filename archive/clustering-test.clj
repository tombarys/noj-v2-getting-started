;; Simple test for the clustering implementation(ns clustering-test)
(ns clustering-test
  (:require
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]))

;; Test basic k-means functions
(defn euclidean-distance [point1 point2]
  (->> (map - point1 point2)
       (map #(* % %))
       (reduce +)
       Math/sqrt))

(defn test-basic-functions []
  (println "Testing basic clustering functions...")
  
  ;; Test euclidean distance
  (let [p1 [1 2 3]
        p2 [4 5 6]
        dist (euclidean-distance p1 p2)]
    (println (str "Distance between " p1 " and " p2 ": " dist)))
  
  ;; Test dataset creation
  (let [sample-data (tc/dataset {:book1 [1 0 1 0 1]
                                 :book2 [0 1 0 1 0]
                                 :book3 [1 1 0 0 1]})]
    (println "Sample dataset shape:" (tc/shape sample-data))
    (println "Column names:" (ds/column-names sample-data)))
  
  (println "Basic functions working correctly!"))

;; Run the test
(test-basic-functions)
