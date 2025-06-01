(ns clustering-test-simple
  (:require
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]))

;; Simple test to verify k-means functions work
(defn euclidean-distance 
  "Vypočítá euklidovskou vzdálenost mezi dvěma body."
  [point1 point2]
  (Math/sqrt 
   (reduce + (map #(* (- %1 %2) (- %1 %2)) point1 point2))))

(defn find-closest-centroid 
  "Najde index nejbližšího centroidu k danému bodu."
  [point centroids]
  (let [distances (map-indexed (fn [idx centroid] 
                                 [idx (euclidean-distance point centroid)]) 
                               centroids)
        closest (apply min-key second distances)]
    (first closest)))

(defn calculate-centroid 
  "Vypočítá centroid (průměr) ze seznamu bodů."
  [points]
  (when (seq points)
    (let [n (count points)
          dims (count (first points))]
      (mapv (fn [dim]
              (/ (reduce + (map #(nth % dim) points)) n))
            (range dims)))))

(defn dataset-to-points 
  "Převede dataset na seznam vektorů (bodů)."
  [dataset]
  (let [feature-cols (ds/column-names dataset)]
    (->> (tc/rows dataset :as-maps)
         (mapv (fn [row]
                 (mapv #(get row % 0) feature-cols))))))

;; Simple test data
(def test-data 
  (tc/dataset {:x [1 2 3 8 9 10]
               :y [1 2 3 8 9 10]}))

(println "Test data:")
(println test-data)

(println "\nConverted to points:")
(def points (dataset-to-points test-data))
(println points)

(println "\nTesting centroid calculation:")
(println (calculate-centroid [[1 1] [2 2] [3 3]]))

(println "\nTesting closest centroid:")
(println (find-closest-centroid [1 1] [[0 0] [5 5]]))
