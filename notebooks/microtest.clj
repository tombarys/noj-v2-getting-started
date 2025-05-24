;; Porovnání ds/categorical->one-hot vs tc/pivot->wider

(ns microtest
  (:require
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]
   [scicloj.ml.metamorph :as ml]
   [clojure.string :as str]
   [tech.v3.dataset.modelling :as ds-mod]))

(def test-data
  {:x-1 [0 1 0 1 0 1]
   :x-2 [1 0 1 0 1 0]
   :cat [:a :b :c :a :b :c]
   :y [:a :a :b :b :a :b]})

;; Prozkoumejme metadata jednotlivých sloupců
(defn inspect-column-metadata [dataset]
  (doseq [col-name (ds/column-names dataset)]
    (let [col (ds/column dataset col-name)
          metadata (meta col)]
      (println (str col-name ": " metadata)))))

;; Test
(def test-ds (-> test-data
                 ds/->dataset
                 (ds/categorical->number [:y])
                 (ds/categorical->one-hot [:cat])
                 (ds-mod/set-inference-target :y)))

(inspect-column-metadata test-ds)

;; Zkuste taky po rename
(def renamed-ds (ds/rename-columns test-ds {:cat-a :cata :cat-b :catb :cat-c :catc}))
(inspect-column-metadata renamed-ds)

(defn clean-one-hot-metadata [dataset]
  (reduce (fn [ds col-name]
            (let [col (ds/column ds col-name)
                  old-meta (meta col)
                  clean-meta (dissoc old-meta :one-hot-map :categorical-map)]
              (ds/add-or-update-column ds col-name
                                       (with-meta (vec col) clean-meta))))
          dataset
          (ds/column-names dataset)))

(def fixed-solution
  (-> test-data
      ds/->dataset
      (ds/categorical->number [:y])
      (ds/categorical->one-hot [:cat])
      #_(ds/rename-columns {:cat-a :cata :cat-b :catb :cat-c :catc})
      clean-one-hot-metadata
      (ds-mod/set-inference-target :y)))

(def fixed-model
  (ml/train fixed-solution
            {:model-type :metamorph.ml/dummy-classifier}))