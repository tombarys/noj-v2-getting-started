(ns microtest
  (:require
   [tech.v3.dataset :as ds]
   [scicloj.ml.tribuo]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset.modelling :as ds-mod]))

(def simple-ready-for-train
  (->
   {:x-1 [0 1 0]
    :x-2 [1 0 1]
    :cat  [:a :b :c]
    :y [:a :a :b]}

   (ds/->dataset)
   (ds/categorical->number [:y])
   (ds/categorical->one-hot [:cat])
   (ds-mod/set-inference-target [:y])))

(def simple-split-for-train
  (first
   (tc/split->seq simple-ready-for-train :holdout {:seed 112723})))

(def dummy-model
  (ml/train (ds-mod/set-inference-target (:train simple-split-for-train) :y)
            {:model-type :metamorph.ml/dummy-classifier}))

;; TERMINAL:
;; Execution error at tech.v3.dataset.impl.dataset.Dataset/fn (dataset.clj:299).
;; Failed to find column :cat-c