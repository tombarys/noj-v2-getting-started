(ns microtest
   (:require
    [tech.v3.dataset :as ds]
    [scicloj.ml.tribuo]
    [tablecloth.api :as tc]
    [scicloj.metamorph.ml :as ml]
    [scicloj.metamorph.ml.loss]
    [tech.v3.dataset.modelling :as ds-mod]))

(def simple-ready-for-train
  (->
   {:x-1 [0 1 0]
    :x-2 [1 0 1]
    :cat [:a :b :c]
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
            {:model-type :scicloj.ml.tribuo/classification
             :tribuo-components [{:name "trainer"
                                  :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                                  :properties {"maxDepth"  "6"
                                               :seed       "12345"}}]
             :tribuo-trainer-name "trainer"}))


dummy-model
(def dummy-predictions
  (ml/predict (:test simple-split-for-train) dummy-model))

(tc/select-columns dummy-predictions [:y :y-predicted])

(def dummy-accuracy
  (scicloj.metamorph.ml.loss/classification-accuracy
   (ds/column (:test simple-split-for-train) :y)
   (ds/column dummy-predictions :y)))