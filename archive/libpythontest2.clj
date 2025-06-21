(ns libpythontest2)

;; sklearn-clj verze

(require
 '[scicloj.metamorph.ml :as ml]
 '[scicloj.metamorph.core :as mm]
 '[scicloj.metamorph.ml.toydata :as toydata]
 '[tablecloth.api :as tc]
 '[tech.v3.dataset :as ds]
 '[scicloj.kindly.v4.kind :as kind]
 '[tech.v3.dataset.column-filters :as ds-cf]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[libpython-clj2.python :refer [py. py.-] :as py]
 '[scicloj.sklearn-clj]
 '[scicloj.sklearn-clj.ml])

(py/initialize! :python-executable "/Users/tomas/Dev/noj-v2-getting-started/.venv/bin/python")

(def iris
  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))

(kind/table
 iris)

(def pipe-fn
  (mm/pipeline
   (fn [ctx]
     (let [model-fn (ml/model {:model-type :sklearn.classification/logistic-regression
                               :max-iter 1000
                               :verbose true})
           model-ctx (model-fn ctx)]
       #_(update model-ctx :metamorph/data
                 tc/rename-columns {0 :predicted-species})
       model-ctx))))

(def pipe-fn-2
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model {:model-type :sklearn.classification/k-neighbors-classifier
              :weights :distance})))

(def trained-ctx 
  (mm/fit-pipe iris pipe-fn))

(def simulated-new-data
  (tc/head (tc/shuffle iris) 10))

(def prediction
  (:metamorph/data
   (mm/transform-pipe
    simulated-new-data
    pipe-fn
    trained-ctx)))

(def prediction-ctx
  (mm/transform-pipe
   simulated-new-data
   pipe-fn
   trained-ctx))

(kind/table
 (:metamorph/data prediction-ctx))

(->> @ml/model-definitions* keys sort)