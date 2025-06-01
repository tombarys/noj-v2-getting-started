(ns libpythontest2)

(require '[scicloj.metamorph.ml :as ml]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[scicloj.kindly.v4.kind :as kind]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.column-filters :as ds-cf]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :refer [py. py.-] :as py])

;; Initialize Python with the correct path (only once!)
(py/initialize! :python-executable "/opt/homebrew/bin/python3")

;; Import sklearn modules after initialization
(def sklearn-ensemble (py/import-module "sklearn.ensemble"))
(def sklearn-model-selection (py/import-module "sklearn.model_selection"))

(def iris
  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))

(kind/table iris)

;; Use sklearn directly via libpython-clj2
(def RandomForestClassifier (py/get-attr sklearn-ensemble "RandomForestClassifier"))
(def train_test_split (py/get-attr sklearn-model-selection "train_test_split"))