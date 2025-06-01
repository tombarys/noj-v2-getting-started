(ns libpythontest2)

(require '[scicloj.metamorph.ml :as ml]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[scicloj.kindly.v4.kind :as kind]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.column-filters :as ds-cf]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :as py]
         '[libpython-clj2.require :refer [require-python]])

(py/initialize! :python-executable "/Users/tomas/Dev/noj-v2-getting-started/.venv/bin/python")

(py/run-simple-string "import sys") 

(require-python '[pandas :as pd])
                '[numpy :as np]
                '[sklearn.ensemble :as ensemble]
                '[sklearn.model_selection :as model-selection])

;; Import sklearn modules after initialization
(def sklearn-ensemble (py/import-module "sklearn.ensemble"))
(def sklearn-model-selection (py/import-module "sklearn.model_selection"))

;; Test pandas import
(def pandas (libpython-clj2.python/import-module "pandas"))
(def numpy (py/import-module "numpy"))

;; Verify it works
(py/py. pandas "DataFrame" {"a" [1 2 3] "b" [4 5 6]})

(def iris
  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))

(kind/table iris)

;; Use sklearn directly via libpython-clj2
(def RandomForestClassifier (py/get-attr sklearn-ensemble "RandomForestClassifier"))
(def train_test_split (py/get-attr sklearn-model-selection "train_test_split"))