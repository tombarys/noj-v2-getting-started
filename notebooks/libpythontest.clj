(ns libpythontest
  (:require
   [libpython-clj2.python :as py :refer [py. py.. py.- get-attr call-attr get-item]]
   [scicloj.ml.core :as ml]
   [scicloj.ml.metamorph :as mm]
   [scicloj.sklearn-clj :refer :all]
   [scicloj.ml.dataset :as ds]
   [tech.v3.dataset.tensor :as dst]
   [scicloj.sklearn-clj :as sklearn-clj]
   [scicloj.sklearn-clj.ml]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.metamorph.ml.toydata :as toydata]
   [libpython-clj2.require :refer [require-python]]))


(py/initialize! {:python-executable "/Users/tomas/Dev/noj-v2-getting-started/venv/bin/python" #_"/Users/tomas/Library/Python/3.9/bin/python3"
                 })
(require-python '[numpy :as np])
(require-python '[pandas :as pd])

(def dates (pd/date_range "1/1/2000" :periods 8))
(def table (pd/DataFrame (call-attr np/random :randn 8 4) :index dates :columns ["A" "B" "C" "D"]))
(def row-date (pd/date_range :start "2000-01-01" :end "2000-01-01"))
(get-item (get-attr table :loc) row-date)