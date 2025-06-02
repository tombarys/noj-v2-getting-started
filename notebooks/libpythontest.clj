(ns libpythontest
  (:require
   [tech.v3.dataset :as ds]
   [libpython-clj2.python :refer [py.-]]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.sklearn-clj :refer :all]))

;; libpython verze

;; example train and test dataset 

(def train-ds
  (-> (ds/->dataset {:x1 [1 1 2 2]
                     :x2 [1 2 2 3]
                     :y  [6 8 9 11]})
      (ds-mod/set-inference-target :y)))

(def test-ds
  (->
   (ds/->dataset {:x1 [3]
                  :x2 [5]
                  :y  [0]})
   (ds-mod/set-inference-target :y)))

;; fit a liner expression model from sklearn, class sklearn.linear_model.LinearRegression

#_(def lin-reg
  (fit train-ds :sklearn.linear-model :linear-regression))

(def log-reg
  (fit train-ds :sklearn.ensemble :random-forest-regressor))

;; Call predict with new data on the estimator
(->> (predict test-ds log-reg)
     (ds/concat (ds/select-columns test-ds [:x1 :x2])))

;; => _unnamed [1 3]:
;;    | :x1 | :x2 |   :y |
;;    |-----|-----|------|
;;    |   3 |   5 | 16.0 |

;; use an other estimator, this time: sklearn.preprocessing.StandardScaler
(def data
  (ds/->dataset {:x1 [0 0 1 1]
                 :x2 [0 0 1 1]}))

data

;; fit the scaler on data                 
(def scaler
  (fit data :sklearn.preprocessing :standard-scaler))

(py.- scaler mean_)
;; => [0.5 0.5]
;;

;; apply the scaling on new data  
(transform (ds/->dataset {:x1 [2] :x2 [2]})  scaler)
;; => :_unnamed [1 2]:
;;    | :x1 | :x2 |
;;    |-----|-----|
;;    | 3.0 | 3.0 |
