(ns nextbook-libpython-categorical-test
  (:import [java.text Normalizer Normalizer$Form]) 
  (:require
    [scicloj.kindly.v4.kind :as kind]))

(require
 '[libpython-clj2.python :as py]
 '[tech.v3.dataset :as ds]
 '[tablecloth.api :as tc]
 '[clojure.string :as str]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.categorical :as ds-cat]
 '[scicloj.metamorph.ml.loss :as loss])

;; Inicializace Python prostředí s explicitní cestou
(let [python-path (or (System/getenv "PYTHON_EXECUTABLE")
                      "/opt/homebrew/Caskroom/miniconda/base/envs/noj-ml/bin/python"
                      "python3")]
  (println "Initializing Python with:" python-path)
  (py/initialize! :python-executable python-path)
  (println "Python initialized successfully!")
  
  ;; Test sklearn dostupnosti
  (try
    (def sklearn-test (py/import-module "sklearn"))
    (println "sklearn version:" (py/get-attr sklearn-test "__version__"))
    (catch Exception e
      (println "ERROR: sklearn not available:" (.getMessage e))
      (throw e))))

(require '[scicloj.sklearn-clj :as sk-clj])
(println "✓ sklearn-clj loaded in test file!")

;; Test základního použití
(println "Testing basic sklearn-clj functionality...")

(System/exit 0)
