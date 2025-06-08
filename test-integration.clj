(println "Starting Python integration test...")

;; Test 1: Basic requires
(println "Step 1: Loading basic dependencies...")
(require '[libpython-clj2.python :as py])
(require '[tech.v3.dataset :as ds])
(require '[tablecloth.api :as tc])
(require '[clojure.string :as str])
(require '[tech.v3.dataset.modelling :as ds-mod])
(require '[tech.v3.dataset.categorical :as ds-cat])
(require '[scicloj.metamorph.ml.loss :as loss])
(println "✓ Basic dependencies loaded")

;; Test 2: Python initialization
(println "Step 2: Initializing Python...")
(let [python-path (or (System/getenv "PYTHON_EXECUTABLE")
                      "/opt/homebrew/Caskroom/miniconda/base/envs/noj-ml/bin/python"
                      "python3")]
  (println "Using Python path:" python-path)
  (py/initialize! :python-executable python-path)
  (println "✓ Python initialized successfully!"))

;; Test 3: sklearn import
(println "Step 3: Testing sklearn import...")
(try
  (def sklearn-test (py/import-module "sklearn"))
  (println "✓ sklearn imported, version:" (py/get-attr sklearn-test "__version__"))
  (catch Exception e
    (println "✗ ERROR importing sklearn:" (.getMessage e))
    (throw e)))

;; Test 4: sklearn-clj
(println "Step 4: Loading sklearn-clj...")
(try
  (require '[scicloj.sklearn-clj :as sk-clj])
  (println "✓ sklearn-clj loaded successfully!")
  (catch Exception e
    (println "✗ ERROR loading sklearn-clj:" (.getMessage e))
    (throw e)))

(println "🎉 ALL TESTS PASSED!")
(System/exit 0)
