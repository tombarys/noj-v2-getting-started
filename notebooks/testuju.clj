(ns testuju
  (:require
   [scicloj.kindly.v4.kind :as kind]
   [scicloj.tableplot.v1.plotly :as plotly]
   [java-time.api :as jt]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tablecloth.api :as tc]
   [clojure.string :as str]
   [scicloj.ml.tribuo :as tribuo]
   [tablecloth.column.api.column :as tcc]
   [clojure.math :as math]
   [scicloj.metamorph.ml.toydata :as toydata]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.classification :as classification]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.core :as mm]
   [tech.v3.dataset.column-filters :as ds-cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [libpython-clj2.python :refer [py. py.-] :as py])
  (:import [java.text Normalizer Normalizer$Form]))



; ## Inicializace datasetu

(defonce raw-ds (tc/dataset "/Users/tomas/Downloads/melvil2.csv" {:key-fn keyword :separator \tab}))

; (ns-unmap *ns* 'raw-ds)

;; ## Pomocné funkce pro sanitizaci názvů sloupců

(defn sanitize-name-str [s]
  (if-not (and (nil? s) (empty? s))
    (let [hyphens (str/replace s #"_" "-")
          nfd-normalized (Normalizer/normalize hyphens Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "")
          ;; Můžete přidat další pravidla, např. odstranění speciálních znaků
          ;; alphanumeric-and-underscore (str/replace no-diacritics #"[^a-zA-Z0-9_]" "")
          lower-cased (str/lower-case no-diacritics)]
      lower-cased)
    s))


; ## Pomocné funkce pro kategorizaci

(kind/hiccup
 [:div {:style {:max-height "600px"
                :overflow-y :auto}}
  (kind/table
   raw-ds)])

(defn cat-tloustka [ds]
  (map #(cond
          (< % 300) "normal"
          (< % 450) "nadnormal"
          :else "bichle")
       (ds/column ds :pocet-stran)))

(defn cat-cena [ds]
  (map #(cond
          (< % 299) "pod 299"
          (< % 399) "od 300 do 399"
          (< % 499) "od 400 do 499"
          :else "500 a vic")
       (ds/column ds :dpc-papir)))

(defn cat-prodejnost
  "Vrací sloupec :mesicni-prodej-ks"
  [ds]
  (map #(if (number? %)
          (cond
            (< % 50) "underperformer"
            (< % 412) "normal"
            :else "bestseller")
          :na)
       (ds/column ds :mesicni-prodej-ks)))

(def end-date (jt/local-date 2025 3 1))

;; ## Pomocné funkce pro zpracování dat


(defn months-on-market [date]
  (when-let [days (when date (jt/time-between date end-date :days))]
    (long (Math/round (/ days 30.4375)))))

;; ## Začátek zpracování 


(ds/head
 (as-> raw-ds %
   (tc/rename-columns % :all (fn [col] (if col (keyword (sanitize-name-str (name col))) col)))
   (tc/update-columns % [:titul-knihy :podtitul :vazba :barevnost :edice :tema :cenova-kategorie :tloustka]
                      (fn [column-data]
                        (map sanitize-name-str column-data)))
   (tc/add-column % :tloustka (cat-tloustka %))))

(def ds-cleaned
  (as-> raw-ds %
    (tc/rename-columns % :all (fn [col] (if col (keyword (sanitize-name-str (name col))) col)))
    (tc/update-columns % [:titul-knihy :podtitul :vazba :barevnost :edice :tema :cenova-kategorie :tloustka]
                    (fn [column-data]
                      (map sanitize-name-str column-data)))
    (tc/add-column % :tloustka (cat-tloustka %))
    (tc/add-column % :cenova-kategorie (cat-cena %))
    (tc/add-column % :na-trhu (map months-on-market (ds/column raw-ds :Datum_zahajeni_prodeje)))
    (tc// % :mesicni-prodej-ks [:celkovy-prodej-ks :na-trhu])
    (tc/round % :mesicni-prodej-ks :mesicni-prodej-ks)
    (tc/convert-types % :dpc-papir :float64)
    (tc/add-column % :prodejnost (cat-prodejnost %))
    (tc/convert-types % [:tloustka :barevnost :cesky-autor :tema :cenova-kategorie] :categorical)))

ds-cleaned


(def ds-kategorie
  (tc/drop-columns ds-cleaned [:ks-papir :ks-e-kniha :ks-audiokniha :trzby-papir :trzby-e-kniha :trzby-audiokniha
                               :dpc-e-kniha :dpc-audiokniha :datum-zahajeni-prodeje
                               :titul-knihy :podtitul :na-trhu :mesicni-trzby :pocet-stran
                               :celkovy-prodej-trzby :celkovy-prodej-ks :edice :mesicni-prodej-ks :dpc-papir]))

ds-kategorie


;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
(plotly/layer-point ds-cleaned {:=x :prodejnost
                                :=y :mesicni-prodej-ks
                                :=text :titul-knihy})

(plotly/splom ds-cleaned {:=colnames [:tloustka :mesicni-prodej-ks :prodejnost]
                          :=color :tloustka
                          :=text :titul-knihy})


(kind/table
 (map meta (tc/columns ds-cleaned)))

(tc/info ds-cleaned)

(tc/head ds-cleaned)

#_(tcc/typeof (:dpc_papir ds-kategorie))

#_(tc/info ds-kategorie :columns)

(ds/column-names ds-kategorie)


#_(map
 #(hash-map
   :col-name %
   :values  (distinct (get ds-kategorie %)))
 (ds/column-names ds-kategorie))

;;;;
(def numeric-melvil-data2
  (-> ds-kategorie
      (tc/drop-missing)
      (ds/categorical->number [:prodejnost] ["underperformer" "normal" "bestseller"] :float64)
      (ds/categorical->one-hot [:tloustka :barevnost :tema :cesky-autor :vazba :cenova-kategorie])
      (ds-mod/set-inference-target :prodejnost)))

(def split
  (first
   (tc/split->seq numeric-melvil-data2 :holdout {:seed 112723})))

(def split-fixed
  {:train (ds-mod/set-inference-target (:train split) :prodejnost)
   :test (ds-mod/set-inference-target (:test split) :prodejnost)})


;; Now train the model with the properly configured dataset
(def rf-model
  (ml/train split
            {:model-type :scicloj.ml.tribuo/classification
             :target-column :prodejnost
             :tribuo-components [{:name "random-forest"
                                  :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                                  :properties {:maxDepth "8"
                                               :useRandomSplitPoints "false"
                                               :fractionFeaturesInSplit "0.5"}}]
             :tribuo-trainer-name "random-forest"}))

;; Make predictions on the test set
(def rf-predictions
  (ml/predict (:test split-fixed) rf-model))

;; Evaluate accuracy
(def accuracy
  (loss/classification-accuracy
   (ds/column (:test split-fixed) :prodejnost)
   (ds/column rf-predictions :prodejnost)))

(println "Random Forest Accuracy:" accuracy)

;; Let's examine everything about the training dataset
(let [train-ds (:train test)]
  (println "=== COMPREHENSIVE DATASET ANALYSIS ===")
  (println "Dataset type:" (type train-ds))
  (println "Dataset shape:" (ds/shape train-ds))
  (println "Column count:" (ds/column-count train-ds))
  (println "Row count:" (ds/row-count train-ds))
  
  (println "\n=== COLUMN NAMES ANALYSIS ===")
  (let [col-names (ds/column-names train-ds)]
    (println "Column names type:" (type col-names))
    (println "Column names count:" (count col-names))
    (println "First 5 columns:" (take 5 col-names))
    (println "Contains :tloustka-normal?" (contains? (set col-names) :tloustka-normal))
    (println "Contains :prodejnost?" (contains? (set col-names) :prodejnost)))
  
  (println "\n=== METADATA ANALYSIS ===")
  (let [metadata (meta train-ds)]
    (println "Metadata keys:" (keys metadata))
    (println "Target columns:" (:target-columns metadata))
    (println "Target column:" (:target-column metadata)))
  
  (println "\n=== COLUMN ACCESS TEST ===")
  (try
    (println ":tloustka-normal sample:" (take 3 (ds/column train-ds :tloustka-normal)))
    (catch Exception e
      (println "Error accessing :tloustka-normal:" (.getMessage e))))
  
  (try
    (println ":prodejnost sample:" (take 3 (ds/column train-ds :prodejnost)))
    (catch Exception e
      (println "Error accessing :prodejnost:" (.getMessage e)))))




(def dummy-model (ml/train (:train split)
                           {:model-type :metamorph.ml/dummy-classifier}))

;; (def dummy-prediction
;;   (ml/predict (:test split) dummy-model))

;; (-> dummy-prediction :Prodejnost frequencies)

;; (loss/classification-accuracy
;;  (:Prodejnost (ds-cat/reverse-map-categorical-xforms (:test split)))
;;  (:Prodejnost (ds-cat/reverse-map-categorical-xforms dummy-prediction)))


;; (def rf-model
;;   (ml/train (:train split)
;;             {:model-type :scicloj.ml.tribuo/classification
;;              :tribuo-components [{:name "random-forest"
;;                                   :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
;;                                   :properties {:maxDepth "8"
;;                                                :useRandomSplitPoints "false"
;;                                                :fractionFeaturesInSplit "0.5"}}]
;;              :tribuo-trainer-name "random-forest"}))


;; (def rf-prediction
;;   (ml/predict (:test split) rf-model))

;; (-> rf-prediction
;;     (tc/head)
;;     (tc/rows))

;; (loss/classification-accuracy
;;  (:Prodejnost (ds-cat/reverse-map-categorical-xforms (:test split)))
;;  (:Prodejnost (ds-cat/reverse-map-categorical-xforms rf-prediction)))

;; (def pipeline
;;   (mm/pipeline
;;    (ds/categorical->one-hot ds-kategorie
;;                             [:Tloustka :Barevnost :Tema :Cesky_autor :Vazba :Cenova_kategorie])
;;    #_(ds/categorical->number ds-kategorie :Prodejnost)
;;    #_(ds-cat/fit-one-hot ds-kategorie :Prodejnost)
;;    #_(ml/model {:model-type :random-forest
;;                 :target-column :Prodejnost  ;; sloupec, který předpovídáme
;;                 :options {:n-estimators 100
;;                           :max-depth 10}})))


;; ### Zjištění dostupných modelů
(keys (ns-publics 'scicloj.metamorph.core))