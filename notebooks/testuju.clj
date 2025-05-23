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

(py/initialize!)
(defonce raw-ds (tc/dataset "/Users/tomas/Downloads/melvil2.csv" {:key-fn keyword :separator \tab}))

; (ns-unmap *ns* 'raw-ds)

;; ## Pomocné funkce pro sanitizaci názvů sloupců

(defn sanitize-name-str [s]
  (if-not (and (nil? s) (empty? s))
    (let [hyphens (str/replace s #"_" "-")
          nfd-normalized (Normalizer/normalize hyphens Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "")
          no-spaces (str/trim no-diacritics)
          ;; Můžete přidat další pravidla, např. odstranění speciálních znaků
          ;; alphanumeric-and-underscore (str/replace no-diacritics #"[^a-zA-Z0-9_]" "")
          lower-cased (str/lower-case no-spaces)]
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



(map meta (vals (:train split)))

(meta split)

(:train split-fixed)

#_(keys (ns-publics 'scicloj.metamorph.ml.sklearn))

;; ### Toto funguje

;; Test základní ML bez složitých dependencies
(def simple-test
  (tc/dataset {:x [1 2 3 4 5]
               :y ["a" "b" "a" "b" "a"]}))

(def sample-split 
  (first 
   (tc/split->seq simple-test :holdout {:seed 112723})))

(def simple-model
  (ml/train (:train simple-test)
            {:model-type :metamorph.ml/dummy-classifier
             :target-column :y}))

(println "Basic ML works:" (some? simple-model))

;; Train with sklearn random forest instead
(def sklearn-rf-model
  (ml/train (:train split)
            {:model-type :sklearn.ensemble/random-forest-classifier
             :n-estimators 100
             :max-depth 8
             :random-state 42}))

sklearn-rf-model

(map meta (vals (:train split-fixed)))

;; Make predictions
(def sklearn-predictions
  (ml/predict (:test split-fixed) sklearn-rf-model))

;; Evaluate accuracy
(def sklearn-accuracy
  (loss/classification-accuracy
   (ds/column (:test split-fixed) :prodejnost)
   (ds/column sklearn-predictions :prodejnost)))

(println "Sklearn Random Forest Accuracy:" sklearn-accuracy)

