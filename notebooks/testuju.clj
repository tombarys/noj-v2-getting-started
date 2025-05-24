(ns testuju
  (:require
   [tech.v3.dataset.tensor]
   [scicloj.kindly.v4.kind :as kind]
   [scicloj.tableplot.v1.plotly :as plotly]
   [java-time.api :as jt]
   [tech.v3.dataset :as ds]
   [scicloj.ml.tribuo]
   [tech.v3.dataset.categorical :as ds-cat]
   [tablecloth.api :as tc]
   [clojure.string :as str]
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

(defn clean-one-hot-metadata [dataset]
  (reduce (fn [ds col-name]
            (let [col (ds/column ds col-name)
                  old-meta (meta col)
                  clean-meta (dissoc old-meta :one-hot-map :categorical-map)]
              (ds/add-or-update-column ds col-name
                                       (with-meta (vec col) clean-meta))))
          dataset
          (ds/column-names dataset)))

(defn sanitize-name-str [s]
  (if-not (and (nil? s) (empty? s))
    (let [hyphens (str/replace s #"_" "-")
          nfd-normalized (Normalizer/normalize hyphens Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "")
          no-spaces (str/replace no-diacritics #" " "-")
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
          (< % 299) "levna"
          (< % 399) "drazsi"
          (< % 499) "draha"
          :else "nejdrazsi")
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

; ## Začátek zpracování 

(def ds
  (as-> raw-ds %
    #_(tc/select-rows % (range 10))
    (tc/rename-columns % :all (fn [col] (if col (keyword (sanitize-name-str (name col))) col)))
    (tc/update-columns % [:titul-knihy :podtitul :vazba :barevnost :edice :tema :cenova-kategorie :tloustka]
                       (fn [column-data]
                         (map sanitize-name-str column-data)))
    (tc/add-column % :tloustka (cat-tloustka %))
    (tc/add-column % :cenova-kategorie (cat-cena %))
    (tc/add-column % :na-trhu (map months-on-market (ds/column raw-ds :Datum_zahajeni_prodeje)))
    (tc// % :mesicni-prodej-ks [:celkovy-prodej-ks :na-trhu])
    (tc/round % :mesicni-prodej-ks :mesicni-prodej-ks)
    (tc/add-column % :prodejnost (cat-prodejnost %))
    (tc/drop-columns % [:ks-papir :ks-e-kniha :ks-audiokniha :trzby-papir :trzby-e-kniha :trzby-audiokniha
                        :dpc-e-kniha :dpc-audiokniha :datum-zahajeni-prodeje
                        :titul-knihy :podtitul :na-trhu :mesicni-trzby :pocet-stran
                        :celkovy-prodej-trzby :celkovy-prodej-ks :edice :mesicni-prodej-ks :dpc-papir])
    #_(tc/convert-types % [:tloustka :barevnost :cesky-autor :tema :cenova-kategorie :vazba] :categorical)))

(kind/table
 (tc/info ds)
 {:width 800
  :height 400
  :title "Dataset info"})


;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
(plotly/layer-point ds {:=x :prodejnost
                        :=y :cenova-kategorie
                        :=text :titul-knihy})

(plotly/splom ds {:=colnames [:tloustka :prodejnost]
                  :=color :tloustka
                  :=text :titul-knihy})


(kind/table
 (map meta (tc/columns ds)))

;; ## Příprava dat pro strojové učení

(def numeric-melvil-data2
  (-> ds
      (ds/categorical->number [:prodejnost] ["underperformer" "normal" "bestseller"] :float-64)
      (ds/categorical->one-hot [:tloustka :barevnost :cesky-autor :tema :vazba :cenova-kategorie])
      (tc/drop-missing)
      clean-one-hot-metadata
      (ds-mod/set-inference-target [:prodejnost])))


(kind/table
 numeric-melvil-data2)

(kind/table
 (tc/info numeric-melvil-data2 :columns))

;; Re-define split and split-fixed based on the new numeric-melvil-data2
(def split
  (first
   (tc/split->seq numeric-melvil-data2 :holdout {:seed 112223})))

#_(def split-fixed
  {:train (ds-mod/set-inference-target (:train split) :prodejnost)
   :test (ds-mod/set-inference-target (:test split) :prodejnost)})

(def rf-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "random-forest"
                         :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                         :properties {:maxDepth "8"
                                      :useRandomSplitPoints "false"
                                      :fractionFeaturesInSplit "0.5"}}]
    :tribuo-trainer-name "random-forest"}))


;; ## Make predictions
(def rf-predictions
  (ml/predict (:test split) rf-model))

(kind/table
 rf-predictions
 {:width 800
  :height 400
  :title "RF Predictions"})

;; ## Evaluate accuracy
(def rf-accuracy
  (loss/classification-accuracy
   (ds/column (:test split) :prodejnost)
   (ds/column rf-predictions :prodejnost)))

(println "RF Random Forest Accuracy:" rf-accuracy)

; ## Dummy model for testing

#_(def dummy-model
  (ml/train (ds-mod/set-inference-target (:train split) :y)
            {:model-type :metamorph.ml/dummy-classifier}))


(def lreg-model (ml/train (:train split)
                          {:model-type :scicloj.ml.tribuo/classification
                           :tribuo-components [{:name "logistic"
                                                :type "org.tribuo.classification.sgd.linear.LinearSGDTrainer"}]
                           :tribuo-trainer-name "logistic"}))


;; Vytvoření vlastní vizualizace rozhodnutí
(defn create-decision-visualization [model test-data]
  (let [predictions (ml/predict test-data model)
        data (map-indexed
              (fn [idx row]
                {:index idx
                 :prediction (get-in predictions [:row idx :species])
                 :actual (get row :y)})
              (ds/rows test-data :as-maps))]
    ;; Použijte Clay nebo jinou viz knihovnu pro zobrazení
    {:data data
     :mark "circle"
     :encoding {:x {:field "index" :type "quantitative"}
                :y {:field "prediction" :type "nominal"}
                :color {:field "actual" :type "nominal"}}}))

#_(kind/vega
 (create-decision-visualization rf-model (:test split))
 {:width 800
  :height 400
  :title "Rozhodnutí modelu pro testovací data"})

; Vytvoření vizualizace rozhodnutí pro dummy model
#_(kind/vega
 (create-decision-visualization dummy-model (:test split))
 {:width 800
  :height 400
  :title "Rozhodnutí Dummy modelu pro testovací data"})