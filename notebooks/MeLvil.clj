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

;; (ns-unmap *ns* 'raw-ds)

;; ## Pomocné funkce pro sanitizaci sloupců

(defn add-missing-columns [ds all-columns]
  (let [missing-cols (clojure.set/difference (set all-columns) (set (ds/column-names ds)))]
    (reduce (fn [d col]
              (ds/add-or-update-column d col (repeat (count (ds/rows d)) 0)))
            ds
            missing-cols)))

(defn clean-one-hot-metadata [dataset]
  (reduce
   (fn [ds col-name]
     (ds/add-or-update-column ds col-name (vec (ds/column ds col-name))))
   dataset
   (ds/column-names dataset)))

(kind/table 
 (tc/info raw-ds))
(kind/table 
 (tc/info (clean-one-hot-metadata raw-ds)))


(defn sanitize-name-str [s]
  (if-not (and (nil? s) (empty? s))
    (let [hyphens (str/replace s #"_" "-")
          nfd-normalized (Normalizer/normalize hyphens Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "") ; dočasně
          no-spaces (str/replace nfd-normalized #" " "-")
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

(defn process-ds [raw-ds]
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
    (tc/add-column % :prodejnost (cat-prodejnost %))
    (tc/drop-columns % [:ks-papir :ks-e-kniha :ks-audiokniha :trzby-papir :trzby-e-kniha :trzby-audiokniha
                        :dpc-e-kniha :dpc-audiokniha :datum-zahajeni-prodeje
                        :titul-knihy :podtitul :na-trhu :mesicni-trzby :pocet-stran
                        :celkovy-prodej-trzby :celkovy-prodej-ks :edice :mesicni-prodej-ks :dpc-papir])
    #_(tc/convert-types % [:tloustka :barevnost :cesky-autor :tema :cenova-kategorie :vazba] :categorical)))

(def ds (process-ds raw-ds))

(kind/table
 (tc/info ds)
 {:width 800
  :height 400
  :title "Dataset info"})


;; ## Příprava dat pro strojové učení

(defn transform-ds [ds]
  (-> ds
      (ds/categorical->number [:prodejnost] ["underperformer" "normal" "bestseller"] :float-64)
      (ds/categorical->one-hot [:tloustka :barevnost :cesky-autor :tema :vazba :cenova-kategorie])
      (tc/drop-missing)
      clean-one-hot-metadata
      (tc/update-columns [:tloustka :barevnost :cesky-autor :tema :vazba :cenova-kategorie] vec)
      (ds-mod/set-inference-target [:prodejnost])))


(def ds-transformed
  (transform-ds ds))

(def all-columns
  (->> (ds/column-names ds-transformed)
       (map keyword)
       (filter #(not= % :prodejnost))
       (into [])))

(kind/table
 ds-transformed)


;; ## Re-define split 
(def split
  (first
   (tc/split->seq ds-transformed :holdout {:ratio 0.85 :seed 112223})))

;;; ## Trénink modelu

(def rf-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "random-forest"
                         :target-columns [:prodejnost]
                         :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                         :properties {:maxDepth "10"
                                      :useRandomSplitPoints "true"
                                      :fractionFeaturesInSplit "0.3"}}]
    :tribuo-trainer-name "random-forest"}))


;; # Make predictions
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

; RF Random Forest Accuracy:
rf-accuracy


; ## Predicting a book success

(defn process-ds-for-prediction [raw-ds]
  (as-> raw-ds %
    (tc/add-column % :tloustka (cat-tloustka %))
    (tc/add-column % :cenova-kategorie (cat-cena %))
    (tc/drop-columns % [:ks-papir :ks-e-kniha :ks-audiokniha :trzby-papir :trzby-e-kniha :trzby-audiokniha
                        :dpc-e-kniha :dpc-audiokniha :datum-zahajeni-prodeje
                        :titul-knihy :podtitul :na-trhu :mesicni-trzby :pocet-stran
                        :celkovy-prodej-trzby :celkovy-prodej-ks :edice :mesicni-prodej-ks :dpc-papir])
    #_(tc/convert-types % [:tloustka :barevnost :cesky-autor :tema :cenova-kategorie :vazba] :categorical)))

(ml/predict
 (-> (tc/dataset
      [{:barevnost "cernobila"
        :cesky-autor "ne"
        :pocet-stran 450
        :dpc-papir 650
        :tema "produktivita"
        :vazba "vazba-pevna"
        :prodejnost nil}])
     (process-ds-for-prediction)
     (ds/categorical->one-hot [:tloustka :barevnost :cesky-autor :tema :vazba :cenova-kategorie])
     (add-missing-columns all-columns)
     clean-one-hot-metadata
     (ds-mod/set-inference-target [:prodejnost]))
 rf-model)


; ## Dummy model for testing

#_(def dummy-model
    (ml/train (ds-mod/set-inference-target (:train split) :y)
              {:model-type :metamorph.ml/dummy-classifier}))


(def lreg-model (ml/train (:train split)
                          {:model-type :scicloj.ml.tribuo/classification
                           :tribuo-components [{:name "logistic"
                                                :type "org.tribuo.classification.sgd.linear.LinearSGDTrainer"}]
                           :tribuo-trainer-name "logistic"}))


