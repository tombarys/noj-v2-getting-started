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

 ;; ## Začátek zpracování 

; Vybere řádky, kde je nil v libovolném sloupci
 (tc/select-rows raw-ds (fn [row] (some nil? (vals row))))

 (def ds
   (as-> raw-ds %
     (tc/select-rows % (range 10))
     (tc/rename-columns % :all (fn [col] (if col (keyword (sanitize-name-str (name col))) col)))
     (tc/update-columns % [:titul-knihy :podtitul :vazba :barevnost :edice :tema :cenova-kategorie :tloustka]
                        (fn [column-data]
                          (map sanitize-name-str column-data)))
     #_(tc/add-column % :tloustka (cat-tloustka %))
     #_(tc/add-column % :cenova-kategorie (cat-cena %))
     (tc/add-column % :na-trhu (map months-on-market (ds/column raw-ds :Datum_zahajeni_prodeje)))
     (tc// % :mesicni-prodej-ks [:celkovy-prodej-ks :na-trhu])
     (tc/round % :mesicni-prodej-ks :mesicni-prodej-ks)
     (tc/add-column % :prodejnost (cat-prodejnost %))
     (tc/drop-columns % [:ks-papir :ks-e-kniha :ks-audiokniha :trzby-papir :trzby-e-kniha :trzby-audiokniha
                         :dpc-e-kniha :dpc-audiokniha :datum-zahajeni-prodeje
                         :titul-knihy :podtitul :na-trhu :mesicni-trzby :pocet-stran
                         :celkovy-prodej-trzby :celkovy-prodej-ks :edice :mesicni-prodej-ks :dpc-papir
                        ; mažeme kvůli testu jen:
                         :cesky-autor :tema :vazba])
     #_(tc/convert-types % [:tloustka :barevnost :cesky-autor :tema :cenova-kategorie :vazba] :categorical)))

 (kind/table
  (tc/info ds))

 (ds/sample ds)


 ;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
 (plotly/layer-point ds {:=x :prodejnost
                         :=y :mesicni-prodej-ks
                         :=text :titul-knihy})

 (plotly/splom ds {:=colnames [:tloustka :prodejnost]
                   :=color :tloustka
                   :=text :titul-knihy})


 (kind/table
  (map meta (tc/columns ds)))

 (tc/info ds)

 (tc/head ds)

 #_(tcc/typeof (:dpc_papir ds-kategorie))

 (tc/info ds :columns)

 (ds/column-names ds)




 #_(map
    #(hash-map
      :col-name %
      :values  (distinct (get ds %)))
    (ds/column-names ds))

 (def numeric-melvil-data2
   (-> ds

       ;; 1. Convert target column to numerical.
       ;;    Values not in the list ["underperformer" "normal" "bestseller"] (e.g., :na) will become nil.
       (ds/categorical->number [:prodejnost] ["underperformer" "normal" "bestseller"] :float-64)

       ;; 2. One-hot encode other categorical features.
       (ds/categorical->one-hot [#_:tloustka :barevnost #_#_#_:cesky-autor :tema :vazba #_:cenova-kategorie])

       ;; 4. Drop rows that contain any nil values.
       ;;    This is critical to remove rows where :prodejnost might have become nil,
       ;;    or any nils introduced by other steps.
       (tc/drop-missing)

       ;; 5. Set the inference target on the fully cleaned dataset.
       ;;    Ensure the keyword matches the (potentially sanitized) name of your target column.
       ;;    Since "prodejnost" has no hyphens, its name shouldn't change in step 3.

       (ds-mod/set-inference-target [:prodejnost])))


 numeric-melvil-data2

 (kind/table
  (tc/info numeric-melvil-data2 :columns))

 ;; Re-define split and split-fixed based on the new numeric-melvil-data2
 (def split
   (first
    (tc/split->seq numeric-melvil-data2 :holdout {:seed 112223})))

 (:train split)
 (:test split)

 ;; It's good practice to ensure the target is also set on the split datasets,
 ;; though it should be inherited if set before splitting.
 (def split-fixed
   {:train (ds-mod/set-inference-target (:train split) :prodejnost)
    :test (ds-mod/set-inference-target (:test split) :prodejnost)})

 (map #(vals %) (:train split-fixed))

 (def sklearn-rf-model
   (ml/train (:train split)
             {:model-type :sklearn.ensemble/random-forest-classifier
              :target-columns :prodejnost
              :n-estimators 100
              :max-depth 8
              :random-state 42}))

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
   (ml/train (ds-mod/set-inference-target (:train sample-split) :y)
             {:model-type :scicloj.ml.tribuo/classification}))


 (println "Basic ML works:" (some? simple-model))

 ;; Train with sklearn random forest instead
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



; test

 (def simple-ready-for-train
   (->
    {:x-1 [0 1 0]
     :x-2 [1 0 1]
     :cat  [:a :b :c]
     :y [:a :a :b]}

    (ds/->dataset)
    (ds/categorical->number [:y])
    (ds/categorical->one-hot [:cat])
    (ds-mod/set-inference-target [:y])))

 (def simple-split-for-train
   (first
    (tc/split->seq simple-ready-for-train :holdout {:seed 112723})))

 (def dummy-model
   (ml/train (ds-mod/set-inference-target (:train simple-split-for-train) :y)
             {:model-type :metamorph.ml/dummy-classifier}))


 (def lreg-model (ml/train (:train simple-split-for-train)
                           {:model-type :scicloj.ml.tribuo/classification
                            :tribuo-components [{:name "logistic"
                                                 :type "org.tribuo.classification.sgd.linear.LinearSGDTrainer"}]
                            :tribuo-trainer-name "logistic"}))