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

(kind/hiccup
 [:div {:style {:max-height "600px"
                :overflow-y :auto}}
  (kind/table
   raw-ds)])


; ## Pomocné funkce pro kategorizaci

(defn cat-tloustka [ds]
  (map #(cond
          (< % 300) "normal"
          (< % 450) "nadnormal"
          :else "bichle")
       (ds/column ds :Pocet_stran)))

(defn cat-cena [ds]
  (map #(cond
          (< % 299) "pod 299"
          (< % 399) "od 300 do 399"
          (< % 499) "od 400 do 499"
          :else "500 a víc")
       (ds/column ds :DPC_papir)))

(defn cat-prodejnost
  "Vrací sloupec :Mesicni_prodej_KS"
  [ds]
  (map #(if (number? %)
          (cond
            (< % 50) "underperformer"
            (< % 412) "normal"
            :else "bestseller")
          :na)
       (ds/column ds :Mesicni_prodej_KS)))

(def end-date (jt/local-date 2025 3 1))

(defn months-on-market [date]
  (when-let [days (when date (jt/time-between date end-date :days))]
    (long (Math/round (/ days 30.4375)))))

;; ## Pomocné funkce pro zpracování dat


(defn sanitize-name-str [s]
  (if-not (and (nil? s) (empty? s))
    (let [no-hyphens (str/replace s #"-" "_")
          nfd-normalized (Normalizer/normalize no-hyphens Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "")
          ;; Můžete přidat další pravidla, např. odstranění speciálních znaků
          ;; alphanumeric-and-underscore (str/replace no-diacritics #"[^a-zA-Z0-9_]" "")
          lower-cased (str/lower-case no-diacritics)]
      lower-cased)
    s))

;; ## Začátek zpracování 

(def ds-cleaned
  (as-> raw-ds %
    (tc/rename-columns % :all #(if %
                                 (keyword (sanitize-name-str (name %)))
                                 %))
    (tc/add-column % :Tloustka (cat-tloustka raw-ds))
    (tc/add-column % :Cenova_kategorie (cat-cena raw-ds))
    (tc/add-column % :Na_trhu (map months-on-market (ds/column raw-ds :Datum_zahajeni_prodeje)))
    (tc// % :Mesicni_prodej_KS [:Celkovy_prodej_KS :Na_trhu])
    (tc/round % :Mesicni_prodej_KS :Mesicni_prodej_KS)
    (tc/convert-types % :DPC_papir :float64)
    (tc/add-column % :Prodejnost (cat-prodejnost %))
    (tc/convert-types % [:Tloustka :Barevnost :Cesky_autor :Tema :Cenova_kategorie] :categorical)))

;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
(plotly/layer-point ds-cleaned {:=x :Prodejnost
                                  :=y :Mesicni_prodej_KS
                                  :=text :Titul_knihy})

(plotly/splom ds-cleaned {:=colnames [:Tloustka :Mesicni_prodej_KS :Prodejnost]
                            :=color :Tloustka
                            :=text :Titul_knihy})


(kind/table
 (map meta (tc/columns ds-cleaned)))

(tc/info ds-cleaned)

(tc/head ds-cleaned)

(def ds-kategorie
  (tc/drop-columns ds-cleaned [:KS_papir :KS_e-kniha :KS_audiokniha :Trzby_papir :Trzby_e-kniha :Trzby_audiokniha
                                 :DPC_e-kniha :DPC_audiokniha :Datum_zahajeni_prodeje
                                 :Titul_knihy :Podtitul :Na_trhu :Mesicni_trzby :Pocet_stran
                                 :Celkovy_prodej_trzby :Celkovy_prodej_KS :Edice :Mesicni_prodej_KS :DPC_papir]))


#_(tcc/typeof (:DPC_papir ds-kategorie))

#_(tc/info ds-kategorie :columns)

(ds/columns ds-kategorie)


(map
 #(hash-map
   :col-name %
   :values  (distinct (get ds-kategorie %)))
 (ds/column-names ds-kategorie))


;; ## Testování a debug


#_(def relevant-test-data
    (-> ds-kategorie
        (tc/select-columns :all)
        (tc/drop-missing)
        (tc/rename-columns :all #(if %
                                   (keyword (sanitize-name-str (name %)))
                                   %))
        (tc/add-column :vazba1 #(sanitize-name-str %))

        #_(ds/categorical->number [:Prodejnost] ["underperformer" "normal" "bestseller"] :float64)
        #_(ds-mod/set-inference-target :Prodejnost)))

#_(kind/dataset relevant-test-data)

;; ## Fitování modelu

(def cat-maps
  [(ds-cat/fit-categorical-map relevant-melvil-data :Tloustka ["normal" "nadnormal" "bichle"] :float64)
   (ds-cat/fit-categorical-map relevant-melvil-data :Barevnost ["Barevná" "Černobílá" "Dvoubarva" "Černobílá s barevnou vsádkou"] :float64)
   (ds-cat/fit-categorical-map relevant-melvil-data :Tema ["Podnikání"
                                                           "Vzdělávání a výchova"
                                                           "Produktivita"
                                                           "Psychologie"
                                                           "Zdraví"
                                                           "Budoucnost"
                                                           "Hvězdné příběhy"
                                                           "Historie"
                                                           "Ekologie"
                                                           "psychologie"] :float64)
   (ds-cat/fit-categorical-map relevant-melvil-data :Cesky_autor ["ano" "ne"] :float64)
   (ds-cat/fit-categorical-map relevant-melvil-data :Vazba ["Měkká V4" "Měkká V2 s chlopněmi" "Měkká V2" "Pevná bez přebalu V8" "Pevná s přebalem V8"] :float64)
   (ds-cat/fit-categorical-map relevant-melvil-data :Cenova_kategorie ["pod 299" "od 300 do 399" "od 400 do 499" "500 a víc"] :float64)])

#_(kind/map cat-maps)

#_(def numeric-melvil-data
  (reduce (fn [ds cat-map]
            (ds-cat/transform-categorical-map ds cat-map))
          relevant-melvil-data
          cat-maps))

(def ds-transformed
  (let [target-col-original-name :Prodejnost ;; Název cílového sloupce před sanitizací
        ;; Sanitizovaný název cílového sloupce, který budeme používat v set-inference-target
        target-col-sanitized-keyword (keyword (sanitize-column-name-str target-col-original-name))]
    (-> ds-kategorie
        (tc/drop-missing)
        (ds/categorical->number [:Prodejnost] ["underperformer" "normal" "bestseller"] :float64)
        (ds/categorical->one-hot [:Tloustka :Barevnost :Tema :Cesky_autor :Vazba :Cenova_kategorie])
        (tc/rename-columns :all #(if %
                                   (keyword (sanitize-column-name-str (name %)))
                                   %))
        ;; Použijte sanitizovaný název cílového sloupce
        (ds-mod/set-inference-target target-col-sanitized-keyword))))

(ds/column-names ds-transformed)
(-> ds-transformed meta :tech.v3.dataset/target-column)

#_(ds/column-names numeric-melvil-data2)

;; Pak teprve rozděl na trénovací a testovací sadu
(def split
  (first
   (tc/split->seq ds-transformed :holdout {:seed 112723})))


(second split)
;; 3. Použij to pro trénink modelu
(def rf-model
  (ml/train (:train split)
            {:model-type :scicloj.ml.tribuo/classification
             :tribuo-components [{:name "random-forest"
                                  :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                                  :properties {:maxDepth "8"
                                               :useRandomSplitPoints "false"
                                               :fractionFeaturesInSplit "0.5"}}]
             :tribuo-trainer-name "random-forest"}))
;; We start with a dummy model, which simply predicts the majority class.
(def dummy-model (ml/train (:train split)
                           {:model-type :metamorph.ml/dummy-classifier}))

(def dummy-prediction
  (ml/predict (:test split) dummy-model))

(-> dummy-prediction :Prodejnost frequencies)

(loss/classification-accuracy
 (:Prodejnost (ds-cat/reverse-map-categorical-xforms (:test split)))
 (:Prodejnost (ds-cat/reverse-map-categorical-xforms dummy-prediction)))


(def rf-model
  (ml/train (:train split)
            {:model-type :scicloj.ml.tribuo/classification
             :tribuo-components [{:name "random-forest"
                                  :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                                  :properties {:maxDepth "8"
                                               :useRandomSplitPoints "false"
                                               :fractionFeaturesInSplit "0.5"}}]
             :tribuo-trainer-name "random-forest"}))


(def rf-prediction
  (ml/predict (:test split) rf-model))

(-> rf-prediction
    (tc/head)
    (tc/rows))

(loss/classification-accuracy
 (:Prodejnost (ds-cat/reverse-map-categorical-xforms (:test split)))
 (:Prodejnost (ds-cat/reverse-map-categorical-xforms rf-prediction)))

(def pipeline
  (mm/pipeline
   (ds/categorical->one-hot ds-kategorie
                            [:Tloustka :Barevnost :Tema :Cesky_autor :Vazba :Cenova_kategorie])
   #_(ds/categorical->number ds-kategorie :Prodejnost)
   #_(ds-cat/fit-one-hot ds-kategorie :Prodejnost)
   #_(ml/model {:model-type :random-forest
                :target-column :Prodejnost  ;; sloupec, který předpovídáme
                :options {:n-estimators 100
                          :max-depth 10}})))


;; ### Zjištění dostupných modelů
(keys (ns-publics 'scicloj.metamorph.core))