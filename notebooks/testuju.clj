(ns testuju
  (:require
   [scicloj.kindly.v4.kind :as kind]
   [scicloj.tableplot.v1.plotly :as plotly]
   [java-time.api :as jt]
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]
   [tablecloth.column.api.column :as tcc]
   [clojure.math :as math]
   [scicloj.metamorph.ml.toydata :as toydata]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.core :as mm]
   [tech.v3.dataset.column-filters :as ds-cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [libpython-clj2.python :refer [py. py.-] :as py]))

; ## Inicializace datasetu

(defonce raw-ds (tc/dataset "/Users/tomas/Downloads/melvil2.csv" {:key-fn keyword :separator \tab}))

#_(ns-unmap *ns* 'raw-ds)

(kind/hiccup
 [:div {:style {:max-height "400px"
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

;; ## Začátek zpracování 

(def ds+kategorie
  (as-> raw-ds %
    (tc/add-column % :Tloustka (cat-tloustka raw-ds))
    (tc/add-column % :Cenova_kategorie (cat-cena raw-ds))
    (tc/add-column % :Na_trhu (map months-on-market (ds/column raw-ds :Datum_zahajeni_prodeje)))
    (tc// % :Mesicni_prodej_KS [:Celkovy_prodej_KS :Na_trhu])
    (tc/round % :Mesicni_prodej_KS :Mesicni_prodej_KS)
    (tc/convert-types % :DPC_papir :float64)
    (tc/add-column % :Prodejnost (cat-prodejnost %))))

;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
(plotly/layer-point ds+kategorie {:=x :Prodejnost
                                  :=y :Mesicni_prodej_KS
                                  :=text :Titul_knihy})

(plotly/splom ds+kategorie {:=colnames [:Tloustka :Mesicni_prodej_KS :Prodejnost]
                            :=color :Tloustka
                            :=text :Titul_knihy})
 

(kind/table
 (map meta (tc/columns ds+kategorie)))

(tc/info ds+kategorie)

(tc/head ds+kategorie)

(def ds-kategorie
  (tc/drop-columns ds+kategorie [:KS_papir :KS_e-kniha :KS_audiokniha :Trzby_papir :Trzby_e-kniha :Trzby_audiokniha
                        :DPC_e-kniha :DPC_audiokniha :Datum_zahajeni_prodeje
                        :Titul_knihy :Podtitul :Na_trhu :Mesicni_trzby :Pocet_stran
                        :Celkovy_prodej_trzby :Celkovy_prodej_KS :Edice :Mesicni_prodej_KS]))

(tc/convert-types ds-kategorie :DPC_papir :float64)

(tcc/typeof (:DPC_papir ds-kategorie))

(tc/info ds-kategorie :columns)
