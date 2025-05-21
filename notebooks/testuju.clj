(ns testuju
  (:require
   [scicloj.kindly.v4.kind :as kind]
   [scicloj.tableplot.v1.plotly :as plotly]
   [java-time.api :as jt]
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]
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
  (let [ds+ (-> raw-ds
                (tc/add-column :Tloustka (cat-tloustka raw-ds))
                (tc/add-column :Cenova_kategorie (cat-cena raw-ds))
                (tc/add-column :Na_trhu (map months-on-market (ds/column raw-ds :Datum_zahajeni_prodeje))))
        ds+ (-> ds+
                (tc/add-column :Mesicni_prodej_KS (map (fn [prodej na-trhu] (math/round (/ prodej na-trhu)))
                                                       (ds/column ds+ :Celkovy_prodej_KS)
                                                       (ds/column ds+ :Na_trhu)))
                (tc/add-column :Prodejnost cat-prodejnost ds+))]
    (tc/drop-columns ds+ [:KS_papir :KS_e-kniha :KS_audiokniha :Trzby_papir :Trzby_e-kniha :Trzby_audiokniha
                          :DPC_e-kniha :DPC_audiokniha :Datum_zahajeni_prodeje
                          #_:Titul_knihy :Podtitul :Na_trhu :Mesicni_trzby :Pocet_stran
                          :Celkovy_prodej_trzby #_:Celkovy_prodej_KS :Edice #_:Mesicni_prodej_KS])))


;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
(plotly/layer-point ds+kategorie {:=x :Prodejnost
                                  :=y :Mesicni_prodej_KS
                                  :=text :Titul_knihy})

(tc/info ds+kategorie)

(tc/head ds+kategorie)

#_(tc/info ds :columns)

