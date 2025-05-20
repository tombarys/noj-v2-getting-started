(ns testuju)
(require '[scicloj.kindly.v4.kind :as kind]
         '[scicloj.metamorph.ml :as ml]
         '[java-time.api :as jt]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.column-filters :as ds-cf]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :refer [py. py.-] :as py])
      

(def feed-string
  (slurp "/Users/tomas/Downloads/melvil2.csv"))

(kind/hiccup
 [:div {:style {:max-height "400px"
                :overflow-y :auto}}
  (kind/code
   feed-string)])

(defn categorize-pages [page-count]
  (cond
    (< page-count 300) "normal"
    (< page-count 450) "nadnormal"
    :else "bichle"))

(defonce raw-ds (tc/dataset "/Users/tomas/Downloads/melvil2.csv" {:separator \tab}))

(def ds (ds/column raw-ds "Pocet_stran" ))

(defn cat-tloustka [ds]
  (map #(cond
          (< % 300) "normal"
          (< % 450) "nadnormal"
          :else "bichle")
       (ds/column ds "Pocet_stran")))

(def end-date (jt/local-date 2025 3 1))


;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
;; filepath: /Users/tomas/Dev/noj-v2-getting-started/notebooks/testuju.clj
(defn months-on-market [date]
  (when (some? date)
    (let [start (if (instance? java.time.LocalDate date)
                  date
                  (try (jt/local-date "yyyy-MM-dd" date)
                       (catch Exception _ nil)))
          days (when start (jt/time-between start end-date :days))]
      (when days (long (Math/round (/ days 30.4375)))))))

(def ds+kategorie
  (-> raw-ds
      (tc/add-column :Tloustka (cat-tloustka raw-ds))
      (tc/add-column :Na_trhu (map months-on-market
                                   (ds/column raw-ds "Datum_zahajeni_prodeje")))))


(tc/info ds+kategorie)

(tc/head ds+kategorie)

#_(tc/info ds :columns)

