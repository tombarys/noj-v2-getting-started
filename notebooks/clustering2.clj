(ns kmeans-example
  (:require [tech.v3.dataset :as ds]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [scicloj.sklearn-clj.ml])) ; registruje sklearn modely

(defn simple-kmeans
  "Jednoduchá K-Means clustering funkce.
   
   Parametry:
   - data: dataset (tech.ml.dataset)
   - feature-columns: vektor názvů sloupců pro clustering [:sloupec1 :sloupec2]
   - k: počet clusterů (default 3)
   - random-state: pro reprodukovatelnost (default 42)
   
   Vrací: dataset s přidaným sloupcem :cluster obsahujícím přiřazené clustery"
  ([data feature-columns]
   (simple-kmeans data feature-columns 3))
  ([data feature-columns k]
   (simple-kmeans data feature-columns k 42))
  ([data feature-columns k random-state]
   (let [;; Pipeline pro K-Means
         pipeline-fn (ml/pipeline
                      ;; Vyber pouze potřebné sloupce
                      (mm/select-columns feature-columns)
                      ;; Standardní škálování dat (doporučené pro K-Means)
                      (mm/model {:model-type :sklearn.preprocessing/standard-scaler})
                      ;; K-Means model
                      {:metamorph/id :kmeans}
                      (mm/model {:model-type :sklearn.cluster/kmeans
                                 :n-clusters k
                                 :random-state random-state
                                 :n-init "auto"}))

         ;; Aplikuj pipeline na data
         ctx (ml/fit-pipe data pipeline-fn)

         ;; Predikce clusterů
         result (ml/transform-pipe data pipeline-fn ctx)]

     ;; Přidej cluster labels do původního datasetu
     #_(ds/add-column data :cluster (:sklearn.cluster/kmeans result)))))

;; Alternativní verze jen s predikcí (bez škálování)
(defn basic-kmeans
  "Základní K-Means bez preprocessing.
   
   Parametry stejné jako simple-kmeans"
  ([data feature-columns]
   (basic-kmeans data feature-columns 3))
  ([data feature-columns k]
   (basic-kmeans data feature-columns k 42))
  ([data feature-columns k random-state]
   (let [pipeline-fn (ml/pipeline
                      (mm/select-columns feature-columns)
                      {:metamorph/id :kmeans}
                      (mm/model {:model-type :sklearn.cluster/kmeans
                                 :n-clusters k
                                 :random-state random-state
                                 :n-init "auto"}))

         ctx (ml/fit-pipe data pipeline-fn)
         result (ml/transform-pipe data pipeline-fn ctx)]

     (ds/add-column data :cluster (:sklearn.cluster/kmeans result)))))

;; Užitečná helper funkce pro analýzu výsledků
(defn analyze-clusters
  "Analyzuje výsledky clusteringu.
   Vrací základní statistiky o clusterech."
  [clustered-data feature-columns]
  (-> clustered-data
      (ds/group-by [:cluster])
      (ds/aggregate {:count ds/row-count
                     :mean-features (fn [ds]
                                      (select-keys
                                       (ds/descriptive-stats ds feature-columns)
                                       [:mean]))})))

;; Příklad použití:
(comment
  ;; Vytvořit testovací data
  (def test-data
    (ds/->dataset {:x [1 1 2 2 10 10 11 11]
                   :y [1 2 1 2 10 11 10 11]
                   :jmeno ["a" "b" "c" "d" "e" "f" "g" "h"]}))

  ;; Spustit K-Means s 2 clustery
  (def result (simple-kmeans test-data [:x :y] 2))

  ;; Zobrazit výsledek
  (println result)

  ;; Analyzovat clustery
  (def stats (analyze-clusters result [:x :y]))
  (println stats)

  ;; Jen základní K-Means bez škálování
  (def basic-result (basic-kmeans test-data [:x :y] 2))
  (println basic-result))

;; Pro vizualizaci (vyžaduje dodatečné knihovny)
(defn plot-clusters
  "Jednoduchá vizualizace 2D clusterů (vyžaduje plotting knihovnu)"
  [clustered-data x-col y-col]
  ;; Tuto funkci byste implementovali s vaší oblíbenou plotting knihovnou
  ;; např. Hanami, Vega-Lite, nebo oz
  (println "Pro vizualizaci použijte plotting knihovnu jako Hanami:")
  (println (str "Data: " (ds/column-names clustered-data))))