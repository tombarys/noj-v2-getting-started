;; # Clustering - pokusy
(ns clustering-template
  (:require
   [fastmath.ml.clustering :as fm-cluster]
   [scicloj.ml.smile.clustering :as smile-cluster]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [scicloj.kindly.v4.kind :as kind]
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]
   [scicloj.tableplot.v1.plotly :as plotly]
   [fastmath.stats]
   [scicloj.ml.metamorph :as mlmm]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.preprocessing :as prepro]
   [clojure.string :as str]))

;; https://scicloj.github.io/scicloj.ml-tutorials/polyglot_kmeans.html#/

(clojure.repl/doc fm-cluster/kmeans++)

(def smallish-ds
  [1 2 3 4 5 6 7 8 9 10
   11 12 13 14 15
   16 17 18 19 20
   21 22 23 24])


(fm-cluster/kmeans++ smallish-ds
                     {:clusters 4})

(def fitted-ctx-2
  (mm/fit
   smallish-ds
   (mlmm/std-scale :all {})
   {:metamorph/id :k-means}
   (scicloj.ml.smile.clustering/cluster
    :k-means
    [3 300]
    :cluster)))

(-> fitted-ctx-2 :k-means)


(def distortion-2
  (->> [{:metamorph/id :k-means}
        [:scicloj.ml.smile.clustering/cluster
         :k-means [3 300]
         :cluster]]
       (mm/->pipeline)
       (mm/fit-pipe smallish-ds)
       :k-means
       :info))

distortion-2

;; Add cluster assignments to your dataset
(def clustered-ds
  (let [clusters (-> fitted-ctx-2 :metamorph/data :k-means :cluster-ids)
        with-clusters (ds/add-column smallish-ds {:cluster clusters})]
    with-clusters))


(kind/table
 clustered-ds)

