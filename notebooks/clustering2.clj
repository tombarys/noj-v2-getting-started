(require '[fastmath.ml.clustering :as cluster])

(def data1 [1.0 2.0 1.1 2.1
            10.0 10.0
            7.4 7.5 7.9 8.1 8.2
            5.3])

(doc cluster/kmeans++)

(cluster/kmeans++ data1 {:clusters 4})


(cluster/dbscan data1)
