(ns tablecloth-learning)

(require '[tablecloth.api :as tc])

;; ## Základy
;; https://scicloj.github.io/tablecloth/#columns-and-rows


(def DS (tc/dataset {:V1 (take 9 (cycle [1 2]))
                     :V2 (range 1 10)
                     :V3 (take 9 (cycle [0.5 1.0 1.5]))
                     :V4 (take 9 (cycle ["A" "B" "C"]))}))

(tc/dataset nil {:column-names [:a :b]})

(tc/dataset {:A 33})

(tc/dataset {:A [1 2 3]})


(tc/dataset {:A [3 4 5 "D"] :B ["X" "Y" "Z" "W"]})

(tc/dataset {:A [1 2 3 4 5 6] :B "X" :C :a})

(tc/dataset {:A [[3 4 5] [:a :b]] :B "X"})

(tc/dataset {:A [1 (tc/dataset {:E [1 2 3]} ) 2]  :B "X"})

(tc/dataset [{:a 1 :b 3} {:b 2 :a 99}])

(tc/dataset [[:a 1] [:b 2] [:c 3]])

(-> (map int-array [[1 2] [3 4] [5 6]])
    (into-array)
    (tc/dataset))

(-> (map to-array [[:a :z] ["ee" "ww"] [9 10]])
    (into-array)
    (tc/dataset {:column-names [:a :b :c]
                 :layout :as-columns}))

(defonce ds (tc/dataset "https://vega.github.io/vega-lite/examples/data/seattle-weather.csv"))

ds

;; ### Supr malý dataset

(tc/dataset 999 {:single-value-column-name "my-single-value"
                 :error-column? false})

;; ### Jméno datasetu!

(tc/dataset 999 {:single-value-column-name ""
                 :dataset-name "Single value"
                 :error-column? false})

;; ### Zápis a načtení

(tc/write! DS "output.nippy.gz")

(tc/dataset "output.nippy.gz")

;; ### Počet řádků a další parametry

(tc/row-count ds)

(tc/column-count ds)

(tc/shape ds)

;; ### Další podrobnosti

(tc/dataset-name ds)

(tc/info ds)

(tc/info ds :basic)

(tc/info ds :columns)

;; ### Přejmenování

(->> "seattle-weather"
     (tc/set-dataset-name ds))

;; ## [Sloupce](https://scicloj.github.io/tablecloth/#columns-and-rows)

;; toto je identické

(ds "wind")

(tc/column ds "wind")

