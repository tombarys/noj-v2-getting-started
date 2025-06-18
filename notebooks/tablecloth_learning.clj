(ns tablecloth-learning)

(require '[tablecloth.api :as tc]
         '[scicloj.kindly.v4.kind :as kind])

;; ## [Základy](https://scicloj.github.io/tablecloth/#columns-and-rows) 


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

(kind/table
 (tc/dataset {:A [1 (tc/dataset {:E [1 2 3]}) 2]  :B "X"}))

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

(tc/columns ds :as-maps)

;; ### numerické řádky jako double arrays

(-> ds
    (tc/select-columns :type/numerical)
    (tc/head)
    (tc/rows :as-double-arrays))

;; ### numerické sloupce jako double arrays

(-> ds
    (tc/select-columns :type/numerical)
    (tc/head)
    (tc/columns :as-double-arrays))

(clojure.pprint/pprint (take 2 (tc/rows ds :as-maps)))

(-> {:a [1 nil 2]
     :b [3 4 nil]}
    (tc/dataset)
    (tc/rows :as-maps))


;; vypustí chybějící 
(-> {:a [1 nil 2]
     :b [3 4 nil]}
    (tc/dataset)
    (tc/rows :as-maps {:nil-missing? false}))

;; získání jedné hodnoty na řádku 2 ve sloupci wind
(get-in ds ["wind" 2])

;; ## Výpisy

;; ### V md (pro export?)
(tc/print-dataset (tc/group-by DS :V1) {:print-line-policy :markdown})

;; ### Hezky v REPLu
(tc/print-dataset (tc/group-by DS :V1) {:print-line-policy :repl})

;; ### Mini v REPLu
(tc/print-dataset (tc/group-by DS :V1) {:print-line-policy :single})


;; # Grouping

(-> ds
    (tc/group-by "wind"))

(-> DS
    (tc/group-by :V1)
    (tc/as-regular-dataset)
    #_(tc/column-names))

(tc/columns (tc/group-by DS :V2) )

(keys (tc/group-by DS :V1 {:result-type :as-map}))

(vals (tc/group-by DS :V1 {:result-type :as-map}))

(tc/group-by DS :V1 {:result-type :as-indexes})


;; ### Rozložení group
(-> {"a" [1 1 2 2]
     "b" ["a" "b" "c" "d"]}
    (tc/dataset)
    (tc/group-by "a")
    (tc/groups->seq))


(-> {"a" [1 1 2 2]
     "b" ["a" "b" "c" "d"]}
    (tc/dataset)
    (tc/group-by "a")
    (tc/groups->map))


;; ### a teď po více než jeden sloupec groupy

(tc/group-by DS [:V1 :V3] {:result-type :as-seq})

;; ### lze grupovat čistě pomocí indexů

DS

(tc/group-by DS {"skupina-a" [1 2 1 2]
                 "skupina-b" [5 5 5 1]} {:result-type :as-seq})

;; ### grupování podle výsledku operace v jednom řádku nad sloupci
;; When map is used as a group name, ungrouping restore original column names.

(tc/group-by DS (fn [row] (* (:V1 row)
                             (:V3 row))) {:result-type :as-seq})

;; ### rozdělení pomocí predikátu na různé grupy

(tc/group-by DS (comp #(< % 1.0) :V3) {:result-type :as-seq})

