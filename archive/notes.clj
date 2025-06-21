(ns notes)

;(ns-unmap *ns* 'raw-ds)

; (tc/select-rows raw-ds #(re-find #"A.E" (str (:produkt-produkty %))))

(kind/table
 (ds/sample
  (tc/select-rows raw-ds #(str/includes? (str (:produkt-produkty %)) "Ukaž")) 100))


(def parsed-real-ds
  (->>
   (ds/column raw-ds :produkt-produkty)
   (mapcat parse-books)
   distinct
   sort))

parsed-real-ds


processed-ds


(def ds
  (tc/select-rows processed-ds #(= 3000 (:zakaznik %))))

ds

(kind/table
 (ds/sample processed-ds 100))

(kind/hiccup
 [:div {:style {:max-height "600px"
                :overflow-y :auto}}
  (kind/table
   processed-ds)])


(ds/unique-by-column processed-ds :zakaznik)

(defn hybrid-recommend [model input-books & {:keys [top-k cf-weight model-weight]
                                             :or {top-k 5 cf-weight 0.7 model-weight 0.3}}]
  "Kombinuje collaborative filtering s model-based doporučením."
  (let [cf-results (collaborative-recommend input-books :top-k (* 2 top-k))
        model-results (better-model-predict model input-books :top-k (* 2 top-k))

        ;; Kombinujeme skóre
        all-books (into #{} (concat (map :book cf-results)
                                    (map :book model-results)))

        combined-scores (for [book all-books]
                          (let [cf-prob (or (:probability (first (filter #(= (:book %) book) cf-results))) 0.0)
                                model-prob (or (:probability (first (filter #(= (:book %) book) model-results))) 0.0)
                                combined-prob (+ (* cf-weight cf-prob) (* model-weight model-prob))]
                            {:book book :probability combined-prob}))

        ;; Seřadíme a vezmeme top-k
        top-recommendations (->> combined-scores
                                 (sort-by :probability >)
                                 (take top-k))]

    top-recommendations))

;; Test hybrid přístupu
#_(hybrid-recommend xgboost-simple-model [:nexus] :top-k 5)

;; # Make predictions
(def predictions
  (ml/predict (:test split) rf-model))

(def accuracy
  (loss/classification-accuracy
   (ds/column (:test split) :next-predicted-buy)
   (ds/column predictions :next-predicted-buy)))

accuracy
;; ## Náhodný baseline
(def random-baseline-accuracy
  (let [unique-targets (-> (:train split)
                           (ds/column :next-predicted-buy)
                           distinct
                           count)]
    (/ 1.0 unique-targets)))

(println "Náhodný baseline:" random-baseline-accuracy)
(println "Model accuracy:" 0.149)
(println "Zlepšení oproti náhodnému:" (/ 0.149 random-baseline-accuracy))


