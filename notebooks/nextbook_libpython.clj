(ns nextbook-libpython
  (:require
   [tech.v3.dataset :as ds]
   [scicloj.kindly.v4.kind :as kind]
   [tablecloth.api :as tc]
   [clojure.string :as str]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.sklearn-clj :refer :all])
  (:import [java.text Normalizer Normalizer$Form]))


(defn sanitize-column-name-str [s]
  (if (or (nil? s) (empty? s))
    s
    (let [hyphens (str/replace s #"_" "-")
          trimmed (str/trim hyphens)
          nfd-normalized (Normalizer/normalize trimmed Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "") ; dočasně
          no-spaces (str/replace no-diacritics #" " "-")
          no-brackets (str/replace no-spaces #"\(|\)" "")
          lower-cased (str/lower-case no-brackets)]
      lower-cased)))

(def raw-ds (tc/dataset
             "/Users/tomas/Downloads/wc-orders-report-export-17477349086991.csv"
             {:header? true :separator ","
              :column-allowlist ["Produkt (produkty)" "Zákazník"]
              #_#_:num-rows 100
              :key-fn #(keyword (sanitize-column-name-str %))})) ;; tohle upraví jen názvy sloupců!

(kind/table
 (tc/info raw-ds))

;; ## Pomocné funkce pro sanitizaci sloupců
(defn parse-books [s]
  (->> (str/split s #",\s\d+")
       (map #(str/replace % #"\d*×\s" ""))
       (map #(str/replace % #"," ""))
       (map #(str/replace % #"\(A\+E\)|\[|\]|komplet|a\+e|\s\(P\+E\)|\s\(e\-kniha\)|\s\(P\+E\)|\s\(P\+A\)|\s\(E\+A\)|papír|papir|audio|e\-kniha" ""))
       (map #(str/replace % #"\+" ""))
       (map #(str/trim %))
       (map sanitize-column-name-str)
       (map #(str/replace % #"\-\-.+$" "")) ;; zdvojené názvy
       (map #(str/replace % #"\-+$" "")) ;; pomlčky na konci
       (map #(str/replace % #"3" "k3")) ;; eliminace čísel 3 na začátku dvou knih
       (remove (fn [item] (some (fn [substr] (str/includes? (name item) substr))
                                ["balicek" "poukaz" "zapisnik" "limitovana-edice" "aktualizovane-vydani"])))
       distinct
       (mapv keyword)))

;; ## Funkce pro vytvoření one-hot-encoding sloupců z nakoupených knih
(defn create-one-hot-encoding [raw-ds]
  (let [;; Nejdříve agregujeme všechny nákupy podle zákazníka
        customer-books (-> raw-ds
                           (ds/drop-missing :zakaznik)
                           (tc/group-by [:zakaznik])
                           (tc/aggregate {:all-products #(str/join ", " (ds/column % :produkt-produkty))})
                           (tc/rename-columns {:summary :all-products}))

        ;; Získáme všechny unikátní knihy ze všech řádků
        all-books (->> (ds/column customer-books :all-products)
                       (mapcat parse-books)
                       distinct
                       sort)

        ;; Pro každého zákazníka vytvoříme řádky kde každá kniha je postupně target
        rows-with-books (mapcat
                         (fn [customer-row]
                           (let [customer-name (:zakaznik customer-row)
                                 books-bought (parse-books (:all-products customer-row))]
                             (when (> (count books-bought) 1)  ; Pouze zákazníci s více než jednou knihou
                               ;; Pro každou koupenou knihu vytvoříme řádek
                               (for [target-book books-bought]
                                 (let [feature-books (set (remove #(= % target-book) books-bought))
                                       one-hot-map (reduce (fn [acc book]
                                                             (assoc acc book (if (contains? feature-books book) 1 0)))
                                                           {}
                                                           all-books)]
                                   (merge {:zakaznik customer-name
                                           :next-predicted-buy target-book}
                                          one-hot-map))))))
                         (tc/rows customer-books :as-maps))

        ;; Vytvoříme nový dataset z one-hot dat
        one-hot-ds (tc/dataset rows-with-books)
        _ (println "Zákošů s více než 1 knihou je: " (ds/row-count rows-with-books))]

    ;; Vrátíme dataset s one-hot encoding a nastaveným inference targetem
    (-> one-hot-ds
        (ds/categorical->number [:zakaznik :next-predicted-buy])
        #_(ds/drop-columns [:zakaznik])
        (ds-mod/set-inference-target [:next-predicted-buy]))))


;; ## Splitnutí datasetu pro další zpracování

(def processed-ds (create-one-hot-encoding raw-ds))

processed-ds

(def split
  (-> processed-ds
      (tc/split->seq  :holdout {:seed 42})
      first))

#_(def xgboost-simple-model ;; funguje
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "xgboost-simple"
                         :target-columns [:next-predicted-buy]
                         :type "org.tribuo.classification.xgboost.XGBoostClassificationTrainer"
                         :properties {:numTrees "100"
                                      :maxDepth "6"
                                      :eta "0.1"}}]
    :tribuo-trainer-name "xgboost-simple"}))

#_(loss/classification-accuracy
 (ds/column (:test split)
            :next-predicted-buy)
 (ds/column (ml/predict (:test split) xgboost-simple-model)
            :next-predicted-buy))


(def log-reg
  (fit (:train split) :sklearn.neighbors :k-neighbors-classifier))

(->> (predict (:test split) log-reg))