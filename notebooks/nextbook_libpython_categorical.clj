(ns nextbook-libpython-categorical
  (:import [java.text Normalizer Normalizer$Form])
  (:require
   [scicloj.kindly.v4.kind :as kind]
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]))

(require
 '[libpython-clj2.python :as py]
 '[tech.v3.dataset :as ds]
 '[scicloj.kindly.v4.kind :as kind]
 '[tablecloth.api :as tc]
 '[clojure.string :as str]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.categorical :as ds-cat]
 '[scicloj.metamorph.ml.loss :as loss])

;; Inicializace Python prostředí s explicitní cestou
(let [python-path (or (System/getenv "PYTHON_EXECUTABLE")
                      "/opt/homebrew/Caskroom/miniconda/base/envs/noj-ml/bin/python"
                      "python3")]
  (println "Initializing Python with:" python-path)
  (py/initialize! :python-executable python-path)
  (println "Python initialized successfully!")

  ;; Test sklearn dostupnosti
  (try
    (def sklearn-test (py/import-module "sklearn"))
    (println "sklearn version:" (py/get-attr sklearn-test "__version__"))
    (catch Exception e
      (println "ERROR: sklearn not available:" (.getMessage e))
      (throw e))))

(require
 '[scicloj.sklearn-clj :as sk-clj])

;; # Pomocné funkce
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

(def raw-ds
  (tc/dataset
   "/Users/tomas/Downloads/wc-orders-report-export-1749189240838.csv"
   {:header? true :separator ","
    :column-allowlist ["Produkt (produkty)" "Zákazník"]
    #_#_:num-rows 2000
    :key-fn #(keyword (sanitize-column-name-str %))})) ;; tohle upraví jen názvy sloupců!

(kind/table
 (tc/info raw-ds))


(defn parse-books [s]
  (->> (str/split s #",\s\d+")
       (map #(str/replace % #"\d*×\s" ""))
       (map #(str/replace % #"," ""))
       (map #(str/replace % #"\(A\+E\)|\[|\]|komplet|a\+e|\s\(P\+E\+A\)|\s\(e\-kniha\)|\s\(P\+E\)|\s\(P\+A\)|\s\(E\+A\)|papír|papir|audio|e\-kniha|taška" ""))
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


;; ## Pomocné funkce pro sanitizaci sloupců


(defn create-one-hot-encoding-simple [raw-ds]
  (let [;; Nejdříve agregujeme všechny nákupy podle zákazníka
        customer+orders (-> raw-ds
                            (ds/drop-missing :zakaznik)
                            (tc/group-by [:zakaznik])
                            (tc/aggregate {:all-products #(str/join ", " (ds/column % :produkt-produkty))})
                            (tc/rename-columns {:summary :all-products}))

        ;; Získáme všechny unikátní knihy ze všech řádků
        all-titles (->> (ds/column customer+orders :all-products)
                        (mapcat parse-books)
                        distinct
                        sort)

        ;; Pro každého zákazníka vytvoříme řádky kde každá kniha je postupně target
        customers->rows (map
                         (fn [customer-row]
                           (let [customer-name (:zakaznik customer-row)
                                 books-bought-set (set (parse-books (:all-products customer-row)))
                                 one-hot-map (reduce (fn [acc book]
                                                       (assoc acc book (if (contains? books-bought-set book) 1 0)))
                                                     {}
                                                     all-titles)]
                             (merge {:zakaznik customer-name}
                                    one-hot-map)))
                         (tc/rows customer+orders :as-maps))

        ;; Vytvoříme nový dataset z one-hot dat
        one-hot-ds (tc/dataset customers->rows)]

    ;; Vrátíme dataset s one-hot encoding a nastaveným inference targetem
    (-> one-hot-ds
        #_(ds/drop-columns [:zakaznik]))))

(def ds-simple (create-one-hot-encoding-simple raw-ds))

(def columns (-> ds-simple
                 (ds/drop-columns [:zakaznik])
                 ds/column-names))


(def counts
  (-> (map (fn [col]
          [col (-> (tc/sum ds-simple col)
                   (ds/column "summary")
                   first)])
        columns)
      (tc/dataset)
      (tc/rename-columns [:book :customers-bought])
      (tc/order-by :customers-bought :desc)))

counts

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
        (ds/drop-columns [:zakaznik])
        (ds-mod/set-inference-target [:next-predicted-buy]))))



;; ## Nejdříve korelační matička

;; Převedeme na one-hot a pro categorical mapping použijeme ds/categorical->number který vytvoří metadata

 ;; Top X nejčastějších knih - korelační matice
 (def corr-matrix
   (let [numeric-features (ds/drop-columns ds-simple [:zakaznik])

         ;; Najdeme top nejčastějších knih (nejvíc jedniček)
         top-books (->> (ds/column-names numeric-features)
                        (map (fn [col] [col (reduce + (ds/column numeric-features col))]))
                        (sort-by second >)
                        (take 75)
                        (map first))

         ;; Filtrujeme dataset na tyto knihy
         top-ds (ds/select-columns numeric-features top-books)

         ;; Převedeme na pandas DataFrame - NEJJEDNODUŠŠÍ ZPŮSOB
         pd (py/import-module "pandas")

         ;; Vytvoříme DataFrame ze sloupců
         df (py/call-attr pd "DataFrame"
                          (into {} (map (fn [col]
                                          [(name col) (vec (ds/column top-ds col))])
                                        top-books)))

         ;; Korelační matice
         corr-matrix (py/call-attr df "corr")
         corr-values (py/->jvm (py/get-attr corr-matrix "values"))
         col-names (py/->jvm (py/get-attr df "columns"))]
     {:corr-values corr-values 
      :col-names col-names}))

(defn book-sum-correlations [correlation-data]
  (let [col-names (:col-names correlation-data)
        corr-values (:corr-values correlation-data)
        num-books (count col-names)]
    (if (< num-books 1) ; Změna podmínky, stačí jedna kniha pro sumu
      (tc/dataset {:book [] :sum-correlation []})
      (let [sum-correlations-data
            (map-indexed
             (fn [idx book-name]
               (let [row-correlations (nth corr-values idx)
                     ;; Součet všech korelací v řádku
                     sum-of-correlations (reduce + row-correlations)]
                 {:book book-name
                  :sum-correlation sum-of-correlations})) ; Změna na sumu
             col-names)]
        (-> (tc/dataset sum-correlations-data)
            (tc/order-by [:sum-correlation] :desc)))))) ; Třídění podle sumy


(book-sum-correlations corr-matrix)

(kind/plotly
 {:data [{:type "heatmap"
          :z (:corr-values corr-matrix)
          :x (:col-names corr-matrix)
          :y (:col-names corr-matrix)
          :colorscale "RdBu"
          :zmid 0}]
  :layout {:title "Korelace top x nejčastějších knih"
           :xaxis {:tickangle 45}
           :width 1200
           :height 900}})


;; # A nyní predikce

(def processed-ds-numeric
  (-> raw-ds
      create-one-hot-encoding
      (ds/categorical->number [:next-predicted-buy])))

(kind/table
 (tc/head processed-ds-numeric))


(kind/table
 (ds/head
  (tc/info processed-ds-numeric)
  30))

(kind/table
 (ds/head processed-ds-numeric))

(def split
  (-> processed-ds-numeric
      (tc/split->seq  :holdout {:seed 42})
      first))

#_(def xgb-model ;; bacha, sekne se
    (sk-clj/fit (:train split) :sklearn.ensemble "GradientBoostingClassifier"
                {:n_estimators 100
                 :learning_rate 0.1
                 :max_depth 6
                 :random_state 42}))


#_(def et-model ;; 0.03
    (sk-clj/fit (:train split) :sklearn.ensemble "ExtraTreesClassifier"
                {:n_estimators 200
                 :max_depth 15
                 :random_state 42
                 :class_weight "balanced"}))

#_(def sgd-model ;; 0.03
    (sk-clj/fit (:train split) :sklearn.linear_model "SGDClassifier"
                {:loss "log_loss"
                 :penalty "elasticnet"
                 :alpha 0.001
                 :random_state 42
                 :class_weight "balanced"}))

#_(def logistic-model ;; 0.016
    (sk-clj/fit (:train split) :sklearn.linear_model "LogisticRegression"
                {:penalty "l1"
                 :solver "liblinear"
                 :C 0.1
                 :random_state 42
                 :class_weight "balanced"}))

#_(def linear-svc-model ;; 0.12
  (sk-clj/fit (:train split) :sklearn.svm "LinearSVC"
              {:dual false
               :random_state 42
               :tol 0.5
               :max_iter 80})) ; {:loss "hinge", :class_weight "balanced"} blbý

#_(def dtree-model ;; 0.11
    (sk-clj/fit (:train split) :sklearn.tree "DecisionTreeClassifier"
                {:random_state 42}))


#_(def rf-model ;; 0.03
    (sk-clj/fit (:train split) :sklearn.ensemble "RandomForestClassifier"
                {:n_estimators 200
                 :max_depth 10
                 :min_samples_split 5
                 :random_state 42
                 :class_weight "balanced"}))

(def nb-model ;; 0.13
    (sk-clj/fit (:train split) :sklearn.naive_bayes "MultinomialNB"))

#_(def knn-model ;; 0.14
    (sk-clj/fit (:train split) :sklearn.neighbors "KNeighborsClassifier"
                {:algorithm "auto"
                 :n_neighbors 90
                 :weights "distance"
                 :metric "cosine"}))

(-> (py/import-module "sklearn.svm")
    (py/get-attr "__dict__")
    py/->jvm
    keys)

;; Helper funkce pro konverzi čísel zpět na kategorie - NATIVNÍ PŘÍSTUP
(defn transfer-categorical-metadata
  "Přenese categorical metadata z reference datasetu do target datasetu"
  [target-dataset reference-dataset column-name]
  (let [ref-col (ds/column reference-dataset column-name)
        ref-meta (meta ref-col)
        target-col (ds/column target-dataset column-name)]
    (ds/new-column column-name
                   target-col
                   ref-meta)))

(defn convert-predictions-to-categories
  "Převede číselné predikce zpět na kategorie pomocí nativní ds-cat/reverse-map-categorical-xforms"
  [prediction-dataset reference-dataset]
  (let [;; Přeneseme categorical metadata z reference datasetu
        prediction-with-metadata (-> prediction-dataset
                                     (ds/remove-column :next-predicted-buy)
                                     (ds/add-column (transfer-categorical-metadata
                                                     prediction-dataset
                                                     reference-dataset
                                                     :next-predicted-buy)))]
    (ds-cat/reverse-map-categorical-xforms prediction-with-metadata)))

;; NATIVNÍ PŘÍSTUP pro accuracy measurement
(loss/classification-accuracy
 (-> (:test split)
     (ds-cat/reverse-map-categorical-xforms)
     (ds/column :next-predicted-buy))
 (-> (sk-clj/predict (:test split) nb-model)
     (convert-predictions-to-categories (:train split))
     (ds/column :next-predicted-buy)))

;; Helper funkce pro predikce - NATIVNÍ PŘÍSTUP
(defn predict-next-book
  "Predikuje další knihu na základě vlastněných knih - používá nativní categorical konverzi"
  [owned-books my-model]
  (try
    (let [; Zajistíme, že owned-books je vždy kolekce
          owned-books-coll (if (coll? owned-books) owned-books [owned-books])
          train-features (-> (:train split)
                             (ds/drop-columns [:next-predicted-buy])
                             tc/column-names)
          ; Filtrujeme pouze knihy, které existují v trénovacích datech
          valid-books (filter #(contains? (set train-features) %) owned-books-coll)
          zero-map (zipmap train-features (repeat 0))
          input-data (merge zero-map (zipmap valid-books (repeat 1)))
          ;; Vytvoříme input dataset s placeholder target sloupcem a nastavíme inference target
          full-input-data (assoc input-data :next-predicted-buy nil)
          input-ds (-> (ds/->dataset [full-input-data])
                       (ds-mod/set-inference-target [:next-predicted-buy]))
          ;; Predikce pomocí sklearn modelu
          raw-pred (sk-clj/predict input-ds my-model)
          ;; NATIVNÍ KONVERZE: Použijeme ds-cat/reverse-map-categorical-xforms místo ruční konverze
          predicted-categories-ds (convert-predictions-to-categories raw-pred (:train split))
          predicted-category (-> predicted-categories-ds
                                 (ds/column :next-predicted-buy)
                                 first)]
      predicted-category)
    (catch Exception e
      (println "Chyba v predict-next-book:" (.getMessage e))
      (println "Stack trace:")
      (.printStackTrace e)
      nil)))

(defn predict-next-n-books [input n]
  (loop [acc []
         predict-from input
         idx n]
    (let [predicted (predict-next-book predict-from nb-model)]
      (if (> idx 0)
        (recur (conj acc predicted) (conj predict-from predicted) (dec idx))
        (distinct acc)))))

(predict-next-book [:mit-vse-hotovo] nb-model)

(predict-next-n-books [:rozvrat] 5)

