(ns nextbook-libpython-categorical
  (:import [java.text Normalizer Normalizer$Form])
  (:require
   [fastmath.ml.clustering :as fm-cluster]
   [scicloj.ml.smile.clustering :as smile-cluster]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [scicloj.kindly.v4.kind :as kind]
   [tech.v3.dataset :as ds]
   [tablecloth.api :as tc]
   [scicloj.tableplot.v1.plotly :as plotly]
   [fastmath.stats]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.preprocessing :as prepro]
   [clojure.string :as str]))

(require
 '[libpython-clj2.python :as py]
 '[clojure.string :as str]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.categorical :as ds-cat]
 '[scicloj.metamorph.ml.loss :as loss])

(py/initialize! {:python-executable "/Users/tomas/miniconda3/bin/python"}) ;; pro iMac

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

;; ## Pomocná funkce pro sanitizaci jmen knih
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
                                ["balicek" "poukaz" "zapisnik" "limitovana-edice" "taska" "aktualizovane-vydani" "cd"])))
       distinct
       (mapv keyword)))

(def orders-raw-ds
  (tc/dataset
   "/Users/tomas/Downloads/wc-orders-report-export-1749189240838.csv"
   {:header? true :separator ","
    #_#_:column-allowlist ["Produkt (produkty)" "Zákazník"]
    #_#_:num-rows 2000
    :key-fn #(keyword (sanitize-column-name-str %))})) ;; tohle upraví jen názvy sloupců!

;; ### Výpis zadarmovek

(defn how-many-books [s]
  (if s
    (->> s
         (re-seq #"(\d+)×")
         (map #(Integer/parseInt (second %)))
         (reduce +))
    0))

(def zadarmovky-with-counts
  (-> orders-raw-ds
      (ds/drop-missing :zakaznik)
      (tc/select-rows #(zero? (:net-sales %)))
      (tc/group-by [:zakaznik])
      (tc/aggregate {:all-products #(str/join ", " (ds/column % :produkt-produkty))})
      (tc/select-columns [:net-sales :zakaznik :all-products])
      (tc/map-columns :how-many-books [:all-products]
                      (fn [produkty] (how-many-books produkty)))
      (ds/sort-by #(:how-many-books %))
      (ds/drop-columns [:net-sales])))

(tc/info
 (-> zadarmovky-with-counts
     (tc/select-rows #(-> % :how-many-books (= 1)))))

(-> zadarmovky-with-counts
    (tc/select-rows #(-> % :how-many-books (> 1)))
    (plotly/layer-bar {:=x :zakaznik
                       :=y :how-many-books
                       :background-color "beige"})
    (assoc-in [:layout] {:width 1200
                         :title "Výtisky, které měly 0 Kč"}))

(-> zadarmovky-with-counts
    #_(tc/group-by [:zakaznik])
    #_(tc/aggregate {:n tc/row-count})
    (plotly/layer-histogram {:=x :how-many-books
                             :=histogram-nbins 20})
    (assoc-in [:layout] {:width 1200
                         :title "Výtisky, které měly 0 Kč"}))



(tc/sum zadarmovky-with-counts :how-many-books)


;; # Funkce pro agregaci datasetu po zákaznících + one-hot-encoding


(defn aggregate-ds-onehot [raw-ds]
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
        #_(ds/drop-columns [""]))))


;; ### Co řádek, to zákazník, ale bez roznásobení kvůli predikci
(def simple-ds-onehot (aggregate-ds-onehot orders-raw-ds))

(def columns (-> simple-ds-onehot
                 (ds/drop-columns [:zakaznik])
                 ds/column-names))

;; ### Kolik zákazníků koupilo danou knihu
(def counts
  (-> (map (fn [col]
             [col (-> (tc/sum simple-ds-onehot col)
                      (ds/column "summary")
                      first)])
           columns)
      (tc/dataset)
      (tc/rename-columns [:book :customers-bought])
      (tc/order-by :customers-bought :desc)))

(kind/table counts)

;; ## Korelační matice 

;; ### Tady děláme korelace mezi knihami, používáme `pandas`
(defn pandas-correlation-and-sums [dataset n-top-books]
  (let [numeric-features (ds/drop-columns dataset [:zakaznik])

        ;; Najdeme top knihy
        top-books (->> (ds/column-names numeric-features)
                       (map (fn [col] [col (reduce + (ds/column numeric-features col))]))
                       (sort-by second >)
                       (take n-top-books)
                       (map first))

        ;; Filtrujeme dataset
        top-ds (ds/select-columns numeric-features top-books)

        ;; Pandas interop - použijeme pandas pro korelaci
        pd (py/import-module "pandas")
        ;; Python: import pandas as pd

        df (py/call-attr pd "DataFrame"
                         (into {} (map (fn [col]
                                         [(name col) (vec (ds/column top-ds col))])
                                       top-books)))
        ;; Python: df = pd.DataFrame({col_name: dataset[col_name].tolist() for col_name in top_books})

        ;; Korelační matice a součty
        corr-matrix (py/call-attr df "corr")
        ;; Python: corr_matrix = df.corr()

        corr-values (py/->jvm (py/get-attr corr-matrix "values"))
        ;; Python: corr_values = corr_matrix.values

        col-names (py/->jvm (py/get-attr df "columns"))
        ;; Python: col_names = df.columns

        ;; Součty přímo z pandas
        sum-values (py/->jvm (py/get-attr (py/call-attr corr-matrix "sum") "values"))
        ;; Python: sum_values = corr_matrix.sum().values


        ;; Vytvoříme dataset se součty
        sum-ds (-> (map (fn [book sum] {:book book :sum-correlation sum})
                        col-names sum-values)
                   tc/dataset
                   (tc/order-by [:sum-correlation] :desc))
        ;; Python: sum_df = pd.DataFrame({'book': col_names, 'sum_correlation': sum_values})
        ;;         sum_df = sum_df.sort_values('sum_correlation', ascending=False)
        ]

    {:correlation-values corr-values
     :column-names col-names
     :correlation-sums sum-ds}))

(def corr-matrix
  (pandas-correlation-and-sums simple-ds-onehot 75)) ; Třídění podle sumy

(kind/plotly
 {:data [{:type "heatmap"
          :z (:correlation-values corr-matrix)
          :x (:column-names corr-matrix)
          :y (:column-names corr-matrix)
          :colorscale "RdBu"
          :zmid 0}]
  :layout {:title "Korelace top x nejčastějších knih"
           :xaxis {:tickangle 45}
           :width 1200
           :height 900}})


(kind/table (:correlation-sums corr-matrix))

;; ## Nejjednodušší řešení 

(-> (tc/select-columns simple-ds-onehot (take 15 (reverse columns)))
    (plotly/layer-correlation)
    (plotly/plot)
    (assoc-in [:layout] {:title "Korelace nově"
                         :width 1200
                         :height 900
                         :xaxis {:tickangle 45}}))


;; # Clustering - pokusy
(clojure.repl/doc fm-cluster/kmeans++)

(def smallish-ds
  (ds/head (-> simple-ds-onehot
               (ds/drop-columns [:zakaznik]))
           1000))

;; 2. Convert correlation matrix to a distance matrix (1 - correlation)

(defn cluster-books-by-cooccurrence [ds n-clusters]
  (let [;; Get book columns (excluding customer ID)
        book-columns (-> ds
                         (ds/drop-columns [:zakaznik])
                         ds/column-names)

        _ (println book-columns)
        ;; Create correlation matrix between books
        book-correlations (-> ds
                              (ds/drop-columns [:zakaznik])
                              (pandas-correlation-and-sums 75))

        _ (println "---- korelace: " book-correlations)
        ;; Convert to distance matrix (transpose for book-to-book relationships)
        book-features (-> book-correlations
                          ds/rows
                          vec)
        _ (println "---- fíčury: " book-features)

        ;; Apply k-means clustering to books
        book-names (vec book-columns)
        clusters nil #_(fm-cluster/kmeans++ book-features n-clusters)]

    {:book-names book-names
     :clusters clusters
     #_#_:cluster-assignments (map #(:cluster %) clusters)}))

;; Cluster books into groups
#_(def book-cluster-result
    (cluster-books-by-cooccurrence smallish-ds 5))

;; Group books by cluster
#_(def books-by-cluster
    (group-by second
              (map vector
                   (:book-names book-cluster-result)
                   (:cluster-assignments book-cluster-result))))


;; # Hraju si se sloupci

(let [data (-> orders-raw-ds
               #_(tc/head 1000)
               (tc/group-by :zakaznik)
               (tc/aggregate {:pocet-objednavek tc/row-count
                              :celkem-prodane-produkty #(reduce + (:prodane-produkty %))})
               (tc/map-columns :ratio [:celkem-prodane-produkty :pocet-objednavek]
                               (fn [produkty objednavky] (if (zero? objednavky) 0 (/ produkty objednavky))))
               (tc/select-rows #(= (:celkem-prodane-produkty %) 1))
               (tc/order-by :ratio :desc)
               (tc/update-columns {:ratio #(map double %)}))

      ;; Calculate summary statistics
      info (tc/info data)]
  [(kind/table data {:use-datatables true
                    :datatables {:scrollY 800}})
   info])

;; # Pomocná funkce pro vizualizaci poměru prodaných produktů k objednávkám přes jednotlivé zákazníky

(-> orders-raw-ds
    #_(tc/head 1000)
    (tc/group-by :zakaznik)
    (tc/aggregate {:pocet-objednavek tc/row-count
                   :celkem-prodane-produkty #(reduce + (:prodane-produkty %))})
    (tc/map-columns :ratio [:celkem-prodane-produkty :pocet-objednavek]
                    (fn [produkty objednavky] (if (zero? objednavky) 0 (/ produkty objednavky))))
    (tc/order-by :ratio :desc)
    (tc/update-columns {:ratio #(map double %)})
    (plotly/layer-histogram {:=x :ratio
                             :=histogram-nbins 20})
    (assoc-in [:layout] {:yaxis {:type "log"
                                 :title "Počet zákazníků (log škála)"}
                         :xaxis {:title "Poměr produktů na objednávku"}
                         :title "Poměr prodaných produktů k objednávkám"
                         :width 800
                         :height 600}))


    ;; # Predikce
    ;; ## Příprava na predikci

    (defn one-hot-encode [raw-ds]
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


    (def multimplied-ds-onehot
      (-> orders-raw-ds
          one-hot-encode
          (ds/categorical->number [:next-predicted-buy])))

    (kind/table
     (tc/head multimplied-ds-onehot))


    (kind/table
     (ds/head
      (-> (tc/info multimplied-ds-onehot)
          (tc/order-by :skew))
      30))


    ;; ### Vysvětlení
    ;;  Ve vašem případě s one-hot encoded daty (pouze hodnoty 0 a 1) je vysoké pozitivní skew (2-20) úplně normální a očekávané.

    ;; **Proč máte takové hodnoty:**

    ;; Pro binární sloupce (0/1) platí:
    ;; - **skew = (q - p) / √(pq)** kde p = pravděpodobnost 1, q = pravděpodobnost 0
    ;; - Pokud má knihu málo zákazníků (např. 5% = p=0.05), pak:
    ;;   - q = 0.95
    ;;   - skew = (0.95 - 0.05) / √(0.05 × 0.95) ≈ 4.1

    ;; **Interpretace pro vaše knihy:**
    ;; - **skew 2-5**: Knihu má asi 10-20% zákazníků
    ;; - **skew 5-10**: Knihu má asi 5-10% zákazníků  
    ;; - **skew 10-20**: Knihu má méně než 5% zákazníků (vzácné knihy)

    ;; **Praktické využití:**
    ;; - Knihy s vysokým skew (>15) jsou velmi vzácné - možná je vyřadit z analýzy
    ;; - Knihy se středním skew (5-10) jsou dobré pro doporučování
    ;; - Knihy s nízkým skew (2-5) jsou populárnější

    ;; To sedí s vaším business případem - většina knih je koupena pouze malým počtem zákazníků, proto vidíte převážně vysoké pozitivní skew hodnoty.

    (kind/table
     (ds/head multimplied-ds-onehot))

    (def split
      (-> multimplied-ds-onehot
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

    (kind/hidden ;; jen výpis modulů pro pozdější použití
     (-> (py/import-module "sklearn.svm")
         (py/get-attr "__dict__")
         py/->jvm
         keys))

    ;; ### Helper funkce pro konverzi čísel zpět na kategorie - NATIVNÍ PŘÍSTUP
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

    ;; ### NATIVNÍ PŘÍSTUP pro accuracy measurement
    (loss/classification-accuracy
     (-> (:test split)
         (ds-cat/reverse-map-categorical-xforms)
         (ds/column :next-predicted-buy))
     (-> (sk-clj/predict (:test split) nb-model)
         (convert-predictions-to-categories (:train split))
         (ds/column :next-predicted-buy)))

    ;; ### Helper funkce pro predikce - NATIVNÍ PŘÍSTUP
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

    (predict-next-book [:jak-na-adhd] nb-model)

    (predict-next-n-books [:jak-na-adhd] 4)

