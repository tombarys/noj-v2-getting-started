(ns nextbook-libpython
  (:import [java.text Normalizer Normalizer$Form]))

(require
 '[libpython-clj2.python :as py]
 '[tech.v3.dataset :as ds]
 '[scicloj.kindly.v4.kind :as kind]
 '[tablecloth.api :as tc]
 '[clojure.string :as str]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.categorical :as cat-mod]
 '[scicloj.metamorph.ml.loss :as loss])

(require
 '[scicloj.sklearn-clj :as sk-clj]
 '[scicloj.sklearn-clj.ml :as ml])

(py/initialize!)

;; Pomocné funkce
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
              :num-rows 500
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
        (ds/drop-columns [:zakaznik])
        (ds-mod/set-inference-target [:next-predicted-buy]))))

;; Nejdříve vytvoříme base dataset z one-hot encoding
(def processed-ds (create-one-hot-encoding raw-ds))

processed-ds

;; Pro categorical mapping použijeme ds/categorical->number který vytvoří metadata
(def processed-ds-numeric
  (ds/categorical->number processed-ds [:next-predicted-buy]))

(def split
  (-> processed-ds-numeric
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
  (sk-clj/fit (:train split) :sklearn.neighbors :k-neighbors-classifier {:n_neighbors 4}))

(loss/classification-accuracy
 (-> (ds/column (:test split) :next-predicted-buy)
     (vary-meta dissoc :categorical-map))  ; <- odstraní categorical metadata
 (ds/column (sk-clj/predict (:test split) log-reg)
            :next-predicted-buy))

;; Helper funkce pro konverzi čísel zpět na kategorie
(defn number->category 
  "Převede číselné predikce zpět na kategorie pomocí categorical metadata"
  [predicted-numbers target-column]
  (let [cat-map (:categorical-map (meta target-column))
        lookup-table (:lookup-table cat-map)]
    (if lookup-table
      ;; Vytvoříme inverzní mapu: číslo -> kategorie 
      (let [inverse-map (into {} (map (fn [[category number]] [number category]) lookup-table))]
        ;; Převedeme float hodnoty na int před hledáním v mapě
        (mapv #(inverse-map (int %)) predicted-numbers))
      ;; Pokud není k dispozici categorical mapping, vrátíme predikce tak jak jsou
      (do 
        (println "Varování: Categorical mapping není k dispozici, vracím číselné predikce")
        predicted-numbers))))

;; Helper funkce pro predikce
(defn predict-next-book
  "Predikuje další knihu na základě vlastněných knih"
  [owned-books]
  (try
    (let [train-features (-> (:train split)
                             (ds/drop-columns [:next-predicted-buy])
                             tc/column-names)
          ; Filtrujeme pouze knihy, které existují v trénovacích datech
          valid-books (filter #(contains? (set train-features) %) owned-books)
          _ (when (empty? valid-books)
              (println "Varování: Žádná z uvedených knih není v trénovacích datech"))
          _ (println "Debug: valid-books count:" (count valid-books))
          _ (println "Debug: train-features count:" (count train-features))
          zero-map (zipmap train-features (repeat 0))
          input-data (merge zero-map (zipmap valid-books (repeat 1)))
          ;; Vytvoříme input dataset s placeholder target sloupcem a nastavíme inference target
          full-input-data (assoc input-data :next-predicted-buy nil)
          input-ds (-> (ds/->dataset [full-input-data])
                       (ds-mod/set-inference-target [:next-predicted-buy]))
          _ (println "Debug: input-ds columns:" (tc/column-names input-ds))
          _ (println "Debug: input-ds shape:" [(ds/row-count input-ds) (ds/column-count input-ds)])
          _ (println "Debug: input-ds inference-target?" (some #(:inference-target? (meta %)) (tc/columns input-ds)))
          ;; Predikce pomocí sklearn modelu
          raw-pred (sk-clj/predict input-ds log-reg)
          _ (println "Debug: raw-pred columns:" (tc/column-names raw-pred))
          predicted-numbers (ds/column raw-pred :next-predicted-buy)
          _ (println "Debug: predicted-numbers:" predicted-numbers)
          target-column (ds/column (:train split) :next-predicted-buy)
          ;; Použijeme elegantní konverzi pomocí categorical metadata
          predicted-categories (number->category predicted-numbers target-column)]
      (first predicted-categories))
    (catch Exception e
      (println "Chyba v predict-next-book:" (.getMessage e))
      (println "Stack trace:")
      (.printStackTrace e)
      nil)))

;; Zjistíme, jaké knihy jsou dostupné v trénovacích datech
(def available-books 
  (-> (:train split)
      (ds/drop-columns [:next-predicted-buy])
      tc/column-names
      sort))

;; Ukážeme prvních 10 dostupných knih
(kind/hiccup [:div
              [:h3 "Prvních 10 dostupných knih:"]
              [:ul (for [book (take 10 available-books)]
                     [:li (name book)])]])

;; Test funkce s existující knihou
(println "Test predikce s první dostupnou knihou:")
(try
  (let [test-book (first available-books)
        prediction (predict-next-book [test-book])]
    (println "Vstupní kniha:" (name test-book))
    (println "Doporučená kniha:" (name prediction)))
  (catch Exception e
    (println "Chyba při predikci:" (.getMessage e))))

;; Diagnostické informace o categorical metadata
(let [target-col (ds/column (:train split) :next-predicted-buy)
      target-meta (meta target-col)]
  (println "\n=== Diagnostické informace ===")
  (println "Target column metadata klíče:" (keys target-meta))
  (when-let [cat-map (:categorical-map target-meta)]
    (println "Categorical map klíče:" (keys cat-map))
    (when-let [lookup (:lookup-table cat-map)]
      (println "Počet kategorií v lookup table:" (count lookup))
      (println "Prvních 5 kategorií:" (take 5 lookup)))))
