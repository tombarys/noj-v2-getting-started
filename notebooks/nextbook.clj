(ns nextbook
  (:require
   [tech.v3.dataset :as ds]
   [scicloj.ml.tribuo]
   [clojure.string :as str]
   [scicloj.kindly.v4.kind :as kind]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.metamorph.ml.loss :as loss])
  (:import [java.text Normalizer Normalizer$Form]))


(defn sanitize-column-name-str [s]
  (if-not (and (nil? s) (empty? s))
    (let [hyphens (str/replace s #"_" "-")
          trimmed (str/trim hyphens)
          nfd-normalized (Normalizer/normalize trimmed Normalizer$Form/NFD)
          no-diacritics (str/replace nfd-normalized #"\p{InCombiningDiacriticalMarks}+" "") ; dočasně
          no-spaces (str/replace no-diacritics #" " "-")
          no-brackets (str/replace no-spaces #"\(|\)" "")
          ;; Můžete přidat další pravidla, např. odstranění speciálních znaků
          ;; alphanumeric-and-underscore (str/replace no-diacritics #"[^a-zA-Z0-9_]" "")
          lower-cased (str/lower-case no-brackets)]
      lower-cased)
    s))

;(ns-unmap *ns* 'raw-ds)

(def raw-ds (tc/dataset
             "/Users/tomas/Downloads/wc-orders-report-export-17477349086991.csv"
             {:header? true :separator ","
              :column-allowlist ["Produkt (produkty)" "Zákazník"]
              #_#_:num-rows 5000
              :key-fn #(keyword (sanitize-column-name-str %))})) ;; tohle upraví jen názvy sloupců!


 ; (tc/select-rows raw-ds #(re-find #"A.E" (str (:produkt-produkty %))))

(kind/table
 (tc/info raw-ds))

(kind/table
 (ds/sample
  (tc/select-rows raw-ds #(str/includes? (str (:produkt-produkty %)) "Ukaž")) 100))


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


(def parsed-real-ds
  (->>
   (ds/column raw-ds :produkt-produkty)
   (mapcat parse-books)
   distinct
   sort))

parsed-real-ds


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
        one-hot-ds (tc/dataset rows-with-books)]

    ;; Vrátíme dataset s one-hot encoding a nastaveným inference targetem
    (-> one-hot-ds
        (ds/categorical->number [:zakaznik])
        #_(ds/drop-columns [:zakaznik])
        (ds-mod/set-inference-target [:next-predicted-buy]))))

(def processed-ds (create-one-hot-encoding raw-ds))

processed-ds


(def ds
  (tc/select-rows processed-ds #(= 3000 (:zakaznik %))))

ds


(kind/table
 (ds/sample processed-ds 100))

(ds/unique-by-column processed-ds :zakaznik)



(kind/hiccup
 [:div {:style {:max-height "600px"
                :overflow-y :auto}}
  (kind/table
   processed-ds)])

(def split
  (first
   (tc/split->seq processed-ds :holdout {:seed 42})))

split

(def rf-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "random-forest"
                         :target-columns [:next-predicted-buy]
                         :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                         :properties {:maxDepth "11" ;; pro 11 je to 14.9
                                      :useRandomSplitPoints "true"
                                      :fractionFeaturesInSplit "1"}}]
    :tribuo-trainer-name "random-forest"}))


(loss/classification-accuracy
 (ds/column (:test split)
            :next-predicted-buy)
 (ds/column (ml/predict (:test split) rf-model)
            :next-predicted-buy))

(def rf2-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "random-forest"
                         :target-columns [:next-predicted-buy]
                         :type "org.tribuo.common.tree.RandomForestTrainer"
                         :properties {:numMembers "100"
                                      :seed "42"
                                      :innerTrainer "inner-trainer"
                                      :combiner "voting-combiner"}}
                        {:name "inner-trainer"
                         :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                         :properties {:maxDepth "10"
                                      :useRandomSplitPoints "false"
                                      :fractionFeaturesInSplit "0.5"}}
                        {:name "voting-combiner"
                         :type "org.tribuo.classification.ensemble.VotingCombiner"}]
    :tribuo-trainer-name "random-forest"}))

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


(def svm-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "svm"
                         :target-columns [:next-predicted-buy]
                         :type "org.tribuo.classification.libsvm.LibSVMClassificationTrainer"
                         :properties {:svmType "C_SVC"
                                      :kernelType "RBF"
                                      :gamma "0.1"
                                      :cost "1.0"}}]
    :tribuo-trainer-name "svm"}))



(def xgboost-simple-model ;; taky funguje
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

(loss/classification-accuracy
 (ds/column (:test split)
            :next-predicted-buy)
 (ds/column (ml/predict (:test split) xgboost-simple-model)
            :next-predicted-buy))



(def xgboost-classification-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "xgboost-classification"
                         :target-columns [:next-predicted-buy]
                         :type "org.tribuo.classification.xgboost.XGBoostClassificationTrainer"
                         :properties {:numTrees "100"
                                      :maxDepth "10"
                                      :eta "0.3"
                                      :subsample "1.0"
                                      :gamma "0.0"
                                      :minChildWeight "1"
                                      :lambda "1.0"
                                      :alpha "0.0"}}]
    :tribuo-trainer-name "xgboost-classification"}))

(loss/classification-accuracy
 (ds/column (:test split)
            :next-predicted-buy)
 (ds/column (ml/predict (:test split) xgboost-classification-model)
            :next-predicted-buy))


(comment
  (def nb-model
    (ml/train
     (:train split)
     {:model-type :scicloj.ml.tribuo/classification
      :tribuo-components [{:name "naive-bayes"
                           :target-columns [:next-predicted-buy]
                           :type "org.tribuo.classification.mnb.MultinomialNaiveBayesTrainer"
                           :properties {:alpha "1.0"}}]
      :tribuo-trainer-name "naive-bayes"})))


(comment
  (def lr-model
    (ml/train
     (:train split)
     {:model-type :scicloj.ml.tribuo/classification
      :tribuo-components [{:name "logistic-regression"
                           :target-columns [:next-predicted-buy]
                           :type "org.tribuo.classification.liblinear.LibLinearClassificationTrainer"
                           :properties {:solverType "L2R_LR"
                                        :cost "1.0"
                                        :epsilon "0.01"}}]
      :tribuo-trainer-name "logistic-regression"})))

;; # Make predictions
(def predictions
  (ml/predict (:test split) rf-model))

(def accuracy
  (loss/classification-accuracy
   (ds/column (:test split) :next-predicted-buy)
   (ds/column predictions :next-predicted-buy)))

accuracy


;; # Předpovědi a doporučení


(defn better-model-predict [model input-books & {:keys [top-k] :or {top-k 3}}]
  "Lepší přístup - najde zákazníky s podobnými knihami a podívá se, co si ještě koupili."
  (let [input-book-keywords (set (map #(if (keyword? %) % (keyword %)) input-books))
        all-books (->> (ds/column-names processed-ds)
                       (remove #{:zakaznik :next-predicted-buy})
                       (set))

        ;; Ověříme, jestli vstupní knihy existují
        found-books (filter #(contains? all-books %) input-book-keywords)

        _ (println "Nalezené knihy:" found-books)

        ;; Najdeme všechny řádky, kde zákazník má alespoň jednu ze vstupních knih
        train-rows (tc/rows (:train split) :as-maps)

        matching-rows (->> train-rows
                           (filter (fn [row]
                                     ;; Zákazník má alespoň jednu z hledaných knih
                                     (some #(= 1 (get row % 0)) found-books)))
                           (filter (fn [row]
                                     ;; Target kniha není jedna z vstupních knih
                                     (not (contains? input-book-keywords (:next-predicted-buy row))))))

        _ (println "Počet matching řádků:" (count matching-rows))

        ;; Místo úpravy vzorců použijeme model přímo na tyto řádky
        model-predictions (for [row matching-rows]
                            (try
                              (let [prediction-ds (-> (tc/dataset [row])
                                                      (ds-mod/set-inference-target [:next-predicted-buy]))
                                    prediction (ml/predict prediction-ds model)
                                    predicted-book (-> prediction
                                                       (ds/column :next-predicted-buy)
                                                       first)]
                                predicted-book)
                              (catch Exception e nil)))

        ;; Spočítáme frekvence
        valid-predictions (remove nil? model-predictions)
        _ (println "Počet validních predikcí:" (count valid-predictions))
        _ (println "Unikátní predikce:" (distinct valid-predictions))

        book-frequencies (frequencies valid-predictions)
        total-predictions (count valid-predictions)

        ;; Vytvoříme top-k podle frekvence
        top-predictions (->> book-frequencies
                             (sort-by second >)
                             (take top-k)
                             (map (fn [[book count]]
                                    {:book book
                                     :probability (/ (double count) total-predictions)})))]

    top-predictions))

;; Alternativně - jednoduché collaborative filtering bez modelu
(defn collaborative-recommend [input-books & {:keys [top-k] :or {top-k 3}}]
  "Jednoduché collaborative filtering - najde podobné zákazníky a podívá se, co si koupili."
  (let [input-book-keywords (set (map #(if (keyword? %) % (keyword %)) input-books))

        ;; Najdeme zákazníky, kteří mají alespoň jednu z vstupních knih
        train-rows (tc/rows (:train split) :as-maps)

        similar-customers (->> train-rows
                               (filter (fn [row]
                                         ;; Má alespoň jednu z hledaných knih
                                         (some #(= 1 (get row % 0)) input-book-keywords)))
                               (map :zakaznik)
                               distinct)

        _ (println "\n---\nPočet podobných zákazníků:" (count similar-customers))

        ;; Najdeme všechny knihy, které si tito zákazníci koupili
        books-bought-by-similar (->> train-rows
                                     (filter #(contains? (set similar-customers) (:zakaznik %)))
                                     (map :next-predicted-buy)
                                     (remove #(contains? input-book-keywords %))) ; Odstranit knihy, které už má
        
        book-frequencies (frequencies books-bought-by-similar)
        total-books (count books-bought-by-similar)

        ;; Top-k podle frekvence
        top-recommendations (->> book-frequencies
                                 (sort-by second >)
                                 (take top-k)
                                 (map (fn [[book count]]
                                        {:book book
                                         :probability (/ (double count) total-books)})))]

    top-recommendations))

;; Test obou přístupů
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
(println "=== Model-based recommendation ===")

(better-model-predict xgboost-simple-model [:nexus] :top-k 5)
(println "\n=== Collaborative filtering ===")

;; Test s více knihami
(collaborative-recommend [:nexus] :top-k 5)
(println "\n=== Test s více knihami ===")



(collaborative-recommend [:prezit :sport-je-bolest] :top-k 5)

;; Test hybrid přístupu
(hybrid-recommend xgboost-simple-model [:nexus] :top-k 5)
