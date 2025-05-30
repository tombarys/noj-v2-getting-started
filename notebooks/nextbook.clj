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
              #_#_:num-rows 5000
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

(kind/table
 (ds/sample processed-ds 20))

(def split
  (first
   (tc/split->seq processed-ds :holdout {:seed 42})))

(comment
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
      :tribuo-trainer-name "random-forest"})))


(comment
  (loss/classification-accuracy
   (ds/column (:test split)
              :next-predicted-buy)
   (ds/column (ml/predict (:test split) rf-model)
              :next-predicted-buy))
  )

(comment
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
  )


(comment
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
  )



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



(comment (def xgboost-classification-model
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
    :tribuo-trainer-name "xgboost-classification"})))

(comment
  (loss/classification-accuracy
   (ds/column (:test split)
              :next-predicted-buy)
   (ds/column (ml/predict (:test split) xgboost-classification-model)
              :next-predicted-buy)))


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


;; # Doporučení s modelem a bez

;; ## Nejdříve s modelem
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


;;  ## === Model-based recommendation ===

(better-model-predict xgboost-simple-model [:telo-scita-rany] :top-k 5)

;; ### Test s více knihami

(better-model-predict xgboost-simple-model [:jed-dal :krad-jako-umelec :ukaz-co-delas!] :top-k 5)

;; ## === Collaborative filtering ===

(collaborative-recommend [:konec-prokrastinace] :top-k 5)

;; ### Test s více knihami

(collaborative-recommend [:jed-dal :krad-jako-umelec :ukaz-co-delas!] :top-k 5)

