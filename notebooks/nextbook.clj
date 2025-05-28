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


(defn sanitize-name-str [s]
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
              #_:num-rows #_10000
              :column-allowlist ["Produkt (produkty)"]
              :key-fn #(keyword (sanitize-name-str %))})) ;; tohle upraví jen názvy sloupců!


(kind/table
 (tc/info raw-ds))

(kind/table
 (ds/sample
  (tc/select-rows raw-ds #(str/includes? (str (:produkt-produkty %)) "Ukaž")) 100))



;; ## Pomocné funkce pro sanitizaci sloupců

(defn books-only [order]
  (remove (fn [item]
            (some (fn [substr] (str/includes? (name item) substr))
                  ["balicek" "poukaz" "zapisnik"]))  order))

(defn parse-books [s]
  (->> (str/split s #",\s\d+") ;; FIXME nutno opravit tento split
       (map #(str/replace % #"\d*×\s" ""))
       (map #(str/replace % #"," ""))
       (map #(str/replace % #"\[|\]|komplet|a\+e|\s\(P\+E\)|\s\(e\-kniha\)|\s\(P\+E\)|\s\(P\+A\)|\s\(E\+A\)|papír|papir|audio|e\-kniha" ""))
       (map #(str/replace % #"\+" ""))
       (map #(str/trim %))
       (map sanitize-name-str)
       (map #(str/replace % #"\-+$" ""))
       (map #(str/replace % #"3" "k3"))
       (remove (fn [item]
                 (some (fn [substr] (str/includes? (name item) substr))
                       ["balicek" "poukaz" "zapisnik"])))
       distinct
       (mapv keyword)))

(def parsed-books
  (parse-books "1× Balíček, 1× 365 Skrytý potenciál, 1× Nové zbraně vlivu, 1× Ukaž, co děláš! (e-kniha), 1× Alchymie (audio), 1× Dárkový poukaz 1 000 Kč "))

parsed-books

(def ds 
  (tc/select-rows raw-ds #(str/includes? (str (:produkt-produkty %)) "Ukaž")))

ds

(def parsed-real-ds
  (->> 
   (ds/column raw-ds :produkt-produkty)
   (mapcat parse-books)
   distinct
   sort))

parsed-real-ds


(defn create-one-hot-encoding [raw-ds]
  (let [;; Získáme všechny unikátní knihy ze všech řádků
        all-books (->> (ds/column raw-ds :produkt-produkty)
                       (mapcat parse-books)
                       distinct
                       sort)

        ;; Pro každý řádek vytvoříme mapu s hodnotami pro one-hot encoding
        ;; ale pouze pro řádky s více než jednou knihou
        rows-with-books (keep-indexed
                         (fn [idx product-string]
                           (let [books-in-row (parse-books product-string)]
                             (when (> (count books-in-row) 1)  ; Filtrujeme řádky s více než jednou knihou
                               (let [;; Rozdělíme knihy - všechny kromě poslední pro features, poslední pro target
                                     feature-books (set (butlast books-in-row))
                                     target-book (last books-in-row)
                                     one-hot-map (reduce (fn [acc book]
                                                           (assoc acc book (if (contains? feature-books book) 1 0)))
                                                         {}
                                                         all-books)]
                                 (merge {:row-index idx
                                         :next-predicted-buy target-book}
                                        one-hot-map)))))
                         (ds/column raw-ds :produkt-produkty))

        ;; Vytvoříme nový dataset z one-hot dat
        one-hot-ds (tc/dataset rows-with-books)

        ;; Získáme indexy řádků, které chceme zachovat
        valid-indices (set (map :row-index rows-with-books))]

    ;; Spojíme původní dataset (bez sloupce produkt-produkty) s one-hot encoding
    ;; ale pouze pro řádky s více knihami
    (-> raw-ds
        (tc/drop-columns [:produkt-produkty])
        (tc/add-column :row-index (range (tc/row-count raw-ds)))
        (tc/select-rows #(contains? valid-indices (:row-index %)))
        (tc/left-join one-hot-ds :row-index)
        (tc/drop-columns [:row-index])
        (ds-mod/set-inference-target [:next-predicted-buy]))))

(def processed-ds (create-one-hot-encoding raw-ds))

processed-ds

(def split
  (first
   (tc/split->seq processed-ds :holdout {:seed 112223})))

split

(def rf-model
  (ml/train
   (:train split)
   {:model-type :scicloj.ml.tribuo/classification
    :tribuo-components [{:name "random-forest"
                         :target-columns [:next-predicted-buy]
                         :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                         :properties {:maxDepth "24"
                                      :useRandomSplitPoints "true"
                                      :fractionFeaturesInSplit "0.8"}}]
    :tribuo-trainer-name "random-forest"}))


;; # Make predictions
(def rf-predictions
  (ml/predict (:test split) rf-model))

(def rf-accuracy
  (loss/classification-accuracy
   (vec (ds/column (:test split) :next-predicted-buy))
   (vec (ds/column rf-predictions :next-predicted-buy))))

rf-accuracy