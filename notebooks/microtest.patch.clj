(ns microtest.patch
  (:require [scicloj.metamorph.ml]))


(in-ns 'scicloj.metamorph.ml)
(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the
  options map selects which model definition to use to train the model.  Returns a map
  containing at least:


  * `:model-data` - the result of that definitions's train-fn.
  * `:options` - the options passed in.
  * `:id` - new randomly generated UUID.
  * `:feature-columns` - vector of column names.
  * `:target-columns` - vector of column names.
  * `:target-datatypes` - map of target columns names -> target columns type
  * `:target-categorical-maps` - the categorical maps of the target columns, if present

 A well behaving model implementaion should use
   :target-column
   :target-datatypes
   :target-categorical-maps

   to construct its prediction dataset so that its matches with the train data target column.
   "
  {:malli/schema [:=> [:cat [:fn dataset?] map?]
                  [map?]]}
  [dataset options]

  ;(assert-categorical-consistency dataset)


  (let [model-options (options->model-def options)
        _ (when (some? (:options model-options))
            (validate-options model-options options))

        combined-hash (when (:use-cache @train-predict-cache)
                        (str  (hash dataset) "___"  (hash options)))

        cached (when combined-hash ((:get-fn @train-predict-cache) combined-hash))]

    (if cached
      cached
      (let [{:keys [train-fn unsupervised?]} model-options
            feature-ds (cf/feature  dataset)
            _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                     "No features provided")
            target-ds (if unsupervised?
                        nil
                        (do
                          (errors/when-not-error (> (ds/row-count (cf/target dataset)) 0) "No target columns provided, see tech.v3.dataset.modelling/set-inference-target")
                          (cf/target dataset)))
            model-data (train-fn feature-ds target-ds
                                 options)
            ;; _ (errors/when-not-error (:model-as-bytes model-data)  "train-fn need to return a map with key :model-as-bytes")
            targets-datatypes
            (zipmap
             (keys target-ds)
             (->>
              (vals target-ds)
              (map meta)
              (map :datatype)))
            cat-maps (ds-mod/dataset->categorical-xforms target-ds)

            model
            (merge
             {:model-data model-data
              :options options
              :train-input-hash combined-hash
              :id (UUID/randomUUID)
              :feature-columns (vec (ds/column-names feature-ds))
              :target-columns (vec (ds/column-names target-ds))
              :target-datatypes targets-datatypes}
             (when-not (== 0 (count cat-maps))
               {:target-categorical-maps cat-maps}))]
        (when combined-hash
          ((:set-fn @train-predict-cache) combined-hash model))

        model))))

(in-ns 'microtest.patch)
