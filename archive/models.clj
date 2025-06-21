(comment (def rf-model
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


(comment (def rf2-model
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
             :tribuo-trainer-name "random-forest"})))

(comment (loss/classification-accuracy
          (ds/column (:test split)
                     :next-predicted-buy)
          (ds/column (ml/predict (:test split) rf-model)
                     :next-predicted-buy)))


(comment (def svm-model
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
             :tribuo-trainer-name "svm"})))



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

(comment (loss/classification-accuracy
          (ds/column (:test split)
                     :next-predicted-buy)
          (ds/column (ml/predict (:test split) xgboost-classification-model)
                     :next-predicted-buy)))


(comment (def nb-model
           (ml/train
            (:train split)
            {:model-type :scicloj.ml.tribuo/classification
             :tribuo-components [{:name "naive-bayes"
                                  :target-columns [:next-predicted-buy]
                                  :type "org.tribuo.classification.mnb.MultinomialNaiveBayesTrainer"
                                  :properties {:alpha "1.0"}}]
             :tribuo-trainer-name "naive-bayes"})))


(comment  (def lr-model
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
