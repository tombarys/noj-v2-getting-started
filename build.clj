(ns build
  (:require [clojure.tools.build.api :as b]))

(def lib 'my/noj-v2-geting-started)
(def version (format "1.2.%s" (b/git-count-revs nil)))
(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
(def uber-file (format "target/%s-%s-standalone.jar" (name lib) version))

(defn clean [_]
  (b/delete {:path "target"}))

(defn uber [_]
  (clean nil)
  (b/copy-dir {:src-dirs ["notebooks" "resources"]
               :target-dir class-dir})
  (b/compile-clj {:basis basis
                  :src-dirs ["notebooks"]
                  :class-dir class-dir})
  (b/uber {:class-dir class-dir
           :uber-file uber-file
           :basis basis
           :main 'nextbook-libpython-categorical
           :conflict-handlers {"META-INF/license/LICENSE.base64.txt" :ignore
                                "META-INF/license/" :ignore
                                "META-INF/LICENSE" :ignore
                                "META-INF/LICENSE.txt" :ignore
                                "META-INF/NOTICE" :ignore
                                "META-INF/NOTICE.txt" :ignore}}))