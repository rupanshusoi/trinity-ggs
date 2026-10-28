(asdf:defsystem #:trinity-ggs
  :description "Trinity's tile-level equality saturation, ported to GGS EqSat,
with a CPU-only analytical cost model in place of GPU profiling."
  :license "MIT License"
  :depends-on (:alexandria :ggs/eqsat)
  :pathname "src/"
  :serial t
  :components ((:file "package")
               (:file "ir")
               (:file "shape")
               (:file "egraph")
               (:file "analysis")
               (:file "rules")
               (:file "run")
               (:file "extract")
               (:file "postprocess")
               (:file "cost")
               (:file "driver"))
  :in-order-to ((test-op (test-op "trinity-ggs/tests"))))

(asdf:defsystem #:trinity-ggs/tests
  :depends-on (:trinity-ggs :fiveam)
  :pathname "tests/"
  :components ((:file "tests"))
  :perform (test-op (o c) (symbol-call :fiveam '#:run! :trinity-ggs)))
