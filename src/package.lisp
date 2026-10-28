(uiop:define-package #:trinity-ggs
    (:use #:cl #:alexandria #:ggs/eqsat)
  (:export
   ;; IR
   #:read-ir #:read-ir-from-string #:read-shapes #:term-string #:*shapes*
   ;; Optimization pipeline
   #:saturate #:extract-best #:postprocess #:program-cost
   #:optimize-program #:optimize-instance #:run-benchmarks
   #:compare-with-trinity
   ;; Cost model
   #:cost-breakdown #:*hardware*))
