(uiop:define-package #:trinity-ggs/tests
    (:use #:cl #:trinity-ggs)
  (:import-from #:fiveam #:def-suite* #:def-test #:is #:is-true #:is-false))

(in-package #:trinity-ggs/tests)

(def-suite* :trinity-ggs)

(defun ir (string) (read-ir-from-string string))

(defun shapes (&rest specs)
  "Shape table from (NAME DIM...) SPECS."
  (let ((table (make-hash-table :test 'equal)))
    (loop for (name . dims) in specs
          do (setf (gethash (trinity-ggs::invert-case name) table) dims))
    table))

(defun run-opt (term shapes &rest args)
  "The optimized program and its cost."
  (multiple-value-bind (program cost) (apply #'optimize-program term shapes args)
    (values program cost)))

(defun egraph-has-p (term shapes predicate)
  "Whether some enode of TERM's saturated e-graph satisfies PREDICATE."
  (let ((*shapes* shapes))
    (let ((trinity-ggs::*egraph* (saturate term)))
      (some (lambda (class) (some predicate (trinity-ggs::class-nodes class)))
            (ggs/eqsat::egraph-class-list trinity-ggs::*egraph*)))))

(defun writes-only-p (class name)
  "Whether CLASS writes tensor NAME and nothing else."
  (let ((writes (trinity-ggs::class-writes class)))
    (and writes
         (every (lambda (a) (equal (trinity-ggs::access-names a)
                                   (list (trinity-ggs::invert-case name))))
                writes))))

(defun count-fsym (fsym term)
  (cond ((eq term fsym) 1)
        ((consp term) (reduce #'+ term :key (lambda (x) (count-fsym fsym x))))
        (t 0)))

;;; Reader and printer

(def-test reader-round-trip ()
  (let ((text "(store (tensor Q1,K1,V1) (load (input C_exp) (index fulltile (tile n))) (index fulltile (elem n)))"))
    (is (string= text (term-string (ir text))))
    (is (equal '("q1" "k1" "v1") (trinity-ggs::split-names (second (second (ir text))))))
    (is (equal (trinity-ggs::split-names (ir "C"))
               (list (first (trinity-ggs::split-names (ir "C,C_perturb"))))))
    (is (eq (ir "x") (ir "x")))
    (is (not (eq (ir "X") (ir "x"))))))

(def-test nullary-normalization ()
  (is (equal (ir "(index fulltile (tile n))") (ir "(index (fulltile) (tile n))"))))

;;; Rewrites

(defparameter *elementwise-chain*
  "(seq
     (loop 0 1024 tile_n n
       (store (tensor B) (exp (load (input A) (index fulltile (tile n)))) (index fulltile (tile n))))
     (loop 0 1024 tile_n n
       (store (output C) (sqrt (load (tensor B) (index fulltile (tile n)))) (index fulltile (tile n)))))")

(defparameter *chain-shapes* '(("A" 16 1024) ("B" 16 1024) ("C" 16 1024)))

(def-test fusion-and-forwarding ()
  "The two elementwise loops fuse into one kernel, B is forwarded and its store
eliminated."
  (multiple-value-bind (program cost)
      (run-opt (ir *elementwise-chain*) (apply #'shapes *chain-shapes*))
    (is (= 1 (+ (count-fsym 'trinity-ggs::ploop program) (count-fsym 'trinity-ggs::sloop program))))
    (is (= 1 (length (trinity-ggs::kernel-terms program))))
    (is (search "(sqrt (exp (load (input A)" (term-string program)))
    (is (< cost (let ((*shapes* (apply #'shapes *chain-shapes*)))
                  (program-cost (postprocess (ir *elementwise-chain*))))))))

(def-test reordering-respects-dependencies ()
  "Statements are only reordered when they touch disjoint tensors: B is
produced before C reads it, while the independent D may move to the front."
  (let ((term (ir "(seq
                    (loop 0 1024 tile_n n
                      (store (tensor B) (exp (load (input A) (index fulltile (tile n)))) (index fulltile (tile n))))
                    (seq
                      (loop 0 512 tile_k k
                        (store (output D) (sqrt (load (input E) (index fulltile (tile k)))) (index fulltile (tile k))))
                      (loop 0 1024 tile_m m
                        (store (output C) (+ (load (output C) (index fulltile)) (rsum (load (tensor B) (index fulltile (tile m))) 1)) (index fulltile)))))"))
        (shapes (shapes '("A" 16 1024) '("B" 16 1024) '("C" 16) '("E" 16 512) '("D" 16 512))))
    (let ((s (term-string (run-opt term shapes))))
      (is (< (search "(store (tensor B)" s) (search "(store (output C)" s))))
    (is-true (egraph-has-p term shapes
                           (lambda (node)
                             (and (eq (trinity-ggs::enode-fsym node) 'trinity-ggs::seq)
                                  (writes-only-p (trinity-ggs::enode-arg 0 node) "D")
                                  (trinity-ggs::class-writes (trinity-ggs::enode-arg 1 node))))))
    (is-false (egraph-has-p term shapes
                            (lambda (node)
                              (and (eq (trinity-ggs::enode-fsym node) 'trinity-ggs::seq)
                                   (writes-only-p (trinity-ggs::enode-arg 0 node) "C")
                                   (some (lambda (a) (equal (trinity-ggs::access-names a) '("b")))
                                         (trinity-ggs::class-writes (trinity-ggs::enode-arg 1 node)))))))))

(def-test accumulation-not-repeated ()
  "The reduction into C does not depend on n, but nesting it in the loop over n
would accumulate it once per iteration, so no loop over n may write C."
  (let ((term (ir "(seq
                    (loop 0 1024 tile_n n
                      (store (output B) (exp (load (input A) (index fulltile (tile n)))) (index fulltile (tile n))))
                    (loop 0 512 tile_k k
                      (store (output C) (+ (load (output C) (index fulltile)) (rsum (load (input E) (index fulltile (tile k))) 1)) (index fulltile))))"))
        (shapes (shapes '("A" 16 1024) '("B" 16 1024) '("C" 16) '("E" 16 512))))
    (is-true (run-opt term shapes))
    (is-false (egraph-has-p term shapes
                            (lambda (node)
                              (and (eq (trinity-ggs::enode-fsym node) 'trinity-ggs::loop)
                                   (eq (trinity-ggs::class-atom (trinity-ggs::enode-arg 3 node)) 'trinity-ggs::n)
                                   (some (lambda (a) (equal (trinity-ggs::access-names a) '("c")))
                                         (trinity-ggs::class-writes (trinity-ggs::enode-arg 4 node)))))))))

(def-test tile-to-elem-fusion ()
  "A loop of 1024/128 = 8 iterations fuses into a loop with step 128, indexing
the element (elem n), as Trinity's tmp_loop does."
  (let ((program (run-opt (ir "(seq
                                (loop 0 1024 128 n
                                  (store (tensor B) (exp (load (input A) (index fulltile (tile n)))) (index fulltile (tile n))))
                                (loop 0 8 tile_h h
                                  (store (output C) (sqrt (load (input D) (index (tile h) fulltile))) (index (tile h) fulltile))))")
                          (shapes '("A" 16 1024) '("B" 16 1024) '("D" 8 16) '("C" 8 16)))))
    (is (search "(index (elem n) fulltile)" (term-string program)))
    (is (= 1 (length (trinity-ggs::kernel-terms program))))))

(def-test scoping ()
  (is-true (trinity-ggs::well-scoped-p
            (ir "(loop 0 8 tile_n n (store (tensor A) (load (input B) (index (tile n))) (index (elem n))))")))
  (is-false (trinity-ggs::well-scoped-p
             (ir "(seq (loop 0 8 tile_n n dummy) (store (tensor A) (load (input B) (index (tile n))) (index fulltile)))")))
  (is-false (trinity-ggs::well-scoped-p
             (ir "(loop 0 8 tile_n n (loop 0 8 tile_n n (store (tensor A) (load (input B) (index (tile n))) (index fulltile))))"))))

;;; Cost model

(def-test fused-is-cheaper ()
  (let ((*shapes* (apply #'shapes *chain-shapes*)))
    (is (< (program-cost (postprocess (ir "(loop 0 1024 tile_n n (seq
               (store (tensor B) (exp (load (input A) (index fulltile (tile n)))) (index fulltile (tile n)))
               (store (output C) (sqrt (load (tensor B) (index fulltile (tile n)))) (index fulltile (tile n)))))")))
           (program-cost (postprocess (ir *elementwise-chain*)))))))

;;; End to end

(def-test vanilla-llama ()
  "In vanilla_llama, the QKV projection has a copy that drifted out of its loop
over n, which looks cheaper on its own. Extraction must still return a closed,
well-scoped program (the driver signals an error otherwise) that beats the
input."
  (multiple-value-bind (program cost stats) (optimize-instance "vanilla_llama" :output nil)
    (is-true program)
    (is (< cost (getf stats :baseline-us)))))
