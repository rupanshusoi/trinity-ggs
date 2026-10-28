(in-package #:trinity-ggs)

;;; Rewrite rules, ported from Trinity's rules.rs. Each rule is its own GGS
;;; rule set so that the runner can schedule rules individually, mirroring
;;; Trinity's per-rule BackoffScheduler settings.

(defvar *rules* '()
  "Names of all rules, in definition order.")

(defvar *rule-schedule* (make-hash-table)
  "Map from rule name to (MATCH-LIMIT . BAN-LENGTH). A NIL match limit means the
rule is never banned. Rules not listed use the runner's defaults.")

(defmacro defrule (name (&key (match-limit nil limit-p) (ban-length 5)) &body rule)
  "Define rule NAME. RULE is (LHS RHS [:guard CONDITION]) or (LHS :eval BODY...),
as in `defrw*'."
  `(progn
     (defrw* ,name ,rule)
     (precompile-eqsat-rule-set ,name)
     (setf *rules* (append (remove ',name *rules*) (list ',name)))
     ,(if limit-p
          `(setf (gethash ',name *rule-schedule*) (cons ,match-limit ,ban-length))
          `(remhash ',name *rule-schedule*))
     ',name))

;;; Appliers. Results are memoized because rules re-match every iteration and
;;; copying sub-e-graphs is not free.

(defvar *applier-memo* (make-hash-table :test 'equal))

(defmacro with-applier-memo ((&rest key-classes) &body body)
  (with-gensyms (key value found)
    `(let ((,key (list* ',(gensym) ,@(mapcar (lambda (c) `(enode-find ,c)) key-classes))))
       (multiple-value-bind (,value ,found) (gethash ,key *applier-memo*)
         (if ,found
             ,value
             (setf (gethash ,key *applier-memo*) (progn ,@body)))))))

(defun rename-tile (class from to)
  "Copy of CLASS with every (tile FROM) index component replaced by (elem TO).
Trinity instead unions (tile FROM) with (elem TO) globally while resolving a
`tmp_loop' and repairs the result during postprocessing."
  ;; As in `rewrite-class', nodes the renaming leaves unchanged are dropped:
  ;; they sit in a class that mentions (tile FROM) through other members.
  (let ((old `(tile ,from))
        (new `(elem ,to)))
    (flet ((rename (index-class)
             (let ((index (class-term index-class)))
               (if (term-mentions-p index old)
                   (term-enode (subst new old index :test #'equal))
                   index-class))))
      (rewrite-class
       class
       (lambda (c)
         (and (some (lambda (a) (term-mentions-p (access-index a) old)) (class-accesses c))
              ;; A nested loop over FROM binds its own occurrences, so its
              ;; class is shared unchanged.
              (notany (lambda (node) (and (eq (enode-fsym node) 'loop)
                                          (eq (class-atom (enode-arg 3 node)) from)))
                      (class-nodes c))))
       (lambda (node visit)
         (case (enode-fsym node)
           (load (let ((index (rename (enode-arg 1 node))))
                   (values (unless (eq index (enode-arg 1 node))
                             (make-enode 'load (enode-arg 0 node) index))
                           t)))
           (store (let ((val (funcall visit (enode-arg 1 node)))
                        (index (rename (enode-arg 2 node))))
                    (values (when (and val (not (and (eq val (enode-arg 1 node))
                                                     (eq index (enode-arg 2 node)))))
                              (make-enode 'store (enode-arg 0 node) val index))
                            t)))
           (t (values nil nil))))))))

(defun fuse-loops (start n m tile-1 tile-2 lv-1 lv-2 body-1 body-2)
  "Trinity's LoopFusion applier: fuse two adjacent loops into one, or return NIL.

Loops with the same variable and range fuse directly. A loop over N with
numeric step TILE-1 also fuses with a loop of N/TILE-1 iterations, whose body
then indexes the element (elem LV-1) instead of (tile LV-2)."
  (with-applier-memo (start n m tile-1 tile-2 lv-1 lv-2 body-1 body-2)
    (when (no-raw-dependency-p body-1 body-2 lv-1)
      (let ((num-1 (num-p tile-1))
            (num-2 (num-p tile-2)))
        (flet ((renamed-loop (end tile keep drop)
                 ;; The body of the kept loop must not refer to DROP: such
                 ;; references are bound outside and must not be renamed.
                 (when (no-loopvar-dependency-p (if (eq keep lv-1) body-1 body-2) drop)
                   (let ((b1 (rename-tile body-1 (class-atom drop) (class-atom keep)))
                         (b2 (rename-tile body-2 (class-atom drop) (class-atom keep))))
                     (when (and b1 b2)
                       (make-enode 'loop start end tile keep (make-enode 'seq b1 b2)))))))
          (cond ((and (same-class-p lv-1 lv-2) (same-class-p n m))
                 (make-enode 'loop start n (if (and (not num-1) num-2) tile-2 tile-1) lv-1
                             (make-enode 'seq body-1 body-2)))
                ((and num-1 (not num-2) (divides-into-p n tile-1 m))
                 (renamed-loop n tile-1 lv-1 lv-2))
                ((and (not num-1) num-2 (divides-into-p m tile-2 n))
                 (renamed-loop m tile-2 lv-2 lv-1))))))))

(defun forward-store (base val index rest)
  "Value forwarding, from Trinity's `modify' hook: in the statements REST that
follow (store BASE VAL INDEX), replace loads of BASE at INDEX by VAL. Loops in
REST are left alone, as are stores that accumulate into BASE."
  (with-applier-memo (base val index rest)
    (let ((symbol (class-base base)))
      (when (and symbol (not (reads-tensor-p val (split-names symbol))))
        (let* ((index-term (class-term index))
               (new (rewrite-class
                     rest
                     (lambda (c)
                       (and (some (lambda (a) (and (eq (access-base a) symbol)
                                                   (equal (access-index a) index-term)))
                                  (class-reads c))
                            ;; Loops are shared unchanged. Merging them into
                            ;; the rewritten class would equate REST with its
                            ;; forwarded version everywhere, not just after
                            ;; the store.
                            (not (class-has-p c 'loop))))
                     (lambda (node visit)
                       (declare (ignore visit))
                       (if (eq (enode-fsym node) 'load)
                           (values (when (and (eq (class-base (enode-arg 0 node)) symbol)
                                              (equal (class-term (enode-arg 1 node)) index-term))
                                     val)
                                   t)
                           (values nil nil))))))
          (unless (or (null new) (same-class-p new rest))
            new))))))

;;; Replacements for Trinity's `modify' hook. These were applied on every
;;; rebuild there, so they are never banned here.

(defrule seq-assoc (:match-limit nil)
  (seq (seq ?a ?b) ?c) (seq ?a (seq ?b ?c)))

(defrule value-forwarding (:match-limit nil)
  (seq (store ?base ?val ?index) ?rest)
  :eval (when-let ((?new (forward-store ?base ?val ?index ?rest)))
          (yield-rewrite (seq (store ?base ?val ?index) ?new))))

;;; Statement reordering

(defrule seq-comm-tail (:match-limit 5000 :ban-length 1)
  (seq ?a (seq ?b ?body)) (seq ?b (seq ?a ?body))
  :guard (and (no-seq-p ?a) (no-seq-p ?b) (no-all-dependency-p ?a ?b)))

(defrule seq-comm (:match-limit nil)
  (seq ?a ?b) (seq ?b ?a)
  :guard (and (no-seq-p ?a) (no-seq-p ?b) (no-all-dependency-p ?a ?b)))

;;; Loop transformations

(defrule loop-fusion (:match-limit nil)
  (seq (loop ?start ?n ?tile1 ?lv1 ?body1) (loop ?start ?m ?tile2 ?lv2 ?body2))
  :eval (when-let ((?fused (fuse-loops ?start ?n ?m ?tile1 ?tile2 ?lv1 ?lv2 ?body1 ?body2)))
          (yield-rewrite ?fused)))

(defrule loop-fusion-tail (:match-limit 5000 :ban-length 1)
  (seq (loop ?start ?n ?tile1 ?lv1 ?body1) (seq (loop ?start ?m ?tile2 ?lv2 ?body2) ?others))
  :eval (when-let ((?fused (fuse-loops ?start ?n ?m ?tile1 ?tile2 ?lv1 ?lv2 ?body1 ?body2)))
          (yield-rewrite (seq ?fused ?others))))

(defrule loop-fission-tail ()
  (seq (loop 0 ?n ?tile ?lv (seq ?body1 ?body2)) ?others)
  (seq (loop 0 ?n ?tile ?lv ?body1) (seq (loop 0 ?n ?tile ?lv ?body2) ?others))
  :guard (no-raw-dependency-p ?body1 ?body2 ?lv))

(defrule loop-fission ()
  (loop 0 ?n ?tile ?lv (seq ?body1 ?body2))
  (seq (loop 0 ?n ?tile ?lv ?body1) (loop 0 ?n ?tile ?lv ?body2))
  :guard (no-raw-dependency-p ?body1 ?body2 ?lv))

(defrule loop-insertion1-tail ()
  (seq (loop 0 ?n ?tile ?lv ?body1) (seq ?body2 ?others))
  (seq (loop 0 ?n ?tile ?lv (seq ?body1 ?body2)) ?others)
  :guard (and (no-loopvar-dependency-p ?body2 ?lv)
              (repeatable-p ?body2)
              (not-same-loop-p ?body2 ?n ?tile ?lv)
              (no-raw-dependency-p ?body1 ?body2 ?lv)))

(defrule loop-insertion1 ()
  (seq (loop 0 ?n ?tile ?lv ?body1) ?body2)
  (loop 0 ?n ?tile ?lv (seq ?body1 ?body2))
  :guard (and (no-loopvar-dependency-p ?body2 ?lv)
              (repeatable-p ?body2)
              (not-same-loop-p ?body2 ?n ?tile ?lv)
              (no-seq-p ?body2)
              (no-raw-dependency-p ?body1 ?body2 ?lv)))

(defrule loop-insertion2-tail ()
  (seq ?body1 (seq (loop 0 ?n ?tile ?lv ?body2) ?others))
  (seq (loop 0 ?n ?tile ?lv (seq ?body1 ?body2)) ?others)
  :guard (and (no-loopvar-dependency-p ?body1 ?lv)
              (repeatable-p ?body1)
              (not-same-loop-p ?body1 ?n ?tile ?lv)
              (no-raw-dependency-p ?body1 ?body2 ?lv)))

(defrule loop-insertion2 ()
  (seq ?body1 (loop 0 ?n ?tile ?lv ?body2))
  (loop 0 ?n ?tile ?lv (seq ?body1 ?body2))
  :guard (and (no-loopvar-dependency-p ?body1 ?lv)
              (repeatable-p ?body1)
              (not-same-loop-p ?body1 ?n ?tile ?lv)
              (no-raw-dependency-p ?body1 ?body2 ?lv)))

(defrule loop-deletion ()
  (loop 0 ?n ?tile ?lv ?body) ?body
  :guard (and (no-loopvar-dependency-p ?body ?lv) (repeatable-p ?body) (not-num-p ?tile)))

(defrule loop-comm-tail ()
  (seq (loop 0 ?n ?tile ?lv (seq (store ?a (+ (load ?a ?idx) ?val1) ?idx)
                                 (store ?b (+ (load ?b ?idx) ?val2) ?idx)))
       (seq (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx) ?others))
  (seq (loop 0 ?n ?tile ?lv (store ?c (+ (load ?c ?idx) (+ ?val1 ?val2)) ?idx)) ?others))

(defrule loop-comm ()
  (seq (loop 0 ?n ?tile ?lv (seq (store ?a (+ (load ?a ?idx) ?val1) ?idx)
                                 (store ?b (+ (load ?b ?idx) ?val2) ?idx)))
       (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx))
  (loop 0 ?n ?tile ?lv (store ?c (+ (load ?c ?idx) (+ ?val1 ?val2)) ?idx)))

(defrule loop-factor-div ()
  (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) (/ ?val1 ?val2)) ?idx))
  (seq (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) ?val1) ?idx))
       (store ?b (/ (load ?b ?idx) ?val2) ?idx))
  :guard (and (no-loopvar-dependency-p ?val2 ?lv) (not-one-p ?val2)))

(defrule loop-factor-div-tail ()
  (seq (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) (/ ?val1 ?val2)) ?idx)) ?others)
  (seq (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) ?val1) ?idx))
       (seq (store ?b (/ (load ?b ?idx) ?val2) ?idx) ?others))
  :guard (and (no-loopvar-dependency-p ?val2 ?lv) (not-one-p ?val2)))

(defrule loop-factor-mul ()
  (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) (* ?val1 ?val2)) ?idx))
  (seq (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) ?val1) ?idx))
       (store ?b (* (load ?b ?idx) ?val2) ?idx))
  :guard (and (no-loopvar-dependency-p ?val2 ?lv) (not-one-p ?val2)))

(defrule loop-factor-mul-tail ()
  (seq (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) (* ?val1 ?val2)) ?idx)) ?others)
  (seq (loop 0 ?n ?tile ?lv (store ?b (+ (load ?b ?idx) ?val1) ?idx))
       (seq (store ?b (* (load ?b ?idx) ?val2) ?idx) ?others))
  :guard (and (no-loopvar-dependency-p ?val2 ?lv) (not-one-p ?val2)))

;;; Algebraic rules

(defrule comm-add () (+ ?a ?b) (+ ?b ?a))
(defrule assoc-add () (+ ?a (+ ?b ?c)) (+ (+ ?a ?b) ?c))
(defrule assoc-add2 () (+ (+ ?a ?b) ?c) (+ ?a (+ ?b ?c)))
(defrule comm-mul () (* ?a ?b) (* ?b ?a))
(defrule assoc-mul () (* ?a (* ?b ?c)) (* (* ?a ?b) ?c))
(defrule assoc-mul2 () (* (* ?a ?b) ?c) (* ?a (* ?b ?c)))
(defrule assoc-matmul () (@ ?a (@ ?b ?c)) (@ (@ ?a ?b) ?c))
(defrule assoc-matmul2 () (@ (@ ?a ?b) ?c) (@ ?a (@ ?b ?c)))
(defrule assoc-div-matmul () (@ (/ ?a (bcast ?b ?axis)) ?c) (/ (@ ?a ?c) (bcast ?b ?axis)))
(defrule dist-mul-add () (* ?a (+ ?b ?c)) (+ (* ?a ?b) (* ?a ?c)))
(defrule dist-mul-sub () (* ?a (- ?b ?c)) (- (* ?a ?b) (* ?a ?c)))
(defrule dist-matmul-add () (@ ?a (+ ?b ?c)) (+ (@ ?a ?b) (@ ?a ?c)))
(defrule dist-matmul-sub () (@ ?a (- ?b ?c)) (- (@ ?a ?b) (@ ?a ?c)))
(defrule factor-mul-add () (+ (* ?a ?b) (* ?a ?c)) (* ?a (+ ?b ?c)))
(defrule factor-mul-sub () (- (* ?a ?b) (* ?a ?c)) (* ?a (- ?b ?c)))
(defrule factor-matmul-add () (+ (@ ?a ?b) (@ ?a ?c)) (@ ?a (+ ?b ?c)))
(defrule factor-matmul-sub () (- (@ ?a ?b) (@ ?a ?c)) (@ ?a (- ?b ?c)))
(defrule exp-mul () (* (exp ?a) (exp ?b)) (exp (+ ?a ?b)))
(defrule exp-div () (/ (exp ?a) (exp ?b)) (exp (- ?a ?b)))
(defrule exp0 () (exp 0) 1)
(defrule recip-mul-div () (* ?x (/ 1 ?x)) 1 :guard (not-zero-p ?x))
(defrule geometry-of-concat ()
  (concat (concat ?x ?z 0) (concat ?y ?w 0) 1) (concat (concat ?x ?y 1) (concat ?z ?w 1) 0))
(defrule geometry-of-concat-inv ()
  (concat (concat ?x ?y 1) (concat ?z ?w 1) 0) (concat (concat ?x ?z 0) (concat ?y ?w 0) 1))
(defrule operator-comm6 () (+ (concat ?x ?z ?a) (concat ?y ?w ?a)) (concat (+ ?x ?y) (+ ?z ?w) ?a))
(defrule operator-comm6-inv () (concat (+ ?x ?y) (+ ?z ?w) ?a) (+ (concat ?x ?z ?a) (concat ?y ?w ?a)))
(defrule operator-comm7 () (* (concat ?x ?z ?a) (concat ?y ?w ?a)) (concat (* ?x ?y) (* ?z ?w) ?a))
(defrule operator-comm7-inv () (concat (* ?x ?y) (* ?z ?w) ?a) (* (concat ?x ?z ?a) (concat ?y ?w ?a)))
(defrule concat-and-matmul0 () (@ ?x (concat ?y ?z 1)) (concat (@ ?x ?y) (@ ?x ?z) 1))
(defrule concat-and-matmul0-inv () (concat (@ ?x ?y) (@ ?x ?z) 1) (@ ?x (concat ?y ?z 1)))
(defrule concat-and-matmul1 () (+ (@ ?a ?b) (@ ?c ?d)) (@ (concat ?a ?c 1) (concat ?b ?d 0)))
(defrule concat-and-matmul1-inv () (@ (concat ?a ?c 1) (concat ?b ?d 0)) (+ (@ ?a ?b) (@ ?c ?d)))
