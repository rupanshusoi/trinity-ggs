(in-package #:trinity-ggs)

;;; E-class analyses
;;;
;;; ACC collects the memory accesses (loads and stores) reachable from a class,
;;; as a list (READS WRITES ACCUMULATIONS), where ACCUMULATIONS are the writes
;;; of stores whose value reads the tensor they write. Like Trinity's
;;; `collect_access_sets', it is the union over every node in the class, which
;;; over-approximates the accesses of any one term and is therefore
;;; conservative for the dependency checks below.
;;;
;;; SHAPE is the tile shape of a tensor expression, computed with Trinity's
;;; default tile size for symbolic tiles.

(defstruct (access (:constructor make-access
                       (base index private &aux (names (split-names base)))))
  "A load or store of tensor BASE (a symbol, possibly naming several fused
tensors) at INDEX, an (index ...) term. PRIVATE is true for intermediates
(tensor X), which a parallel loop can keep private to each iteration."
  base names index private)

(defun access= (a b)
  (and (eq (access-base a) (access-base b))
       (equal (access-index a) (access-index b))))

(defun accesses-overlap-p (a b)
  (names-overlap-p (access-names a) (access-names b)))

(defun union-accesses (to from)
  "Union of access lists, returning TO itself when FROM adds nothing."
  (let ((new (remove-duplicates
              (remove-if (lambda (a) (member a to :test #'access=)) from)
              :test #'access=)))
    (if new (append to new) to)))

(defun merge-acc (to from)
  "Merge (READS WRITES ACCUMULATIONS) lists. Returns TO when unchanged, which
tells GGS not to propagate."
  (cond ((null from) to)
        ((null to) from)
        (t (let ((merged (mapcar #'union-accesses to from)))
             (if (every #'eq merged to) to merged)))))

(defun children-acc (node)
  (let ((result nil))
    (do-enode-args (arg node result)
      (setf result (merge-acc result (enode-analysis-data 'acc arg))))))

(defun access-of (base index)
  (when-let ((symbol (class-base base)))
    (list (make-access symbol (class-term index) (class-has-p base 'tensor)))))

(defun store-acc (base val index)
  (let ((write (access-of base index)))
    (list nil write
          (when (any-overlap-p (first (enode-analysis-data 'acc val)) write)
            write))))

(define-analysis (acc :merge (lambda (to from) (merge-acc to from)) :bottom nil)
  (?node (children-acc ?node))
  ((load ?base ?index) (list (access-of ?base ?index) nil nil))
  ((store ?base ?val ?index) (store-acc ?base ?val ?index)))

(defun enode-shape (node)
  (let ((fsym (enode-fsym node)))
    (cond ((eq fsym 'load)
           (when-let* ((base (class-base (enode-arg 0 node)))
                       (dims (base-dims base)))
             (index-dims (class-term (enode-arg 1 node)) dims)))
          ((plusp (enode-n-args node))
           (op-shape fsym
                     (map-enode-args (lambda (c) (enode-analysis-data 'shape c)) node)
                     (map-enode-args #'class-number node))))))

(defun merge-shape (a b) (or a b))

(define-analysis (shape :merge (lambda (a b) (merge-shape a b)) :bottom nil)
  (?node (enode-shape ?node)))

(defparameter +analyses+ '(acc shape))

;;; Dependency predicates used as rule guards, ported from Trinity's
;;; dependency.rs and utils.rs.

(defun class-reads (class) (first (enode-analysis-data 'acc class)))
(defun class-writes (class) (second (enode-analysis-data 'acc class)))
(defun class-accumulations (class) (third (enode-analysis-data 'acc class)))

(defun class-accesses (class)
  (let ((acc (enode-analysis-data 'acc class)))
    (append (first acc) (second acc))))

(defun any-overlap-p (accesses-1 accesses-2)
  (some (lambda (a) (some (lambda (b) (accesses-overlap-p a b)) accesses-2))
        accesses-1))

(defun reads-tensor-p (class names)
  (some (lambda (a) (names-overlap-p (access-names a) names)) (class-reads class)))

(defun index-terms-equal-p (a b)
  "Trinity's deliberately lax `index_expressions_equal': indices only differ
when the same kind of component disagrees, e.g. (tile n) vs (tile m)."
  (let ((fa (term-fsym a)) (fb (term-fsym b)))
    (cond ((or (eq fa 'const_tile) (eq fb 'const_tile)) t)
          ((and (numberp a) (numberp b)) (= a b))
          ((and (symbolp a) (symbolp b))
           (or (eq a 'fulltile) (eq b 'fulltile) (eq a b)))
          ((and (consp a) (consp b) (eq fa fb) (member fa '(index tile elem + - * /)))
           (and (= (length a) (length b))
                (every #'index-terms-equal-p (cdr a) (cdr b))))
          (t t))))

(defun no-all-dependency-p (a b)
  "A and B touch disjoint tensors, apart from both reading the same ones."
  (let ((r1 (class-reads a)) (w1 (class-writes a))
        (r2 (class-reads b)) (w2 (class-writes b)))
    (not (or (any-overlap-p r2 w1) (any-overlap-p w2 r1) (any-overlap-p w2 w1)))))

(defun repeatable-p (body)
  "Whether BODY may run on every iteration of a loop instead of once (or vice
versa) when it does not depend on the loop variable. That holds unless it
accumulates, like (store a (+ (load a i) x) i), into an input or output.
Accumulations into intermediates are fine in a parallel loop, which keeps them
private and recomputes them per iteration, as in Trinity's prenorm kernels;
sequential loops are checked on the extracted program
\(`repeats-accumulation-p'), since loop types are only decided then.

Not in Trinity, whose loop insertion and deletion can change how often any
accumulation runs."
  (every #'access-private (class-accumulations body)))

(defun cross-iteration-p (write reads loop-var)
  "Trinity's `has_cross_iteration_dependency': WRITE does not vary with
LOOP-VAR (so it accumulates across iterations) while some read does."
  (and (not (index-depends-on-p (access-index write) loop-var))
       (some (lambda (r) (index-depends-on-p (access-index r) loop-var)) reads)))

(defun no-raw-dependency-p (body-1 body-2 loop-var)
  "BODY-2 may run in the same LOOP-VAR iteration as BODY-1: every value it reads
from BODY-1 is written at the same index and is complete by then."
  (let ((lv (class-atom loop-var))
        (r1 (class-reads body-1))
        (w1 (class-writes body-1)))
    (loop for r in (class-reads body-2)
          never (loop for w in w1
                      thereis (and (accesses-overlap-p r w)
                                   (or (not (index-terms-equal-p (access-index r)
                                                                 (access-index w)))
                                       (cross-iteration-p w r1 lv)))))))

(defun no-loopvar-dependency-p (body loop-var)
  (let ((lv (class-atom loop-var)))
    (notany (lambda (a) (index-depends-on-p (access-index a) lv))
            (class-accesses body))))

(defun no-seq-p (class) (not (class-has-p class 'seq)))
(defun num-p (class) (class-number class))
(defun not-num-p (class) (not (class-number class)))
(defun not-one-p (class) (not (class-has-p class 1)))
(defun not-zero-p (class) (not (class-has-p class 0)))

(defun divides-into-p (n tile m)
  "Trinity's `cond1': a loop over N with numeric step TILE has M iterations."
  (let ((n (class-number n)) (tile (class-number tile)) (m (class-number m)))
    (and (integerp n) (integerp tile) (integerp m) (/= tile 0)
         (= (truncate n tile) m))))

(defun not-same-loop-p (body end tile loop-var)
  "BODY is not itself a loop with the given header."
  (notany (lambda (node)
            (and (eq (enode-fsym node) 'loop)
                 (= (enode-n-args node) 5)
                 (same-class-p (enode-arg 1 node) end)
                 (same-class-p (enode-arg 2 node) tile)
                 (same-class-p (enode-arg 3 node) loop-var)))
          (class-nodes body)))
