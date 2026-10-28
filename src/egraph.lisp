(in-package #:trinity-ggs)

;;; Helpers for inspecting and editing the e-graph from rule guards, appliers
;;; and extraction.
;;;
;;; While rules run, GGS matches against a snapshot of each eclass taken at the
;;; last `egraph-rebuild' (the eclass-info FSYM-INDEX). We read the same
;;; snapshot, so the analysis data of every class we look at is up to date.

(defun class-info (class)
  ;; A representative merged away since the last rebuild keeps its snapshot.
  (or (ggs/eqsat::enode-class-info class)
      (ggs/eqsat::enode-class-info (enode-find class))))

(defun class-nodes (class)
  "Enodes of CLASS as of the last `egraph-rebuild'."
  (let ((info (class-info class)))
    (or (loop for entry in (ggs/eqsat::eclass-info-fsym-index info)
              append (cdr entry))
        ;; Created since the last rebuild, so there is no snapshot yet.
        (ggs/eqsat::eclass-info-nodes info))))

(defun enode-args (node)
  (map-enode-args #'identity node))

(defun class-has-p (class fsym)
  (some (lambda (node) (eql (enode-fsym node) fsym)) (class-nodes class)))

(defun class-atom (class)
  "The symbol or number in CLASS, if any."
  (dolist (node (class-nodes class))
    (when (zerop (enode-n-args node))
      (return (enode-fsym node)))))

(defun class-number (class)
  (dolist (node (class-nodes class))
    (when (and (zerop (enode-n-args node)) (numberp (enode-fsym node)))
      (return (enode-fsym node)))))

(defun class-base (class)
  "The tensor symbol of a base class like (tensor X)."
  (dolist (node (class-nodes class))
    (when (and (member (enode-fsym node) '(input tensor output))
               (= (enode-n-args node) 1))
      (return (class-atom (enode-arg 0 node))))))

(defun class-term (class &optional (depth 16))
  "The term obtained by following the first node of every class. Only meant for
small classes with a single term, like indices."
  (when (plusp depth)
    (let ((node (first (class-nodes class))))
      (if (zerop (enode-n-args node))
          (enode-fsym node)
          (cons (enode-fsym node)
                (map-enode-args (lambda (c) (class-term c (1- depth))) node))))))

(defun same-class-p (a b)
  (eq (enode-find a) (enode-find b)))

;;; Context-sensitive rewriting of a sub-e-graph.
;;;
;;; Trinity performs value forwarding and loop-variable renaming inside its
;;; `modify' hook by unioning e-classes globally, then repairs the extracted
;;; terms in a postprocessing pass. GGS has no modify hook, and the global
;;; unions are only valid in some contexts anyway. Instead, appliers build a
;;; rewritten copy of the relevant part of the e-graph: classes that cannot be
;;; affected are shared, the rest are copied node by node.

(defun rebuild-node (node visit)
  "Copy NODE with each argument mapped through VISIT. Return NODE itself if no
argument changed, and NIL if some argument could not be rewritten."
  (let ((args (map-enode-args visit node)))
    (cond ((member nil args) nil)
          ((every #'eq args (enode-args node)) node)
          (t (apply #'make-enode (enode-fsym node) args)))))

(defun rewrite-class (class relevant-p node-fn)
  "Return a class for the rewritten version of CLASS, or NIL.

RELEVANT-P tells whether a class can be affected at all; irrelevant classes are
returned as is. NODE-FN is called with each enode and a visitor function; it
returns two values, the rewritten class and whether it handled the node. Nodes
it does not handle are copied with their arguments visited. Nodes on cycles are
dropped.

A relevant class may contain nodes the rewrite does not touch, equal to the
others only in context (e.g. after loop deletion). Those copies are dropped
rather than merging the rewritten class with the original one."
  (let ((memo (make-hash-table :test 'eq)))
    (labels ((visit (c)
               (multiple-value-bind (result found) (gethash c memo)
                 (cond ((not found)
                        (setf (gethash c memo) :visiting)
                        (setf (gethash c memo)
                              (if (funcall relevant-p c)
                                  (let ((result nil))
                                    (dolist (node (class-nodes c) result)
                                      (let ((new (multiple-value-bind (new handled)
                                                     (funcall node-fn node #'visit)
                                                   (if handled
                                                       new
                                                       (let ((copy (rebuild-node node #'visit)))
                                                         (unless (eq copy node) copy))))))
                                        (when new
                                          (if result
                                              (enode-merge result new)
                                              (setf result new))))))
                                  c)))
                       ((eq result :visiting) nil)
                       (t result)))))
      (visit class))))
