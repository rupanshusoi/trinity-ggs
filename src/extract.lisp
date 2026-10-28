(in-package #:trinity-ggs)

;;; Extraction with the cost model
;;;
;;; One bottom-up fixpoint, as in egg's Extractor, where each eclass keeps the
;;; cheapest term built from its children's current best terms. (Trinity
;;; instead enumerates candidates with a bound on sequential loops and kernels,
;;; then profiles each on a GPU.) The cost model is not a sum over nodes, so
;;; each candidate term is scored as a program fragment on its own:
;;;
;;; - a statement by `program-cost' after `decide-loop-types', with
;;;   intermediates written elsewhere read from memory. A closed statement's
;;;   top-level statements count as kernels; one with free loop variables sits
;;;   inside a loop, so it is scored as a single kernel;
;;; - a tensor expression as a kernel computing it once (its loads and FLOPs);
;;; - anything else (indices, tensor names, numbers) by its size alone.
;;;
;;; All terms of a class appear in the same contexts, so scoring them out of
;;; context still ranks them sensibly, but it is a heuristic: the cost of a
;;; fragment inside a loop or a kernel is not determined by its own cost.
;;;
;;; Classes keep one best term per set of free loop variables. Trinity's rules
;;; equate statements with copies that drifted out of the loop binding their
;;; index (see `well-scoped-p'), and out of context such a copy looks cheap: an
;;; access indexed by an unbound variable counts as a single tile. Keying by
;;; free variables only compares terms that need the same enclosing loops, and
;;; the root takes the term with none.

(defparameter *size-weight* 1e-6
  "Cost (us) per term node, which breaks ties in favour of smaller terms.")

(defun rebinds-loop-var-p (term &optional bound)
  (and (consp term)
       (if (loop-term-p term)
           (or (member (fifth term) bound)
               (rebinds-loop-var-p (sixth term) (cons (fifth term) bound)))
           (some (lambda (x) (rebinds-loop-var-p x bound)) (cdr term)))))

(defun fragment-cost (term kind free)
  "Cost model estimate (us) of TERM on its own, given its free loop variables
FREE. KIND is :statement, :expression or nil (for terms the cost model does
not see).

A statement with free variables is wrapped in a one-iteration loop, making it
a single kernel without changing its accesses or FLOPs. Its grid is then too
small, but that scales all terms of the class alike."
  (ecase kind
    ((nil) 0)
    (:expression (program-cost `(store (tensor |%value|) ,term (index))))
    (:statement (cond ((rebinds-loop-var-p term) most-positive-single-float)
                      (free (program-cost (decide-loop-types `(loop 0 1 1 |%context| ,term))))
                      (t (program-cost (decide-loop-types term)))))))

(defun class-kind (class)
  (cond ((some (lambda (n) (member (enode-fsym n) '(store seq loop dummy)))
               (class-nodes class))
         :statement)
        ((enode-analysis-data 'shape class) :expression)))

(defun term-symbols (term)
  (cond ((and term (symbolp term)) (list term))
        ((consp term) (mapcan #'term-symbols (cdr term)))))

(defun node-free-vars (fsym child-terms child-free)
  "Free loop variables of the term (FSYM . CHILD-TERMS), as a sorted list."
  (let ((free (reduce #'union child-free :initial-value '())))
    (case fsym
      ((tile elem) (setf free (union free (term-symbols (first child-terms)))))
      (loop (setf free (remove (fourth child-terms) free))))
    (sort (copy-list free) #'string< :key #'symbol-name)))

(defstruct (entry (:constructor make-entry (free cost size term)))
  "A class's best term with free loop variables FREE."
  free cost size term)

(defun map-choices (fn lists)
  "Call FN on each list taking one element from each of LISTS."
  (labels ((rec (lists acc)
             (if lists
                 (dolist (x (first lists)) (rec (rest lists) (cons x acc)))
                 (funcall fn (reverse acc)))))
    (rec lists '())))

(defun extract-best (root &key (max-passes 100))
  "The cheapest closed term of ROOT in `*egraph*' under `fragment-cost', its
cost, and the table mapping each class to its entries. A node is re-scored only when a child class changed since it was last
scored, and an entry is replaced only by a strictly cheaper term."
  (let ((best (make-hash-table :test 'eq))       ; class -> list of entries
        (changed-at (make-hash-table :test 'eq)) ; class -> stamp of last change
        (scored (make-hash-table :test 'eq))     ; node -> stamp when last scored
        (kinds (make-hash-table :test 'eq))
        (stamp 0)
        (classes (ggs/eqsat::egraph-class-list *egraph*)))
    (flet ((offer (class entry)
             (let ((old (find (entry-free entry) (gethash class best)
                              :key #'entry-free :test #'equal)))
               (cond ((null old) (push entry (gethash class best)))
                     ((< (entry-cost entry) (* (entry-cost old) (- 1 1e-9)))
                      (setf (gethash class best)
                            (substitute entry old (gethash class best))))
                     (t (return-from offer nil)))
               (setf (gethash class changed-at) (incf stamp))
               t)))
      (dotimes (pass max-passes)
        (let ((changed nil))
          (dolist (class classes)
            (let ((kind (ensure-gethash class kinds (class-kind class))))
              (dolist (node (class-nodes class))
                (let ((children (enode-args node))
                      (last (gethash node scored)))
                  (when (and (every (lambda (c) (gethash c best)) children)
                             (or (null last)
                                 (some (lambda (c) (> (gethash c changed-at) last)) children)))
                    (setf (gethash node scored) stamp)
                    (map-choices
                     (lambda (entries)
                       (let* ((child-terms (mapcar #'entry-term entries))
                              (term (if children
                                        (cons (enode-fsym node) child-terms)
                                        (enode-fsym node)))
                              (size (1+ (reduce #'+ entries :key #'entry-size)))
                              (free (node-free-vars (enode-fsym node) child-terms
                                                    (mapcar #'entry-free entries)))
                              (cost (+ (fragment-cost term kind free) (* *size-weight* size))))
                         (when (offer class (make-entry free cost size term))
                           (setf changed t))))
                     (mapcar (lambda (c) (gethash c best)) children)))))))
          (unless changed (return)))))
    (let ((closed (find nil (gethash (enode-find root) best) :key #'entry-free)))
      (values (and closed (entry-term closed)) (and closed (entry-cost closed)) best))))
