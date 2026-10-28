(in-package #:trinity-ggs)

;;; Postprocessing of extracted programs, ported from Trinity's postprocess.rs:
;;; value forwarding inside loop bodies, dead store elimination, and deciding
;;; which loops are parallel (ploop) and which sequential (sloop).
;;;
;;; Trinity's "loop variable coherence" repair is not needed: we rename loop
;;; variables during fusion instead of unioning them globally.

(defun build-seq (statements)
  (reduce (lambda (s rest) (list 'seq s rest)) statements :from-end t))

(defun store-parts (statement)
  "Base, value and index of a STORE statement."
  (values (second statement) (third statement) (fourth statement)))

(defun store-p (term) (and (consp term) (eq (car term) 'store)))

(defun count-subterm (term subterm)
  (cond ((equal term subterm) 1)
        ((consp term) (reduce #'+ (cdr term) :key (lambda (x) (count-subterm x subterm))))
        (t 0)))

(defun forward-in-body (body)
  "Forward each non-accumulating store in BODY's statement list into later
stores, when exactly one later load reads it (`value_forward_substitution')."
  (let ((statements (coerce (flatten-seq body) 'vector)))
    (loop for i from 0 below (length statements)
          for statement = (aref statements i)
          when (store-p statement)
            do (multiple-value-bind (base val index) (store-parts statement)
                 (let ((load `(load ,base ,index)))
                   (unless (plusp (count-subterm val load))
                     (let ((users (loop for j from (1+ i) below (length statements)
                                        for later = (aref statements j)
                                        when (store-p later)
                                          sum (count-subterm (third later) load))))
                       (when (= users 1)
                         (loop for j from (1+ i) below (length statements)
                               for later = (aref statements j)
                               when (store-p later)
                                 do (setf (aref statements j)
                                          (subst val load later :test #'equal)))))))))
    (build-seq (coerce statements 'list))))

(defun forward-values (term)
  (if (atom term)
      term
      (let ((term (cons (car term) (mapcar #'forward-values (cdr term)))))
        (if (eq (car term) 'loop)
            (append (subseq term 0 5) (list (forward-in-body (sixth term))))
            term))))

(defun load-bases (term)
  "Names of all tensors loaded in TERM."
  (cond ((atom term) '())
        ((eq (car term) 'load)
         (let ((base (second term)))
           (when (consp base) (split-names (second base)))))
        (t (mapcan #'load-bases (cdr term)))))

(defun eliminate-dead-stores (term)
  "Replace stores to tensors that are never loaded, other than outputs, by DUMMY."
  (let ((read (remove-duplicates (load-bases term) :test #'string=)))
    (labels ((walk (term)
               (cond ((atom term) term)
                     ((and (store-p term)
                           (consp (second term))
                           (not (eq (first (second term)) 'output))
                           (not (names-overlap-p (split-names (second (second term))) read)))
                      'dummy)
                     (t (cons (car term) (mapcar #'walk (cdr term)))))))
      (walk term))))

(defun collect-stores (term)
  (cond ((atom term) '())
        ((store-p term) (list term))
        (t (mapcan #'collect-stores (cdr term)))))

(defun loop-carried-term-p (body loop-var)
  "Trinity's `has_loop_carried_dependency' on terms: some store writes the same
location on every iteration from values that vary with LOOP-VAR."
  (some (lambda (store)
          (multiple-value-bind (base val index) (store-parts store)
            (declare (ignore base))
            (and (not (index-depends-on-p index loop-var))
                 (term-mentions-p val loop-var))))
        (collect-stores body)))

(defun contains-seq-p (term)
  (and (consp term)
       (or (eq (car term) 'seq) (some #'contains-seq-p (cdr term)))))

(defun decide-loop-types (term &optional force-sequential)
  "Turn each LOOP into PLOOP or SLOOP. A loop is sequential if it carries a
dependency, or if an enclosing loop's body is a sequence of statements."
  (cond ((atom term) term)
        ((eq (car term) 'loop)
         (destructuring-bind (start end tile var body) (cdr term)
           (list (if (or force-sequential (loop-carried-term-p body var)) 'sloop 'ploop)
                 start end tile var
                 (decide-loop-types body (contains-seq-p body)))))
        (t (cons (car term)
                 (mapcar (lambda (x) (decide-loop-types x force-sequential)) (cdr term))))))

(defun well-scoped-p (term &optional bound)
  "Whether every (tile v) and (elem v) in TERM is bound by an enclosing loop over
v, and no loop rebinds the variable of an enclosing loop.

Trinity's rules check read/write sets but not scoping, so in combination they
can equate a program with one where a statement has drifted out of the loop
binding its index. `extract-best' only returns closed terms; this checks the
result."
  (cond ((atom term) t)
        ((loop-term-p term)
         (let ((var (fifth term)))
           (and (not (member var bound))
                (well-scoped-p (sixth term) (cons var bound)))))
        ((member (car term) '(tile elem))
         (member (second term) bound))
        (t (every (lambda (x) (well-scoped-p x bound)) (cdr term)))))

(defun repeats-accumulation-p (term &optional sequential-vars)
  "Whether some accumulation (a store reading its own tensor) in TERM sits in a
sequential loop whose variable neither its index nor its value mentions, so it
accumulates the same contribution on every iteration. `repeatable-p' lets loop
insertion produce these, expecting a parallel loop."
  (cond ((atom term) nil)
        ((loop-term-p term)
         (repeats-accumulation-p (sixth term)
                                 (if (eq (car term) 'ploop)
                                     sequential-vars
                                     (cons (fifth term) sequential-vars))))
        ((store-p term)
         (multiple-value-bind (base val index) (store-parts term)
           (and (consp base)
                (names-overlap-p (split-names (second base)) (load-bases val))
                (some (lambda (v)
                        (not (or (index-depends-on-p index v) (term-mentions-p val v))))
                      sequential-vars))))
        (t (some (lambda (x) (repeats-accumulation-p x sequential-vars)) (cdr term)))))

(defun postprocess (term)
  (dotimes (i 5) (setf term (forward-values term)))
  (dotimes (i 3) (setf term (eliminate-dead-stores term)))
  (decide-loop-types term))
