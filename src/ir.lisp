(in-package #:trinity-ggs)

;;; Trinity's IR is an s-expression language. Two details need care when
;;; reading it with the Lisp reader:
;;;
;;; - Horizontally fused tensors are named like Q1,K1,V1, so the comma must be
;;;   a constituent character.
;;;
;;; - Names are case sensitive (C_exp, K_cache, X vs x). An :INVERT readtable
;;;   keeps them distinct while the all-lowercase operator names (loop, seq,
;;;   load, ...) still read as the ordinary upcased symbols used in our rules.

(defvar *ir-readtable*
  (let ((rt (copy-readtable nil)))
    (set-syntax-from-char #\, #\a rt)
    (setf (readtable-case rt) :invert)
    rt))

(defun normalize-term (term)
  "Write nullary operators like (fulltile) as atoms, as the e-graph does."
  (cond ((atom term) term)
        ((null (cdr term)) (car term))
        (t (cons (car term) (mapcar #'normalize-term (cdr term))))))

(defun read-ir-from-stream (stream &optional (eof-error-p t))
  (with-standard-io-syntax
    (let ((*readtable* *ir-readtable*)
          (*package* (find-package '#:trinity-ggs))
          (*read-eval* nil)
          (*read-default-float-format* 'double-float))
      (normalize-term (read stream eof-error-p nil)))))

(defun read-ir-from-string (string)
  (with-input-from-string (s string)
    (read-ir-from-stream s)))

(defun read-ir (path)
  "Read the IR term stored in PATH."
  (with-open-file (s path)
    (read-ir-from-stream s)))

(defun invert-case (string)
  "The case transformation of an :INVERT readtable. It is an involution, so it
maps external names to symbol names and back."
  (cond ((notany #'lower-case-p string) (string-downcase string))
        ((notany #'upper-case-p string) (string-upcase string))
        (t string)))

;;; Printing, in the syntax Trinity's own tools (and backend) accept.

(defun atom-string (x)
  (if (symbolp x)
      (invert-case (symbol-name x))
      (let ((*read-default-float-format* 'double-float))
        (princ-to-string x))))

(defun write-term (term stream)
  (if (consp term)
      (progn (write-char #\( stream)
             (loop for (x . rest) on term
                   do (write-term x stream)
                      (when rest (write-char #\Space stream)))
             (write-char #\) stream))
      (write-string (atom-string term) stream)))

(defun term-string (term)
  (with-output-to-string (s) (write-term term s)))

;;; Tensor shapes

(defvar *shapes* (make-hash-table :test 'equal)
  "Map from tensor name (as a symbol name) to its list of dimensions.")

(defun read-shapes (path)
  "Read a shapes file of the form ((\"name\" dim...) ...) into a hash table."
  (let ((table (make-hash-table :test 'equal)))
    (with-open-file (s path)
      (with-standard-io-syntax
        (let ((*read-eval* nil))
          (loop for (name . dims) in (read s)
                do (setf (gethash (invert-case name) table) dims)))))
    table))

(defun split-names (symbol)
  "Symbol names of the tensors fused into base SYMBOL, e.g. Q1,K1,V1. The
reader inverts case per token, so C,C_perturb keeps its C while a lone C reads
as c: split the external spelling and convert each part."
  (mapcar #'invert-case
          (uiop:split-string (invert-case (symbol-name symbol)) :separator ",")))

(defun names-overlap-p (names-1 names-2)
  (intersection names-1 names-2 :test #'string=))

(defun base-dims (symbol)
  "Shape of tensor base SYMBOL. Fused tensors share the shape of their first
member, as in Trinity."
  (values (gethash (first (split-names symbol)) *shapes*)))

;;; Term utilities

(defun term-fsym (term) (if (consp term) (car term) term))

(defun term-args (term) (if (consp term) (cdr term) nil))

(defun term-mentions-p (term subterm)
  (or (equal term subterm)
      (and (consp term)
           (some (lambda (arg) (term-mentions-p arg subterm)) (cdr term)))))

(defun flatten-seq (term)
  "Statements of a (right- or left-nested) SEQ term."
  (if (and (consp term) (eq (car term) 'seq))
      (append (flatten-seq (second term)) (flatten-seq (third term)))
      (list term)))

(defun index-depends-on-p (index loop-var)
  "Whether INDEX, an (index ...) term, depends on LOOP-VAR. Only TILE and ELEM
components refer to loop variables."
  (some (lambda (component)
          (and (consp component)
               (member (car component) '(tile elem))
               (term-mentions-p (second component) loop-var)))
        (term-args index)))

(defun loop-term-p (term)
  (and (consp term) (member (car term) '(loop sloop ploop))))
