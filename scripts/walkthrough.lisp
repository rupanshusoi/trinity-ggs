;;; Walk through optimizing one small program, printing each stage.
;;;
;;;   sbcl --non-interactive --load scripts/walkthrough.lisp [examples/chain]
;;;
;;; The argument names an .ir and a .shapes file. The optimized program is
;;; written to results/walkthrough.ir for scripts/check_equivalence.py.

(load (merge-pathnames "load.lisp" (directory-namestring *load-truename*)))

(in-package #:trinity-ggs)

(defun banner (title) (format t "~2&=== ~a~%~%" title))

(defun show-cost (program)
  (multiple-value-bind (seconds kernels) (cost-breakdown program)
    (loop for k in kernels
          for i from 0
          do (format t "  kernel ~a: ~,2f us = launch ~,1f us + max(~d bytes / BW, ~d FLOPs / peak) / util ~,2f  (grid ~a, tile ~a)~%"
                     i (* 1e6 (kernel-stats-time k)) (* 1e6 (hw :launch-overhead))
                     (round (kernel-stats-bytes k))
                     (round (+ (kernel-stats-tensor-core-flops k) (kernel-stats-simt-flops k)))
                     (min 1 (/ (kernel-stats-grid k) (hw :num-sms)))
                     (kernel-stats-grid k) (kernel-stats-tile k)))
    (format t "  total: ~,2f us~%" (* 1e6 seconds))))

(defun traced-saturation (term iter-limit)
  "Saturate like `saturate', but rebuild after every rule to report what each
one added. Returns the e-graph."
  (let ((*egraph* (make-egraph :analyses +analyses+))
        (*applier-memo* (make-hash-table :test 'equal))
        (*warn-runtime-compile* nil))
    (term-enode term)
    (egraph-rebuild)
    (format t "start: ~a enodes, ~a eclasses~%" (egraph-n-enodes *egraph*) (egraph-n-eclasses *egraph*))
    (dotimes (iter iter-limit)
      (let ((fired nil))
        (dolist (name *rules*)
          (let ((n (egraph-n-enodes *egraph*)) (c (egraph-n-eclasses *egraph*)))
            (dolist (rule (ggs/eqsat::compiled-rules name))
              (funcall rule :match-limit nil))
            (egraph-rebuild)
            (unless (and (= n (egraph-n-enodes *egraph*)) (= c (egraph-n-eclasses *egraph*)))
              (push (format nil "~(~a~) (~@d enodes, ~@d eclasses)" name
                            (- (egraph-n-enodes *egraph*) n) (- (egraph-n-eclasses *egraph*) c))
                    fired))))
        (format t "iteration ~a: ~:[nothing changed, saturated~;~:*~{~a~^, ~}~]~%"
                iter (reverse fired))
        (unless fired (return))))
    (format t "end: ~a enodes, ~a eclasses~%" (egraph-n-enodes *egraph*) (egraph-n-eclasses *egraph*))
    *egraph*))

(defun node-candidates (node table)
  "Every term `extract-best' builds for NODE from the entries of its children,
as (FREE-VARS COST TERM)."
  (let ((results '()))
    (map-choices
     (lambda (entries)
       (let* ((child-terms (mapcar #'entry-term entries))
              (term (if entries (cons (enode-fsym node) child-terms) (enode-fsym node)))
              (size (1+ (reduce #'+ entries :key #'entry-size))))
         (let ((free (node-free-vars (enode-fsym node) child-terms (mapcar #'entry-free entries))))
           (push (list free
                       (+ (fragment-cost term (class-kind (enode-find node)) free)
                          (* *size-weight* size))
                       term)
                 results))))
     (map-enode-args (lambda (c) (gethash c table)) node))
    (nreverse results)))

(defun show-class (class table title)
  "Print the terms built from each node of CLASS, scored now that extraction
has finished, and the entries extraction kept."
  (format t "~a: ~a nodes~%" title (length (class-nodes class)))
  (dolist (node (class-nodes class))
    (dolist (c (node-candidates node table))
      (destructuring-bind (free cost term) c
        (format t "  ~9,3f us  free ~(~a~)  ~a~%" cost free (term-string term)))))
  (format t "entries kept:~%")
  (dolist (e (gethash class table))
    (format t "  ~9,3f us  free ~(~a~)  ~a~%"
            (entry-cost e) (entry-free e) (term-string (entry-term e)))))

(defun body-class (loop-class)
  "The body class of the first LOOP node in LOOP-CLASS."
  (let ((node (find 'loop (class-nodes loop-class) :key #'enode-fsym)))
    (and node (enode-arg 4 node))))

(let* ((stem (or (second (member "--end-toplevel-options" sb-ext:*posix-argv* :test #'string=))
                 "examples/chain"))
       (term (read-ir (repo-path (format nil "~a.ir" stem))))
       (*shapes* (read-shapes (repo-path (format nil "~a.shapes" stem)))))
  (banner "Input program")
  (format t "~a~%~%" (term-string term))
  (show-cost (postprocess term))

  (banner "Saturation, rule by rule (rebuilding after each rule)")
  (traced-saturation term 8)

  (banner "Saturation as the optimizer runs it")
  (multiple-value-bind (egraph root reason iterations) (saturate term :iter-limit 8)
    (let ((*egraph* egraph))
      (format t "~(~a~) after ~a iterations: ~a enodes, ~a eclasses~%"
              reason iterations (egraph-n-enodes egraph) (egraph-n-eclasses egraph))

      (banner "Extraction")
      (multiple-value-bind (best cost table) (extract-best root)
        (show-class (enode-find root) table "Root class")
        (format t "~%")
        (when-let ((body (body-class (enode-find root))))
          (show-class (enode-find body) table "Body class of the root's loop")
          (format t "~%"))
        (format t "extracted (~,3f us before postprocessing):~%  ~a~%" cost (term-string best))

        (banner "Postprocessing: forward values, remove dead stores, choose loop types")
        (let ((program (postprocess best)))
          (format t "~a~%~%" (term-string program))
          (show-cost program)
          (with-open-file (out (repo-path "results/walkthrough.ir") :direction :output
                                                                    :if-exists :supersede
                                                                    :if-does-not-exist :create)
            (write-string (term-string program) out))
          (format t "~%written to results/walkthrough.ir~%"))))))
