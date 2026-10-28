(in-package #:trinity-ggs)

;;; Equality saturation driver.
;;;
;;; `ggs:run-rewrites' has an egg-style backoff scheduler, but with a single
;;; match limit for all rules. Trinity tunes the limit per rule (see
;;; `*rule-schedule*'), so we run the same loop with per-rule settings. As in
;;; egg, a rule banned at iteration I with ban length B is skipped while the
;;; iteration is below I + B, so a ban length of 1 skips no iteration.

(defun run-schedule (rules &key (iter-limit 8) (default-match-limit 1000)
                             (default-ban-length 5) time-limit node-limit verbose)
  "Run RULES on `*egraph*' until saturation or a limit. Returns the stop reason
and the number of iterations run."
  (let ((start (get-internal-real-time))
        (banned-until (make-hash-table))
        (times-banned (make-hash-table))
        (n-enodes (egraph-n-enodes *egraph*))
        (n-eclasses (egraph-n-eclasses *egraph*)))
    (flet ((elapsed () (/ (- (get-internal-real-time) start)
                          internal-time-units-per-second)))
      (loop for iter from 0
            do (cond ((>= iter iter-limit) (return (values :iter-limit iter)))
                     ((and time-limit (>= (elapsed) time-limit))
                      (return (values :time-limit iter)))
                     ((and node-limit (>= n-enodes node-limit))
                      (return (values :node-limit iter))))
               (dolist (name rules)
                 (destructuring-bind (limit . ban-length)
                     (or (gethash name *rule-schedule*)
                         (cons default-match-limit default-ban-length))
                   (unless (< iter (gethash name banned-until -1))
                     (let ((times (gethash name times-banned 0)))
                       (handler-case
                           (dolist (rule (ggs/eqsat::compiled-rules name))
                             (funcall rule :match-limit (and limit (ash limit times))))
                         (ggs/eqsat::match-limit-exceeded ()
                           (when verbose
                             (format t "~&  ~(~a~) banned for ~a iterations~%"
                                     name (ash ban-length times)))
                           (setf (gethash name banned-until) (+ iter (ash ban-length times))
                                 (gethash name times-banned) (1+ times))))))))
               (egraph-rebuild)
               (let ((n-enodes-1 (egraph-n-enodes *egraph*))
                     (n-eclasses-1 (egraph-n-eclasses *egraph*)))
                 (when verbose
                   (format t "~&Iteration ~a: ~a enodes, ~a eclasses (~,1fs)~%"
                           iter n-enodes-1 n-eclasses-1 (elapsed)))
                 (when (and (= n-enodes n-enodes-1) (= n-eclasses n-eclasses-1)
                            (loop for until being the hash-values of banned-until
                                  never (> until (1+ iter))))
                   (return (values :saturated (1+ iter))))
                 (setf n-enodes n-enodes-1 n-eclasses n-eclasses-1))))))

(defun saturate (term &rest args &key (rules *rules*) &allow-other-keys)
  "Run equality saturation on TERM with `*shapes*' describing its tensors.

Returns the e-graph, the root enode, the stop reason and the number of
iterations. Bind `*egraph*' to the e-graph to inspect it. ARGS are passed to
`run-schedule'."
  (let ((*egraph* (make-egraph :analyses +analyses+))
        (*applier-memo* (make-hash-table :test 'equal))
        (*warn-runtime-compile* nil))
    (let ((root (term-enode term)))
      (egraph-rebuild)
      (multiple-value-bind (reason iterations)
          (apply #'run-schedule rules (remove-from-plist args :rules))
        (values *egraph* (enode-find root) reason iterations)))))
