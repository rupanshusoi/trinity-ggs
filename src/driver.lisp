(in-package #:trinity-ggs)

;;; End-to-end pipeline: saturate, extract the cheapest program under the cost
;;; model, and postprocess it.

(defparameter *root*
  (asdf:system-source-directory '#:trinity-ggs))

(defun repo-path (relative) (merge-pathnames relative *root*))

(defun optimize-program (term shapes &key (iter-limit 8) time-limit node-limit verbose)
  "Optimize the program TERM, whose tensors have SHAPES (see `read-shapes').
Returns the optimized program, its estimated cost in microseconds, and a plist
of statistics."
  (let ((*shapes* shapes)
        (start (get-internal-real-time)))
    (flet ((seconds () (float (/ (- (get-internal-real-time) start)
                                 internal-time-units-per-second))))
      (multiple-value-bind (egraph root reason iterations)
          (saturate term :iter-limit iter-limit :time-limit time-limit
                         :node-limit node-limit :verbose verbose)
        (let* ((*egraph* egraph)
               (saturation-time (seconds))
               (program (postprocess (extract-best root)))
               (cost (program-cost program)))
          ;; Extraction only returns closed terms, and the cost model charges
          ;; repeated accumulations, so neither should happen.
          (unless (well-scoped-p program)
            (error "Extracted an ill-scoped program: ~a" (term-string program)))
          (when (repeats-accumulation-p program)
            (error "Extracted a program that repeats an accumulation: ~a"
                   (term-string program)))
          (values program cost
                  (list :stop reason :iterations iterations
                        :enodes (egraph-n-enodes egraph)
                        :eclasses (egraph-n-eclasses egraph)
                        :baseline-us (program-cost (postprocess term))
                        :best-us cost
                        :saturation-seconds saturation-time
                        :total-seconds (seconds))))))))

;;; Benchmarks

(defun read-manifest ()
  (with-open-file (s (repo-path "inputs/manifest.sexp"))
    (with-standard-io-syntax
      (let ((*package* (find-package :keyword)) (*read-eval* nil))
        (read s)))))

(defun find-instance (name)
  (or (find name (read-manifest) :key (lambda (p) (getf p :name)) :test #'string=)
      (error "No benchmark instance named ~s." name)))

(defun instance-inputs (instance)
  (let ((name (getf instance :name)))
    (values (read-ir (repo-path (format nil "inputs/~a.ir" name)))
            (read-shapes (repo-path (format nil "inputs/~a.shapes" name))))))

(defun result-path (name) (repo-path (format nil "results/~a.ir" name)))

(defun optimize-instance (name &rest args &key (output t) &allow-other-keys)
  "Optimize benchmark instance NAME (e.g. \"vanilla_llama\") with the iteration
limit Trinity used for it. Writes the program to results/NAME.ir when OUTPUT is
true. Returns the program, its cost and statistics."
  (let ((instance (find-instance name)))
    (multiple-value-bind (term shapes) (instance-inputs instance)
      (multiple-value-bind (program cost stats)
          (apply #'optimize-program term shapes
                 :iter-limit (getf instance :iter)
                 (remove-from-plist args :output))
        (when output
          (let ((path (result-path name)))
            (ensure-directories-exist path)
            (with-open-file (s path :direction :output :if-exists :supersede)
              (write-line (term-string program) s))))
        (values program cost (list* :name name stats))))))

(defun run-benchmarks (&rest args &key (names (mapcar (lambda (p) (getf p :name))
                                                      (read-manifest)))
                       &allow-other-keys)
  "Optimize each benchmark instance in NAMES and print a summary table. Returns
the statistics of each."
  (format t "~&~20a ~10a ~5a ~9a ~11a ~11a ~8a ~8a~%"
          "instance" "stop" "iters" "enodes" "input (us)" "best (us)" "speedup" "time (s)")
  (loop for name in names
        collect (let ((stats (nth-value 2 (apply #'optimize-instance name
                                                  (remove-from-plist args :names)))))
                  (destructuring-bind (&key stop iterations enodes baseline-us best-us
                                         total-seconds &allow-other-keys)
                      stats
                    (format t "~20a ~10a ~5d ~9d ~11,1f ~11,1f ~7,2fx ~8,1f~%"
                            name (string-downcase stop) iterations enodes
                            baseline-us best-us (/ baseline-us best-us) total-seconds)
                    (finish-output))
                  stats)))

;;; Comparison with Trinity's own outputs

(defparameter *trinity-dir* (merge-pathnames "../trinity/" *root*))

(defun placeholder-p (term)
  "Trinity marks failed extractions with dummydata<N> symbols."
  (if (consp term)
      (some #'placeholder-p term)
      (and (symbolp term) (search "DUMMYDATA" (symbol-name term)))))

(defun read-candidate-file (path)
  "Read Trinity's candidate list: one \"i: program\" per line."
  (with-open-file (s path)
    (loop for line = (read-line s nil)
          while line
          for colon = (position #\: line)
          for term = (and colon (read-ir-from-string (subseq line (1+ colon))))
          when (and (consp term) (not (placeholder-p term)))
            collect term)))

(defun score-programs (programs shapes)
  "Rank PROGRAMS (already postprocessed) by the cost model."
  (let ((*shapes* shapes))
    (sort (mapcar (lambda (p) (cons (program-cost p) p)) programs) #'< :key #'car)))

(defun trinity-candidate-file (instance)
  (let ((workload (getf instance :workload))
        (model (getf instance :model)))
    (find-if #'probe-file
             (list (merge-pathnames (format nil "backend/evaluation/~a/~a_~a_cost6_kern1.txt"
                                            workload workload model)
                                    *trinity-dir*)
                   (merge-pathnames (format nil "backend/evaluation/~a/~a_~a_cost6_kern5_wo_scheduler2.txt"
                                            workload model workload)
                                    *trinity-dir*)))))

(defun trinity-winner-files (instance)
  (directory (repo-path (format nil "reference/trinity-winners/~a_~a_case*.txt"
                                (getf instance :workload) (getf instance :model)))))

(defun compare-with-trinity (name &optional ours-us)
  "Compare our program for instance NAME with Trinity's, all under our cost
model: the best of Trinity's candidate list, and the programs Trinity picked by
GPU profiling (reference/trinity-winners/). OURS-US is our cost; by default it
is that of results/NAME.ir."
  (let* ((instance (find-instance name))
         (shapes (nth-value 1 (instance-inputs instance)))
         (ours (or ours-us
                   (when (probe-file (result-path name))
                     (car (first (score-programs (list (read-ir (result-path name))) shapes))))))
         (candidate-file (trinity-candidate-file instance))
         (trinity (when candidate-file
                    (score-programs (read-candidate-file candidate-file) shapes)))
         (winners (loop for f in (trinity-winner-files instance)
                        collect (cons (pathname-name f)
                                      (car (first (score-programs (list (read-ir f)) shapes)))))))
    (list :name name
          :ours-us ours
          :trinity-candidates (length trinity)
          :trinity-best-us (car (first trinity))
          :trinity-gpu-winners winners)))
