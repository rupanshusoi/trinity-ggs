;;; Run every benchmark instance, then compare with Trinity's outputs.
;;;
;;;   sbcl --dynamic-space-size 16384 --non-interactive --load scripts/bench.lisp [name...]

(load (merge-pathnames "load.lisp" (directory-namestring *load-truename*)))

(in-package #:trinity-ggs)

(let* ((names (or (rest (member "--end-toplevel-options" sb-ext:*posix-argv* :test #'string=))
                  (mapcar (lambda (p) (getf p :name)) (read-manifest))))
       (stats (run-benchmarks :names names)))
  (format t "~%Comparison under the cost model (us):~%")
  (format t "~20a ~10a ~14a ~s~%" "instance" "ours" "trinity best" "trinity GPU picks")
  (loop for name in names
        for s in stats
        do (destructuring-bind (&key ours-us trinity-best-us trinity-gpu-winners &allow-other-keys)
               (cddr (compare-with-trinity name (getf s :best-us)))
             (format t "~20a ~10,1f ~14,1f ~{~{~a=~,1f~}~^ ~}~%"
                     name ours-us trinity-best-us
                     (mapcar (lambda (w) (list (subseq (car w) (1+ (search "_case" (car w))))
                                               (cdr w)))
                             trinity-gpu-winners)))))
