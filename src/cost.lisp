(in-package #:trinity-ggs)

;;; Analytical cost model, standing in for Trinity's GPU profiling.
;;;
;;; A program is a sequence of kernels (its top-level statements). Each kernel
;;; is estimated with a roofline:
;;;
;;;   time = launch + max(bytes / bandwidth, matmul flops / tensor-core peak
;;;                                          + other flops / SIMT peak) / util
;;;
;;; - Bytes count global-memory traffic. Inputs and outputs always live in
;;;   global memory. An intermediate tensor stays on chip when it is used by a
;;;   single kernel and every access sees the same tiles, i.e. the same set of
;;;   enclosing sequential loops whose variable appears in its index. A tensor
;;;   produced tile by tile in one sequential loop and consumed in another must
;;;   be materialized in memory, which matches when Trinity's code generator
;;;   allocates "cross-sloop memory tensors".
;;;
;;; - An access is repeated once per iteration of every enclosing parallel loop
;;;   (each program fetches its own data) and of every enclosing sequential loop
;;;   whose variable it is indexed by. Loop-invariant loads inside a sequential
;;;   loop are assumed to be cached.
;;;
;;; - FLOPs follow Trinity's per-tile counts, multiplied by the trip counts of
;;;   all enclosing loops, so recomputation is charged.
;;;
;;; - util = min(1, grid size / number of SMs), where the grid is the product
;;;   of the kernel's outer parallel loops. This penalizes fusing everything
;;;   into a kernel with too little parallelism.
;;;
;;; - Symbolic tile sizes (tile_n, ...) are chosen per kernel from
;;;   `*tile-choices*', standing in for Triton's autotuning: smaller tiles give
;;;   more parallelism but make every program re-read loop-invariant data.

(defparameter *hardware*
  (list :bandwidth 3.35e12            ; bytes/s, H100 SXM HBM3
        :tensor-core-flops 989e12     ; dense FP16 tensor-core peak
        :simt-flops 67e12             ; FP32 non-tensor peak
        :launch-overhead 3e-6         ; seconds per kernel launch
        :num-sms 132
        :element-bytes 2)             ; FP16
  "Machine parameters for the cost model, defaulting to an NVIDIA H100 SXM.")

(defun hw (key) (getf *hardware* key))

(defstruct frame
  "An enclosing loop: its variable, tile size, trip count and kind."
  var tile trips parallel-p)

(defstruct (mem-access (:conc-name ma-))
  kind                                  ; :read or :write
  base                                  ; e.g. (tensor X)
  index
  frames                                ; enclosing loops, innermost first
  kernel
  bytes)                                ; traffic if it goes to global memory

(defparameter *tile-choices* '(16 32 64 128)
  "Sizes tried for symbolic tiles, one size per kernel.")

(defstruct kernel-stats
  (bytes 0) (tensor-core-flops 0) (simt-flops 0) (grid 1) (tile nil) (time 0))

(defun frame-tile-size-fn (frames)
  (lambda (var)
    (let ((frame (find var frames :key #'frame-var)))
      (if frame (frame-tile frame) *default-tile*))))

(defun access-bytes (base index frames)
  (let ((dims (base-dims (second base))))
    (if (null dims)
        0
        (* (numel (index-dims index dims (frame-tile-size-fn frames)))
           (hw :element-bytes)
           (reduce #'* frames
                   :key (lambda (f)
                          (if (or (frame-parallel-p f)
                                  (index-depends-on-p index (frame-var f)))
                              (frame-trips f)
                              1)))))))

(defun walk-kernel (term kernel stats accesses)
  "Walk the kernel TERM, adding FLOPs to STATS and pushing memory accesses onto
the list in the ACCESSES cons."
  (labels ((repeat (frames) (reduce #'* frames :key #'frame-trips))
           (record (kind base index frames)
             (push (make-mem-access :kind kind :base base :index index :frames frames
                                    :kernel kernel
                                    :bytes (access-bytes base index frames))
                   (car accesses)))
           (expr (term frames)
             (cond ((atom term) nil)
                   ((eq (car term) 'load)
                    (destructuring-bind (base index) (cdr term)
                      (record :read base index frames)
                      (when-let ((dims (base-dims (second base))))
                        (index-dims index dims (frame-tile-size-fn frames)))))
                   (t
                    (let* ((fsym (car term))
                           (shapes (mapcar (lambda (a) (expr a frames)) (cdr term)))
                           (constants (mapcar (lambda (a) (and (numberp a) a)) (cdr term)))
                           (shape (op-shape fsym shapes constants)))
                      (multiple-value-bind (flops tensor-core-p) (op-flops fsym shapes shape)
                        (let ((total (* flops (repeat frames))))
                          (if tensor-core-p
                              (incf (kernel-stats-tensor-core-flops stats) total)
                              (incf (kernel-stats-simt-flops stats) total))))
                      shape))))
           (statement (term frames)
             (when (consp term)
               (case (car term)
                 ((loop sloop ploop)
                  (destructuring-bind (start end tile var body) (cdr term)
                    (let* ((tile (if (numberp tile) tile *default-tile*))
                           (frame (make-frame :var var :tile tile
                                              :trips (max 1 (ceiling (- end start) tile))
                                              :parallel-p (eq (car term) 'ploop))))
                      (statement body (cons frame frames)))))
                 (seq (statement (second term) frames)
                      (statement (third term) frames))
                 (store (destructuring-bind (base val index) (cdr term)
                          (record :write base index frames)
                          (expr val frames)))))))
    (statement term '())))

(defun kernel-grid (term)
  (if (and (consp term) (eq (car term) 'ploop))
      (destructuring-bind (start end tile var body) (cdr term)
        (declare (ignore var))
        (* (max 1 (ceiling (- end start) (if (numberp tile) tile *default-tile*)))
           (kernel-grid body)))
      1))

(defun tiling-frames (access)
  "Enclosing sequential loops that ACCESS walks over tile by tile."
  (remove-if-not (lambda (f)
                   (and (not (frame-parallel-p f))
                        (index-depends-on-p (ma-index access) (frame-var f))))
                 (ma-frames access)))

(defun on-chip-p (access accesses)
  "Whether ACCESS touches an intermediate that can stay on chip. It must also be
written in ACCESSES, so that a program fragment (see `extract-best') pays for
reading intermediates produced elsewhere."
  (and (eq (first (ma-base access)) 'tensor)
       (let* ((names (split-names (second (ma-base access))))
              (related (remove-if-not
                        (lambda (a) (names-overlap-p names (split-names (second (ma-base a)))))
                        accesses))
              (tiling (tiling-frames access)))
         (and (find :write related :key #'ma-kind)
              (every (lambda (a)
                       (and (eql (ma-kernel a) (ma-kernel access))
                            (null (set-exclusive-or (tiling-frames a) tiling))))
                     related)))))

(defun kernel-terms (program)
  (remove-if (lambda (s) (or (atom s) (null (collect-stores s))))
             (flatten-seq program)))

(defun on-chip-tensors (kernels)
  "Base symbols of the intermediates that stay on chip. This does not depend
on tile sizes."
  (let ((accesses (list '())))
    (loop for term in kernels
          for k from 0
          do (walk-kernel term k (make-kernel-stats) accesses))
    (let ((all (car accesses)))
      (remove-duplicates (loop for a in all
                               when (on-chip-p a all)
                                 collect (second (ma-base a)))))))

(defun kernel-cost (term k on-chip)
  "`kernel-stats' of kernel TERM, the K-th, for the current `*default-tile*'."
  (let ((s (make-kernel-stats :tile *default-tile* :grid (kernel-grid term)))
        (accesses (list '())))
    (walk-kernel term k s accesses)
    (dolist (a (car accesses))
      (unless (member (second (ma-base a)) on-chip)
        (incf (kernel-stats-bytes s) (ma-bytes a))))
    (let ((util (min 1 (/ (kernel-stats-grid s) (hw :num-sms)))))
      (setf (kernel-stats-time s)
            (+ (hw :launch-overhead)
               (/ (max (/ (kernel-stats-bytes s) (hw :bandwidth))
                       (+ (/ (kernel-stats-tensor-core-flops s) (hw :tensor-core-flops))
                          (/ (kernel-stats-simt-flops s) (hw :simt-flops))))
                  util))))
    s))

(defun cost-breakdown (program)
  "Estimate the run time of postprocessed PROGRAM. Returns the time in seconds
and a list of per-kernel `kernel-stats', each with its best symbolic tile size."
  (let* ((kernels (kernel-terms program))
         (on-chip (on-chip-tensors kernels))
         (stats (loop for term in kernels
                      for k from 0
                      collect (loop with best = nil
                                    for tile in *tile-choices*
                                    for s = (let ((*default-tile* tile))
                                              (kernel-cost term k on-chip))
                                    when (or (null best)
                                             (< (kernel-stats-time s) (kernel-stats-time best)))
                                      do (setf best s)
                                    finally (return best)))))
    (values (reduce #'+ stats :key #'kernel-stats-time) stats)))

(defun program-cost (program)
  "Estimated run time of postprocessed PROGRAM in microseconds."
  (* 1e6 (values (cost-breakdown program))))
