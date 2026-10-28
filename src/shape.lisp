(in-package #:trinity-ggs)

;;; Shape inference for tile expressions, shared by the e-graph shape analysis
;;; (which, like Trinity, assumes a default size for symbolic tiles) and the
;;; cost model (which knows each loop's tile size).

(defparameter *default-tile* 64
  "Size assumed for symbolic tile sizes such as tile_n. Trinity uses 64.")

(defun default-tile-size (loop-var)
  (declare (ignore loop-var))
  *default-tile*)

(defun index-dims (index base-dims &optional (tile-size-fn #'default-tile-size))
  "Dimensions of the tile selected by INDEX from a tensor of shape BASE-DIMS.
TILE-SIZE-FN maps a loop variable to its tile size."
  (loop with components = (term-args index)
        for dim in base-dims
        for c = (pop components)
        collect (case (term-fsym c)
                  (tile (min dim (funcall tile-size-fn (second c))))
                  (elem 1)
                  (const_tile (min dim (third c)))
                  (t dim))))

(defun numel (shape) (reduce #'* shape))

(defun broadcast-shapes (a b)
  (cond ((null a) b)
        ((null b) a)
        ((/= (length a) (length b)) (if (> (length a) (length b)) a b))
        (t (mapcar (lambda (x y) (cond ((= x y) x) ((= x 1) y) ((= y 1) x) (t (max x y))))
                   a b))))

(defun remove-axis (shape axis)
  (if (and shape (integerp axis) (< -1 axis (length shape)))
      (append (subseq shape 0 axis) (nthcdr (1+ axis) shape))
      shape))

(defun insert-axis (shape axis)
  (if (and shape (integerp axis) (<= 0 axis (length shape)))
      (append (subseq shape 0 axis) (list 1) (nthcdr axis shape))
      shape))

(defparameter +binary-ops+ '(+ - * / <= max min))
(defparameter +unary-ops+ '(exp sqr sqrt sigmoid erf abs))
(defparameter +reduce-ops+ '(rsum rmin rmax))

(defun op-shape (fsym shapes constants)
  "Shape of an operator FSYM applied to arguments with SHAPES. CONSTANTS holds
the numeric value of each argument, or NIL, for axis arguments."
  (flet ((arg (i) (nth i shapes))
         (const (i) (nth i constants)))
    (cond ((member fsym +binary-ops+) (broadcast-shapes (arg 0) (arg 1)))
          ((member fsym +unary-ops+) (arg 0))
          ((member fsym +reduce-ops+) (remove-axis (arg 0) (const 1)))
          (t
           (case fsym
             (cast (arg 1))
             (@ (let ((a (arg 0)) (b (arg 1)))
                  (when (and (>= (length a) 2) (>= (length b) 2))
                    (append (butlast (if (>= (length a) (length b)) a b) 2)
                            (list (nth (- (length a) 2) a) (lastcar b))))))
             (bcast (insert-axis (arg 0) (const 1)))
             (unsqueeze (insert-axis (arg 0) (const 1)))
             (squeeze (remove-axis (arg 0) (const 1)))
             (transpose (let ((a (arg 0)))
                          (when (>= (length a) 2)
                            (append (butlast a 2) (reverse (last a 2))))))
             ((permute3 permute4)
              (let ((a (arg 0)) (perm (rest constants)))
                (when (and a (every #'integerp perm) (= (length perm) (length a)))
                  (mapcar (lambda (i) (nth i a)) perm))))
             (concat (let ((a (arg 0)) (b (arg 1)) (axis (const 2)))
                       (if (and a b (integerp axis) (< -1 axis (length a)))
                           (loop for x in a for y in b for i from 0
                                 collect (if (= i axis) (+ x y) x))
                           a)))
             (t nil))))))

(defun op-flops (fsym shapes result-shape)
  "FLOPs of one application of FSYM, following Trinity's fine-grained cost
model. Second value is true for matmuls, which run on tensor cores."
  (cond ((eq fsym '@)
         (let ((a (first shapes)) (b (second shapes)))
           (if (and (>= (length a) 2) (>= (length b) 2))
               (let ((m (nth (- (length a) 2) a)) (k (lastcar a)) (n (lastcar b)))
                 (values (* (numel (butlast a 2)) m n (1- (* 2 k))) t))
               0)))
        ((member fsym +binary-ops+) (numel result-shape))
        ((or (member fsym +unary-ops+) (eq fsym 'cast)) (* 10 (numel result-shape)))
        ((member fsym +reduce-ops+) (numel (first shapes)))
        (t 0)))
