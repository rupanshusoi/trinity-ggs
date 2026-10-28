;;; Load trinity-ggs and its dependencies through Quicklisp.
;;;
;;;   sbcl --load scripts/load.lisp
;;;
;;; GGS is looked up in $GGS_DIR, defaulting to a ggs checkout next to this
;;; repository.

(require :asdf)

(let* ((repo (truename (merge-pathnames "../" (directory-namestring *load-truename*))))
       (ggs (or (uiop:getenv-absolute-directory "GGS_DIR")
                (merge-pathnames "../ggs/" repo))))
  (pushnew repo asdf:*central-registry* :test #'equal)
  (pushnew ggs asdf:*central-registry* :test #'equal))

(ql:quickload :trinity-ggs :silent t)
