Runtime
=======

Vortex drives async work on a shared background thread pool. The pool is sized on first use
to ``VORTEX_MAX_THREADS`` if that environment variable is set to a non-negative integer,
otherwise to the number of available CPU cores minus one. Use
:func:`vortex.set_worker_threads` to adjust the pool at runtime.

.. autosummary::
   :nosignatures:

   ~vortex.cuda_extension_installed
   ~vortex.set_worker_threads
   ~vortex.worker_threads

.. raw:: html

   <hr>

.. autofunction:: vortex.cuda_extension_installed

.. autofunction:: vortex.set_worker_threads

.. autofunction:: vortex.worker_threads
