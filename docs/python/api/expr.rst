Expressions
===========

Vortex expressions represent simple filtering conditions on the rows of a Vortex array. For example,
the following expression represents the set of rows for which the `age` column lies between 23 and
55:

.. doctest::

   >>> import vortex.expr
   >>> age = vortex.expr.column("age")
   >>> (23 > age) & (age < 55)  # doctest: +SKIP

.. autosummary::
   :nosignatures:

   ~vortex.expr.column
   ~vortex.expr.Expr

.. raw:: html

   <hr>

.. autofunction:: vortex.expr.column

.. autoclass:: vortex.expr.Expr
