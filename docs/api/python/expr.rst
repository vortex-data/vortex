Expressions
===========

Vortex expressions represent simple filtering conditions on the rows of a Vortex array. For example,
the following expression represents the set of rows for which the `age` column lies between 23 and
55:

.. doctest::

   >>> import vortex as vx
   >>> age = vx.col("age")
   >>> (23 > age) & (age < 55)  # doctest: +SKIP

.. autosummary::
   :nosignatures:

   ~vortex.expr.column
   ~vortex.expr.col
   ~vortex.expr.plan
   ~vortex.expr.Expr

.. raw:: html

   <hr>

.. autofunction:: vortex.expr.column

.. autofunction:: vortex.expr.col

.. autofunction:: vortex.expr.not_

.. autofunction:: vortex.expr.and_

.. autofunction:: vortex.expr.cast

.. autofunction:: vortex.expr.is_null

.. autofunction:: vortex.expr.is_not_null

.. autofunction:: vortex.expr.plan

.. autofunction:: vortex.expr.root

.. autofunction:: vortex.expr.literal

.. autoclass:: vortex.expr.Expr
   :members:

   .. py:method:: __getitem__ (name, /)

      Extract a field of a struct array.

      :parameters:

          - **name** (:class:`.str`) -- The name of the field.

      :return type:

          :class:`.vortex.Expr`

      .. rubric:: Examples

      >>> import vortex as vx
      >>> import vortex.expr as ve
      >>> import pyarrow as pa
      >>>
      >>> array = pa.array([
      ...     {"x": 1, "y": {"yy": "a"}},
      ...     {"x": 2, "y": {"yy": "b"}},
      ... ])
      >>>
      >>> vx.io.write(vx.array(array), '/tmp/foo.vortex')
      >>> (vx.file.open('/tmp/foo.vortex')
      ...    .scan(expr=vx.expr.column("y")["yy"] == "a")
      ...    .read_all()
      ...    .to_pylist()
      ... )
      [{'x': 1, 'y': {'yy': 'a'}}]
