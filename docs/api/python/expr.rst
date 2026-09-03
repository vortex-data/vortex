Expressions
===========

Vortex expressions represent simple filtering conditions on the rows of a Vortex array. For example,
the following expression represents the set of rows for which the `age` column lies between 23 and
55:

.. doctest::

   >>> import vortex.expr
   >>> age = vortex.expr.column("age")
   >>> (23 > age) & (age < 55)  # doctest: +SKIP

Expressions are picklable, so a filter built in one process can be sent to another (for example to a
``multiprocessing`` worker or a Ray task). Pickling uses the same protobuf wire format exposed by
:meth:`vortex.expr.Expr.serialize` and :func:`vortex.expr.deserialize`.

.. autosummary::
   :nosignatures:

   ~vortex.expr.Expr
   ~vortex.expr.root
   ~vortex.expr.column
   ~vortex.expr.literal
   ~vortex.expr.get_item
   ~vortex.expr.not_
   ~vortex.expr.and_
   ~vortex.expr.or_
   ~vortex.expr.and_collect
   ~vortex.expr.or_collect
   ~vortex.expr.eq
   ~vortex.expr.not_eq
   ~vortex.expr.gt
   ~vortex.expr.gt_eq
   ~vortex.expr.lt
   ~vortex.expr.lt_eq
   ~vortex.expr.add
   ~vortex.expr.sub
   ~vortex.expr.mul
   ~vortex.expr.div
   ~vortex.expr.between
   ~vortex.expr.is_null
   ~vortex.expr.is_not_null
   ~vortex.expr.fill_null
   ~vortex.expr.like
   ~vortex.expr.ilike
   ~vortex.expr.not_like
   ~vortex.expr.not_ilike
   ~vortex.expr.byte_length
   ~vortex.expr.select
   ~vortex.expr.select_exclude
   ~vortex.expr.pack
   ~vortex.expr.merge
   ~vortex.expr.list_contains
   ~vortex.expr.list_length
   ~vortex.expr.list_sum
   ~vortex.expr.case_when
   ~vortex.expr.zip_
   ~vortex.expr.mask
   ~vortex.expr.cast
   ~vortex.expr.ext_storage
   ~vortex.expr.variant_get
   ~vortex.expr.deserialize

.. raw:: html

   <hr>

Leaves and scope
----------------

.. autofunction:: vortex.expr.root

.. autofunction:: vortex.expr.column

.. autofunction:: vortex.expr.literal

.. autofunction:: vortex.expr.get_item

Boolean logic
-------------

.. autofunction:: vortex.expr.not_

.. autofunction:: vortex.expr.and_

.. autofunction:: vortex.expr.or_

.. autofunction:: vortex.expr.and_collect

.. autofunction:: vortex.expr.or_collect

Comparisons and arithmetic
--------------------------

.. autofunction:: vortex.expr.eq

.. autofunction:: vortex.expr.not_eq

.. autofunction:: vortex.expr.gt

.. autofunction:: vortex.expr.gt_eq

.. autofunction:: vortex.expr.lt

.. autofunction:: vortex.expr.lt_eq

.. autofunction:: vortex.expr.add

.. autofunction:: vortex.expr.sub

.. autofunction:: vortex.expr.mul

.. autofunction:: vortex.expr.div

.. autofunction:: vortex.expr.between

Nullability
-----------

.. autofunction:: vortex.expr.is_null

.. autofunction:: vortex.expr.is_not_null

.. autofunction:: vortex.expr.fill_null

Strings
-------

.. autofunction:: vortex.expr.like

.. autofunction:: vortex.expr.ilike

.. autofunction:: vortex.expr.not_like

.. autofunction:: vortex.expr.not_ilike

.. autofunction:: vortex.expr.byte_length

Structs
-------

.. autofunction:: vortex.expr.select

.. autofunction:: vortex.expr.select_exclude

.. autofunction:: vortex.expr.pack

.. autofunction:: vortex.expr.merge

Lists
-----

.. autofunction:: vortex.expr.list_contains

.. autofunction:: vortex.expr.list_length

.. autofunction:: vortex.expr.list_sum

Conditionals and conversions
----------------------------

.. autofunction:: vortex.expr.case_when

.. autofunction:: vortex.expr.zip_

.. autofunction:: vortex.expr.mask

.. autofunction:: vortex.expr.cast

.. autofunction:: vortex.expr.ext_storage

.. autofunction:: vortex.expr.variant_get

Serialization
-------------

.. autofunction:: vortex.expr.deserialize

The expression class
--------------------

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
