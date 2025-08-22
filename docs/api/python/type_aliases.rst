Type Aliases
============

.. data:: vortex.type_aliases.IntoArrayIterator

          Anything that can produce a sequence of Vortex Arrays.


.. data:: vortex.type_aliases.IntoProjection

          An expression, a list of column names, or None.

          Only the data necessary to evaluate the expression or produce the explicit column list are read.

          If None, all columns from the file are read.

