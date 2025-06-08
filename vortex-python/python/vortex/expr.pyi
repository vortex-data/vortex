from __future__ import annotations
from expr import column
from expr import ident
from expr import literal
import typing

__all__: list = ["column", "ident", "literal", "Expr"]

class Expr:
    """
    An expression describes how to filter rows when reading an array from a file.

    .. seealso::
       :func:`.column`
    """

    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __and__(self, value):
        """
        Return self&value.
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __getitem__(self, key):
        """
        Return self[key].
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
    def __or__(self, value):
        """
        Return self|value.
        """
    def __rand__(self, value):
        """
        Return value&self.
        """
    def __ror__(self, value):
        """
        Return value|self.
        """
    def __str__(self):
        """
        Return str(self).
        """
