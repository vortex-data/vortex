from __future__ import annotations
import typing
__all__: list = ['column', 'ident', 'literal', 'Expr']
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
def column(name):
    """
    Create an expression that refers to a column by its name.
    
    Parameters
    ----------
    name : :class:`str`
        The name of the column.
    
    Returns
    -------
    :class:`vortex.Expr`
    
    Examples
    --------
    
        >>> import vortex.expr as ve
        >>> ve.column("age")
        <vortex.Expr object at ...>
    """
def ident():
    """
    Create an expression that refers to the identity scope.
    
    That is, it returns the full input that the extension is run against.
    
    Returns
    -------
    :class:`vortex.Expr`
    
    Examples
    --------
    
        >>> import vortex.expr as ve
        >>> ve.ident()
        ident()
    """
def literal(dtype, value):
    """
    Create an expression that represents a literal value.
    
    Parameters
    ----------
    dtype : :class:`vortex.DType`
        The data type of the literal value.
    value : :class:`Any`
        The literal value.
    
    Returns
    -------
    :class:`vortex.Expr`
    
    Examples
    --------
    
        >>> import vortex.expr as ve
        >>> ve.literal(ve.int_(), 42)
        literal(int(), 42)
    """
