from typing import Optional


class MopsError(Exception):
    """This class is used to throw exceptions for all errors in mops."""

    def __init__(self, message, status=None):
        super(Exception, self).__init__(message)

        self.message: str = message
        """error message for this exception"""

        self.status: Optional[int] = status
        """status code for this exception"""


LAST_EXCEPTION = None


def _save_exception(e):
    global LAST_EXCEPTION
    LAST_EXCEPTION = e


def _check_status(status):
    if status == 0:
        return
    else:
        raise MopsError(last_error(), status)


def last_error():
    """Get the last error message on this thread"""
    from ._c_lib import _get_library

    lib = _get_library()
    message = lib.mops_get_last_error_message()
    return message.decode("utf8")
