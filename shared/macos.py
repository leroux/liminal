"""macOS app identity helpers (ctypes — no PyObjC dependency)."""

import sys


def set_app_name(name: str) -> None:
    """Set the macOS dock / menu-bar app name (must be called before Tk mainloop).

    Uses the Objective-C runtime directly via ctypes so that we don't need
    PyObjC installed.  On non-macOS platforms this is a no-op.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]

        # We need a trampoline because objc_msgSend is variadic and ctypes
        # requires fixed argtypes per call-site.
        def send(obj, sel):
            lib.objc_msgSend.restype = ctypes.c_void_p
            lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            return lib.objc_msgSend(obj, lib.sel_registerName(sel))

        def send1(obj, sel, arg):
            lib.objc_msgSend.restype = ctypes.c_void_p
            lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p]
            return lib.objc_msgSend(obj, lib.sel_registerName(sel), arg)

        def send2(obj, sel, a, b):
            lib.objc_msgSend.restype = ctypes.c_void_p
            lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p]
            return lib.objc_msgSend(obj, lib.sel_registerName(sel), a, b)

        def nsstr(s: str):
            """Create an autoreleased NSString from a Python str."""
            cls = lib.objc_getClass(b"NSString")
            lib.objc_msgSend.restype = ctypes.c_void_p
            lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_char_p]
            return lib.objc_msgSend(cls,
                                    lib.sel_registerName(b"stringWithUTF8String:"),
                                    s.encode("utf-8"))

        # NSBundle.mainBundle.infoDictionary — this is an NSMutableDictionary
        bundle = send(lib.objc_getClass(b"NSBundle"), b"mainBundle")
        info = send(bundle, b"infoDictionary")

        # info[@"CFBundleName"] = name
        send2(info, b"setObject:forKey:", nsstr(name), nsstr("CFBundleName"))
    except Exception:
        pass  # best-effort; don't crash if anything goes wrong
