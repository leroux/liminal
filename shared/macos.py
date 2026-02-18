"""macOS app identity helpers (ctypes — no PyObjC dependency)."""

import sys


def get_display_scale() -> float:
    """Return the display backing scale factor (2.0 for Retina, 1.0 otherwise).

    Uses NSScreen.mainScreen.backingScaleFactor via ctypes so no PyObjC is needed.
    Returns 1.0 on non-macOS or if detection fails.
    """
    if sys.platform != 'darwin':
        return 1.0
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
        lib.objc_getClass.restype = ctypes.c_void_p
        lib.objc_getClass.argtypes = [ctypes.c_char_p]
        lib.sel_registerName.restype = ctypes.c_void_p
        lib.sel_registerName.argtypes = [ctypes.c_char_p]

        # NSScreen.mainScreen
        lib.objc_msgSend.restype = ctypes.c_void_p
        lib.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        NSScreen = lib.objc_getClass(b'NSScreen')
        screen = lib.objc_msgSend(NSScreen, lib.sel_registerName(b'mainScreen'))

        # backingScaleFactor returns CGFloat (mapped to c_double)
        lib.objc_msgSend.restype = ctypes.c_double
        scale = lib.objc_msgSend(screen, lib.sel_registerName(b'backingScaleFactor'))
        return float(scale)
    except Exception:
        return 1.0


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
