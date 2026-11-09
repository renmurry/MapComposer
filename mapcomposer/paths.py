"""Where bundled resources and the user's editable copies live."""
import os
import sys
import shutil


def resource(*parts):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    if hasattr(sys, "_MEIPASS"):
        base = os.path.join(base, "mapcomposer")
    return os.path.join(base, "resources", *parts)


def user_dir():
    root = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(root, "MapComposer")
    os.makedirs(os.path.join(d, "templates"), exist_ok=True)
    return d


def ensure_user_copies():
    """Copy built-in styles/templates into the user folder once, so they can be edited there."""
    d = user_dir()
    for f in ("styles.yaml", "themes.yaml"):
        if not os.path.exists(os.path.join(d, f)):
            shutil.copy(resource(f), os.path.join(d, f))
    tdir = resource("templates")
    for f in os.listdir(tdir):
        dst = os.path.join(d, "templates", f)
        if f.endswith(".yaml") and not os.path.exists(dst):
            shutil.copy(os.path.join(tdir, f), dst)
    return d
