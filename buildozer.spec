[app]
# (str) Title of your application
title = Brutal Doom Shader

# (str) Package name
package.name = brutaldoomshader

# (str) Package domain (can be anything unique)
package.domain = org.doom

# (str) Source code directory (where your .py file is)
source.dir = .

# (list) Include these file types in your app
source.include_exts = py,png,ico

# (str) The main entry point (your main Python file, without .py)
entrypoint = sprite_doom

# (str) App version
version = 0.1

# (str) Orientation of the app
orientation = portrait

# (list) Python dependencies your app needs
requirements = python3,kivy,pillow,numpy,plyer,cython

# (str) Icon file
icon.filename = icon.ico

# (list) Permissions your app needs
android.permissions = WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

[buildozer]
# Let Buildozer run as root (important for CI/cloud builds)
warn_on_root = 0

# Set logging level
log_level = 2

