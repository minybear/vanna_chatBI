import vanna.legacy
import os
import pkgutil

print(f"Legacy location: {vanna.legacy.__file__}")
package_path = os.path.dirname(vanna.legacy.__file__)

print("\nSubmodules:")
for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
    print(f"  {modname}")
