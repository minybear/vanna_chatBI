import vanna
import os
import pkgutil

print(f"Vanna location: {vanna.__file__}")
package_path = os.path.dirname(vanna.__file__)

print("\nSubmodules:")
for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
    print(f"  {modname} (is_pkg={ispkg})")
