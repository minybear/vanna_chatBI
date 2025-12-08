import vanna.servers
import os
import pkgutil

print(f"Servers location: {vanna.servers.__file__}")
package_path = os.path.dirname(vanna.servers.__file__)

print("\nSubmodules:")
for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
    print(f"  {modname}")
