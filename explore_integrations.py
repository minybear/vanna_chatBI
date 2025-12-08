import vanna.integrations
import os
import pkgutil

print(f"Integrations location: {vanna.integrations.__file__}")
package_path = os.path.dirname(vanna.integrations.__file__)

print("\nSubmodules:")
for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
    print(f"  {modname}")
