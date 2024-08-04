import toml
import re

def parse_requirement(req):
    # This regex will match package names and versions, including complex specifiers
    match = re.match(r'^([^=<>]+)([=<>]+.+)?$', req)
    if match:
        return match.group(1), match.group(2) or "*"
    return None, None

# Read the current Pipfile
with open('Pipfile', 'r') as f:
    pipfile = toml.load(f)

# Ensure 'packages' key exists
if 'packages' not in pipfile:
    pipfile['packages'] = {}

# Read the requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Update the packages in the Pipfile
for req in requirements:
    if req and not req.startswith('#'):
        package, version = parse_requirement(req)
        if package:
            pipfile['packages'][package] = version

# Write the updated Pipfile
with open('Pipfile', 'w') as f:
    toml.dump(pipfile, f)

print("Pipfile has been updated based on requirements.txt")