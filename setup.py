from setuptools import setup, find_packages

# Function to read the dependencies from requirements.txt
def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read dependencies from requirements.txt
install_requires = parse_requirements("requirements.txt")

print(install_requires)

setup(
    name='face_image_quality',
    version='0.1',
    packages=find_packages(),
    install_requires=install_requires,
)