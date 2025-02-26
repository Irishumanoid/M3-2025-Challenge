import subprocess
import sys

def install(package_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

packages = ['requests', 'numpy', 'pandas', 'matplotlib', 'scipy', 
            'ortools', 'statsmodels', 'scikit-learn', "pgmpy"]

for package in packages:
    install(package)