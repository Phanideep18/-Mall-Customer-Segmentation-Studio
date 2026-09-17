"""
Environment and Dependency Verification Script
"""
import sys

modules = [
    ("streamlit", "Web Dashboard Framework"),
    ("pandas", "Data Manipulation & Analysis"),
    ("numpy", "Numerical Operations & Matrices"),
    ("sklearn", "Scikit-Learn Clustering & Metrics"),
    ("scipy", "Hierarchical Linkage & Scientific Computing"),
    ("plotly", "Interactive 2D/3D Data Visualizations"),
    ("matplotlib", "Static Plotting & Dendrograms"),
    ("seaborn", "Statistical Data Visualization")
]

print("=" * 75)
print(f" PYTHON ENVIRONMENT: {sys.version.split()[0]} ({sys.executable})")
print("=" * 75)
print(f"{'Package':<15} | {'Status':<10} | {'Version':<12} | {'Purpose'}")
print("-" * 75)

all_ready = True
for pkg, purpose in modules:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "Installed")
        print(f"{pkg:<15} | {'[OK]':<10} | {ver:<12} | {purpose}")
    except ImportError as e:
        print(f"{pkg:<15} | {'[MISSING]':<10} | {'N/A':<12} | {purpose}")
        all_ready = False

print("=" * 75)
if all_ready:
    print("[SUCCESS] All required modules are installed, verified, and ready to use!")
else:
    print("[WARNING] Some modules are missing. Run: pip install -r requirements.txt")
print("=" * 75)
