"""Locations of the data files and results, relative to the root of the repository."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(REPO, "code")
DATA = os.path.join(REPO, "data")
INPUTS = os.path.join(DATA, "inputs")
RESULTS = os.path.join(REPO, "certificates", "results")

# inputs
KS_FOURFOLDS = os.path.join(INPUTS, "5dTransWH.all.gz")        # Kreuzer-Skarke list of the 1,100,055 CY4 weight systems
CY3_WEIGHTS = os.path.join(INPUTS, "WP4s.txt")                 # the 7555 CY3 weight systems
CY3_HODGE = os.path.join(INPUTS, "WP4_Hodges.txt")             # their Kreuzer-Skarke (h11, h21)
CY3_GROEBNER = os.path.join(INPUTS, "Topological_Data.txt")    # Groebner-basis data of arXiv:2310.03064 (7549 links)
K3_GROEBNER = os.path.join(INPUTS, "K3Topological_Data.txt")   # Groebner-basis data for the 95 K3 links
CY4_POLYS_42 = os.path.join(INPUTS, "CY4Polynomials_42.txt")   # 42 dense CY4 polynomials

# datasets produced by code/
K3_TABLE = os.path.join(DATA, "k3_links.csv")
CY3_TABLE = os.path.join(DATA, "cy3_links.csv")
CY4_TABLE = os.path.join(DATA, "cy4_links.csv.gz")
