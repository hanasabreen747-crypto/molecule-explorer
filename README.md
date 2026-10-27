# Molecule Explorer (Streamlit)

A student project that searches PubChem by molecule name and displays:
- 2D molecular structure
- Generated 3D model (rotate and zoom)
- Canonical/isomeric SMILES
- Formula, molecular weight, LogP, TPSA, HBD, HBA, approximate pi-bond count, rotatable bonds, heavy atoms, ring count, fraction Csp3, and formal charge
- Plain-language interpretations
- Downloadable TXT, JSON, and CSV reports

## Run locally

1. Install Python 3.10 or newer.
2. Extract the ZIP and open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`
6. Open the local URL shown in the terminal (usually http://localhost:8501).

Name lookup needs an internet connection because the app queries PubChem. RDKit calculates descriptors and generates structures locally.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open Streamlit Community Cloud and deploy the repository, selecting `app.py` as the main file.
3. Wait for dependencies to install.

## Notes
- The first PubChem result is used. For ambiguous names, verify the returned CID, formula, and structure.
- π-bond count is approximate and based on a Kekulé representation; aromatic bonding is delocalized.
- Descriptor interpretations are educational heuristics, not medical or toxicological conclusions.
