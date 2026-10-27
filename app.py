import io
import json
from datetime import datetime

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, rdMolDescriptors
from pubchempy import get_compounds

st.set_page_config(page_title="Molecule Explorer", page_icon="🧪", layout="wide")

st.title("🧪 Molecule Explorer")
st.caption("Search a molecule by name and explore its structure, SMILES, properties, and interpretation.")

st.info(
    "Molecule names are looked up through PubChem, so name search requires an internet connection. "
    "Calculated descriptors are estimates based on the retrieved molecular structure."
)

@st.cache_data(show_spinner=False, ttl=3600)
def lookup_molecule(name: str):
    compounds = get_compounds(name.strip(), "name")
    if not compounds:
        return None
    c = compounds[0]
    return {
        "name": getattr(c, "iupac_name", None) or name.strip(),
        "cid": getattr(c, "cid", None),
        "smiles": getattr(c, "canonical_smiles", None) or getattr(c, "isomeric_smiles", None),
        "formula": getattr(c, "molecular_formula", None),
        "pubchem_iupac": getattr(c, "iupac_name", None),
    }

def make_3d_viewer(mol):
    mol3d = Chem.AddHs(Chem.Mol(mol))
    status = AllChem.EmbedMolecule(mol3d, randomSeed=42)
    if status != 0:
        raise ValueError("RDKit could not generate a 3D conformer for this structure.")
    # MMFF is preferred; fall back to UFF when MMFF parameters are unavailable.
    if AllChem.MMFFHasAllMoleculeParams(mol3d):
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=300)
    else:
        AllChem.UFFOptimizeMolecule(mol3d, maxIters=300)
    mol_block = Chem.MolToMolBlock(mol3d)
    viewer = py3Dmol.view(width=720, height=480)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    return viewer._make_html()

def count_pi_bonds(mol):
    # A Kekulé form gives an understandable approximate count for aromatic systems.
    # Triple bonds contribute two pi bonds each.
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
    except Exception:
        pass
    pi_bonds = 0
    for bond in kekule.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            pi_bonds += 2
    return pi_bonds

def property_data(mol):
    return [
        ("Molecular formula", rdMolDescriptors.CalcMolFormula(mol), "Counts each element in the molecule."),
        ("Molecular weight", f"{Descriptors.MolWt(mol):.3f} g/mol", "Approximate molar mass from the molecular structure."),
        ("LogP", f"{Descriptors.MolLogP(mol):.3f}", "Estimated preference for an octanol-like (fat-like) phase versus water. Higher values usually mean more lipophilic."),
        ("TPSA", f"{rdMolDescriptors.CalcTPSA(mol):.3f} Å²", "Topological polar surface area. Higher values often indicate more polar surface and can be associated with lower passive membrane permeability."),
        ("HBD", str(Lipinski.NumHDonors(mol)), "Hydrogen-bond donors: commonly O–H or N–H groups able to donate a hydrogen bond."),
        ("HBA", str(Lipinski.NumHAcceptors(mol)), "Hydrogen-bond acceptors: atoms such as suitable O or N atoms that can accept a hydrogen bond."),
        ("π bonds (approx.)", str(count_pi_bonds(mol)), "Estimated from a Kekulé representation: double bonds count as one π bond and triple bonds as two. Aromatic bonding is delocalized, so this is a representation-based count."),
        ("Rotatable bonds", str(Lipinski.NumRotatableBonds(mol)), "A rough measure of molecular flexibility; the exact definition depends on the descriptor rules."),
        ("Heavy atoms", str(mol.GetNumHeavyAtoms()), "Number of non-hydrogen atoms."),
        ("Ring count", str(rdMolDescriptors.CalcNumRings(mol)), "Number of rings according to RDKit's ring-counting method."),
        ("Fraction Csp³", f"{Descriptors.FractionCSP3(mol):.3f}", "Fraction of carbon atoms that are sp³-hybridized; larger values often indicate more saturated carbon structure."),
        ("Formal charge", str(Chem.GetFormalCharge(mol)), "Net formal charge encoded by the structure."),
    ]

def interpret_value(label, raw_value):
    try:
        value = float(str(raw_value).split()[0])
    except Exception:
        return "See the definition above; this value is structure-derived."
    if label == "LogP":
        if value < 0:
            return "Relatively hydrophilic by this estimate; water affinity may be greater."
        if value < 3:
            return "Moderate lipophilicity by this estimate."
        return "Relatively lipophilic by this estimate; water solubility may be limited."
    if label == "TPSA":
        if value < 40:
            return "Low polar surface area; this descriptor alone does not predict absorption."
        if value < 90:
            return "Moderate polar surface area."
        return "High polar surface area; passive membrane permeability may be lower in some contexts."
    if label == "HBD":
        return "No donor groups identified." if value == 0 else f"{int(value)} potential hydrogen-bond donor group(s) identified."
    if label == "HBA":
        return "No acceptor atoms identified." if value == 0 else f"{int(value)} potential hydrogen-bond acceptor atom(s) identified."
    if label == "Rotatable bonds":
        return "Relatively rigid by this simple descriptor." if value <= 2 else "Has several rotatable bonds and may be more flexible."
    if label == "π bonds (approx.)":
        return "The structure contains no π bonds in this representation." if value == 0 else "π bonds contribute to unsaturation and electronic structure; aromatic bonding is delocalized."
    return "Interpret together with the molecular structure and other descriptors."

with st.sidebar:
    st.header("Search")
    molecule_name = st.text_input("Molecule name", value="caffeine", placeholder="e.g., ethanol, aspirin, glucose")
    search = st.button("Search molecule", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Tip: use a common chemical name. If a name is ambiguous, verify the returned structure and CID.")

if "search_name" not in st.session_state:
    st.session_state.search_name = "caffeine"
if search:
    st.session_state.search_name = molecule_name.strip()

query = st.session_state.search_name
if not query:
    st.warning("Enter a molecule name in the sidebar.")
    st.stop()

try:
    with st.spinner(f"Searching PubChem for {query}..."):
        result = lookup_molecule(query)
except Exception as e:
    st.error("Could not reach PubChem. Check your internet connection and try again.")
    st.caption(f"Technical detail: {type(e).__name__}: {e}")
    st.stop()

if not result or not result.get("smiles"):
    st.error(f"No usable structure was found for “{query}”. Try a more common name.")
    st.stop()

mol = Chem.MolFromSmiles(result["smiles"])
if mol is None:
    st.error("The returned SMILES could not be parsed by RDKit.")
    st.stop()

Chem.rdDepictor.Compute2DCoords(mol)
props = property_data(mol)
display_name = query.title()
formula = rdMolDescriptors.CalcMolFormula(mol)

top1, top2, top3 = st.columns(3)
top1.metric("Molecular formula", formula)
top2.metric("Molecular weight", f"{Descriptors.MolWt(mol):.2f} g/mol")
top3.metric("PubChem CID", str(result["cid"] or "Not available"))

st.subheader(display_name)
if result.get("pubchem_iupac"):
    st.caption(f"IUPAC name: {result['pubchem_iupac']}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Properties & interpretation", "SMILES", "Download report"])

with tab1:
    left, right = st.columns(2)
    with left:
        st.markdown("#### 2D structure")
        image = Draw.MolToImage(mol, size=(600, 420), kekulize=True)
        st.image(image, use_container_width=True)
    with right:
        st.markdown("#### 3D structure")
        st.caption("Drag to rotate; scroll or pinch to zoom. Hydrogens are added for the 3D model.")
        try:
            html = make_3d_viewer(mol)
            components.html(html, height=500, scrolling=False)
        except Exception as e:
            st.warning("The 3D conformer could not be generated for this molecule.")
            st.caption(f"Technical detail: {type(e).__name__}: {e}")

with tab2:
    df = pd.DataFrame(props, columns=["Property", "Value", "Meaning"])
    df["Interpretation"] = [interpret_value(row[0], row[1]) for row in props]
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("Interpretations are general educational guidance, not predictions of clinical effect, toxicity, or drug suitability.")

with tab3:
    st.markdown("#### Canonical SMILES")
    st.code(result["smiles"], language="text")
    st.caption("SMILES is a text representation of a molecular structure. Different valid SMILES strings can represent the same molecule.")
    st.markdown("#### Isomeric SMILES (if available)")
    try:
        from pubchempy import Compound
        compounds = get_compounds(query, "name")
        iso = getattr(compounds[0], "isomeric_smiles", None) if compounds else None
        st.code(iso or result["smiles"], language="text")
    except Exception:
        st.code(result["smiles"], language="text")

with tab4:
    report = {
        "app": "Molecule Explorer",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "searched_name": query,
        "pubchem_cid": result["cid"],
        "iupac_name": result.get("pubchem_iupac"),
        "smiles": result["smiles"],
        "molecular_formula": formula,
        "properties": [
            {"property": name, "value": value, "meaning": meaning,
             "interpretation": interpret_value(name, value)}
            for name, value, meaning in props
        ],
        "note": "Calculated descriptors and interpretations are educational estimates. Verify important results against authoritative chemistry references."
    }
    report_text = [
        "MOLECULE EXPLORER REPORT",
        "=" * 28,
        f"Search term: {query}",
        f"PubChem CID: {result['cid']}",
        f"IUPAC name: {result.get('pubchem_iupac') or 'Not available'}",
        f"Formula: {formula}",
        f"SMILES: {result['smiles']}",
        "",
        "PROPERTIES",
        "-" * 10,
    ]
    for name, value, meaning in props:
        report_text += [f"{name}: {value}", f"Meaning: {meaning}", f"Interpretation: {interpret_value(name, value)}", ""]
    report_text += [report["note"]]
    st.download_button(
        "Download text report (.txt)",
        data="\n".join(report_text),
        file_name=f"{query.lower().replace(' ', '_')}_molecule_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.download_button(
        "Download structured report (.json)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name=f"{query.lower().replace(' ', '_')}_molecule_report.json",
        mime="application/json",
        use_container_width=True,
    )
    st.download_button(
        "Download properties table (.csv)",
        data=pd.DataFrame(props, columns=["Property", "Value", "Meaning"]).to_csv(index=False),
        file_name=f"{query.lower().replace(' ', '_')}_properties.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Data source: PubChem name lookup. 2D/3D structures and descriptors are processed locally by RDKit. This tool is for learning and should not be used as the sole source for safety-critical decisions.")
