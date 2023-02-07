import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List

# from sklearn.preprocessing import LabelBinarizer
# from rdkit.Chem.Pharm3D import EmbedLib, Pharmacophore
# from rdkit.Numerics import rdAlignment
import fire
import numpy as np
import pandas as pd
# chemical imports
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, TorsionFingerprints
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.ML.Cluster import Butina
# from rdkit.ML.Descriptors import MoleculeDescriptors
from rich.console import Console

console = Console()

fdefstr = """
AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor},!$([D1]-[C;D3]=[O,S,N])]
  Family Donor
  Weights 1
EndFeature

AtomType NAcceptor [$([N&v3;H1,H2]-[!$(*=[O,N,P,S])])]
Atomtype NAcceptor [$([N;v3;H0])]
AtomType NAcceptor [$([n;+0])]
AtomType ChalcAcceptor [$([O,S;H1;v2]-[!$(*=[O,N,P,S])])]
AtomType ChalcAcceptor [O,S;H0;v2]
Atomtype ChalcAcceptor [O,S;-]
Atomtype ChalcAcceptor [o,s;+0]
AtomType HalogenAcceptor [F]
DefineFeature SingleAtomAcceptor [{NAcceptor},{ChalcAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature

# this one is delightfully easy:
DefineFeature AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature

AtomType CarbonOrArom_NonCarbonyl [$([C,a]);!$([C,a](=O))]
AtomType BasicNH2 [$([N;H2&+0][{CarbonOrArom_NonCarbonyl}])]
AtomType BasicNH1 [$([N;H1&+0]([{CarbonOrArom_NonCarbonyl}])[{CarbonOrArom_NonCarbonyl}])]
AtomType BasicNH0 [$([N;H0&+0]([{CarbonOrArom_NonCarbonyl}])([{CarbonOrArom_NonCarbonyl}])[{CarbonOrArom_NonCarbonyl}])]
AtomType BasicNakedN [N,n;X2;+0]
DefineFeature BasicGroup [{BasicNH2},{BasicNH1},{BasicNH0},{BasicNakedN}]
  Family PosIonizable
  Weights 1.0
EndFeature

# aromatic rings of various sizes:
DefineFeature Arom5 a1aaaa1
  Family Aromatic
  Weights 1.0,1.0,1.0,1.0,1.0
EndFeature
DefineFeature Arom6 a1aaaaa1
  Family Aromatic
  Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
DefineFeature Arom7 a1aaaaaa1
  Family Aromatic
  Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
DefineFeature Arom8 a1aaaaaaa1
  Family Aromatic
  Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
"""
# =====================================
# Hydrophobe, Acceptor, Donor, ZnBinder, Aromatic, LumpedHydrophobe, NegIonizable, and PosIonizable.
# We then calculated the pharmacophore signature using distance bins of [(0,3), (3,6), (6,9)]


PROF_fdefstr = """
AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor},!$([D1]-[C;D3]=[O,S,N])]
  Family Donor
  Weights 1
EndFeature

AtomType NAcceptor [$([N&v3;H1,H2]-[!$(*=[O,N,P,S])])]
Atomtype NAcceptor [$([N;v3;H0])]
AtomType NAcceptor [$([n;+0])]
AtomType ChalcAcceptor [$([O,S;H1;v2]-[!$(*=[O,N,P,S])])]
AtomType ChalcAcceptor [O,S;H0;v2]
Atomtype ChalcAcceptor [O,S;-]
Atomtype ChalcAcceptor [o,s;+0]
AtomType HalogenAcceptor [F]
DefineFeature SingleAtomAcceptor [{NAcceptor},{ChalcAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature

# this one is delightfully easy:
DefineFeature AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature

"""

def CalculatePharm2D3pointFingerprint(
    mol, signature_factory="gobi", view=False, gen_3d=True, optim=True, addHs=True
):
    """
    Calculate Pharm2D3point Fingerprints
    """

    if signature_factory == "gobi":
        sigFactory = Gobbi_Pharm2D.factory
    elif signature_factory == "custom":
        featFactory = ChemicalFeatures.BuildFeatureFactoryFromString(fdefstr)  # type: ignore
        sigFactory = SigFactory(
            featFactory, minPointCount=2, maxPointCount=3, trianglePruneBins=True
        )
    else:  # base
        featFactory = ChemicalFeatures.BuildFeatureFactory(  # type: ignore
            os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        )
        sigFactory = SigFactory(
            featFactory, minPointCount=2, maxPointCount=3, trianglePruneBins=True
        )
    sigFactory.SetBins([(0, 3), (3, 6), (6, 9)])
    sigFactory.Init()
    if addHs:
        # remove Hs and add them back
        mol = Chem.AddHs(mol)  # type: ignore
    if gen_3d:
        AllChem.EmbedMolecule(mol)  # type: ignore
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)  # type: ignore
        if optim:
            AllChem.MMFFOptimizeMolecule(mol)  # type: ignore
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)  # type: ignore
    dm = Chem.Get3DDistanceMatrix(mol)  # type: ignore
    res = Generate.Gen2DFingerprint(mol, sigFactory, dMat=dm)
    BitVect = np.array([int(x) for x in list(res.ToBitString())])
    if view:
        for i in list(res.GetOnBits()):
            print(f"{sigFactory.GetBitDescription(i):80}==>{i}")
    return BitVect, res


def calc_energy(mol, conformerId, minimizeIts):
    ff = AllChem.MMFFGetMoleculeForceField(  # type: ignore
        mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId  # type: ignore
    )
    ff.Initialize()
    ff.CalcEnergy()
    results = {}
    if minimizeIts > 0:
        results["converged"] = ff.Minimize(maxIts=minimizeIts)
    results["energy_abs"] = ff.CalcEnergy()
    return results


def cluster_conformers(mol, mode="RMSD", threshold=2.0):
    if mode == "TFD":
        dmat = TorsionFingerprints.GetTFDMatrix(mol)
    else:
        dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
    return Butina.ClusterData(
        dmat,
        mol.GetNumConformers(),
        threshold,
        isDistData=True,
        reordering=True,
    )


def align_conformers(mol, clust_ids):
    rmslist = []
    AllChem.AlignMolConformers(mol, confIds=clust_ids, RMSlist=rmslist)  # type: ignore
    return rmslist


def get_pharmacophore_fp(
    smiles, num_confs=100, view=False, maxIters=100, top=0.5, addHs=True
):
    """Generate pharmacophore fingerprint for a given smiles

    Args:
        smiles (str): SMILES string
        num_confs (int, optional): Number of conformers to Generate. Defaults to 100.
        view (bool, optional): _description_. Defaults to False.
        maxIters (int, optional): _description_. Defaults to 100.
        top (float, optional): _description_. Defaults to 0.2.
        addHs (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    try:
        assert top >= 0, "top must be positive"
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        # sanitize mol
        mol.UpdatePropertyCache(strict=False)
        
        Chem.SanitizeMol(mol)  # type: ignore
        if addHs:
            mol = Chem.AddHs(mol)  # type: ignore
        # AllChem.ETKDGv3()
        ## super slow on large molecules, so inititiating random 3D coords
        ps = AllChem.ETKDGv3()  # type: ignore
        ps.randomSeed = 0xF00D
        ps.useRandomCoords = True
        ps.pruneRmsThresh = 0.15
        # AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=maxIters)
        cid = AllChem.EmbedMultipleConfs(mol, num_confs, ps)  # type: ignore
        all_mol = []
        energy_confs = {}
        for i in range(len(cid)):
            AllChem.MMFFOptimizeMolecule(mol, confId=i)  # type: ignore
            # get conformer
            conformer = Chem.Mol(mol, confId=i)  # type: ignore
            energy = calc_energy(mol, i, 0)
            energy_confs[i] = energy["energy_abs"]
            # print(f"Conformer {i} energy: {energy['energy_abs']}")
            fp, _ = CalculatePharm2D3pointFingerprint(
                conformer, view=view, gen_3d=False, optim=False, addHs=False
            )
            all_mol.append(fp)
            # print(f"Conformer {i} fingerprint: {fp} and {np.array(fp).sum()}")
            # sort conformers based on energy
        sorted_energy = sorted(energy_confs.items(), key=lambda x: x[1])
        # normalize the pharmacophore fingerprint
        sorted_mol = [all_mol[z] for z, _ in sorted_energy]
        # take top % of element from sorted_mol
        sorted_mol = sorted_mol[:: int(1 / top)]
        norm_all_mol = np.array(sorted_mol)
        norm_all_mol = norm_all_mol.mean(axis=0)
        return norm_all_mol.reshape(1, -1)
    except Exception as e:
        print(e)
        return "Error"


def load_core_index(filename):
    with open(filename, "r") as f:
        core_index = [line.strip().split() for line in f]
    return pd.DataFrame(core_index[1:], columns=core_index[0])



if __name__ == "__main__":
    # main()
    # print(get_pharmacophore_fp("CC1=CC=C(C=C1)C2=CC=CC=C2", view=True).shape)
    fire.Fire(get_pharmacophore_fp)
