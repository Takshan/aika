from rdkit import Chem
from gensim.models import word2vec
import numpy as np

from mol2vec.features import (DfVec, MolSentence, mol2alt_sentence,
                              sentences2vec)


class Smi2Vec:
    """Get mol2vec for a list of smiles."""

    def __init__(self, word2vec_path: str) -> None:
        """Initializes Smi2vec.

        Args:
            word2vec_path (str): path to the word2vec model.
        """
        # loading model
        self.word2vec_model = word2vec.Word2Vec.load(word2vec_path)

    def get_mol2vec(self, smiles: str) -> np.ndarray:
        """Converts smiles to mol2vec.

        Args:
            smiles (str): SMILES string.

        Returns:
            np.ndarray: mol2vec.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol_sentence = MolSentence(mol2alt_sentence(mol, 1))
        return DfVec(sentences2vec([mol_sentence], self.word2vec_model))
