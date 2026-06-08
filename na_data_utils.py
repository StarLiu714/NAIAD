import numpy as np
import torch
import itertools

def sample_bernoulli_rv(p):
    """
    Given a probability p, representing the success probability of a Bernoulli
    distribution, sample X ~ Bernoulli(p).

    Arguments:
        p (float): a float between 0 and 1, representing the success probability
            of a Bernoulli distribution.
    
    Returns:
        x (int): the result of sampling the random variable X ~ Bernoulli(p).
            P(X = 1) = p
            P(X = 0) = 1 - p.
    """
    # Check that 0 <= p <= 1.
    if p < 0 or p > 1:
        raise ValueError("The success probability p must be between 0 and 1 inclusive.")
    
    # Handle the edge cases, otherwise utilize the numpy uniform distribution.
    if p == 0:
        x = 0
    elif p == 1:
        x = 1
    else:
        # Sample the Y ~ Uniform(0, 1) distribution.
        uniform_sample = np.random.uniform(0.0, 1.0)

        # P(Y < p) = p.
        if uniform_sample < p:
            x = 1
        else:
            x = 0
    
    return x

def sample_bernoulli_rvs(p, n):
    """
    Given a probability p, representing the success probability of a Bernoulli
    distribution, and a number of samples n, sample X ~ Bernoulli(p) n times
    independently.

    Arguments:
        p (float): a float between 0 and 1, representing the success probability
            of a Bernoulli distribution.
        n (int): the number of samples to draw.
    
    Returns:
        x (np.int32 np.ndarray): an n length array; the result of sampling the 
            random variable X ~ Bernoulli(p) n times.
            P(X = 1) = p
            P(X = 0) = 1 - p.
    """
    # Sample Bernoulli(p) distribution n times.
    x = []
    for i in range(n):
        x.append(sample_bernoulli_rv(p))
    
    # Convert to numpy array.
    x = np.array(x, dtype = np.int32)

    return x

class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 cif_parser,
                 pdb_parser,
                 atom_list_to_save=['N', 'CA', 'C', 'O', #protein atoms
                                    'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'" #nucleic acid atoms
                                   ],
                 parse_protein=1,
                 parse_dna=1,
                 parse_rna=1,
                 parse_rna_as_dna=0,
                 na_shared_tokens=0,
                 protein_backbone_occ_cutoff=0.8,
                 protein_side_chain_occ_cutoff=0.5, 
                 dna_backbone_occ_cutoff=0.8,
                 dna_side_chain_occ_cutoff=0.5,
                 rna_backbone_occ_cutoff=0.8,
                 rna_side_chain_occ_cutoff=0.5,
                 crop_large_structures=0,
                 batch_tokens=6000,
                 na_ref_atom="C1'",
                 drop_protein_probability=0):
        self.protein_backbone_occ_cutoff = protein_backbone_occ_cutoff
        self.protein_side_chain_occ_cutoff = protein_side_chain_occ_cutoff
        self.dna_backbone_occ_cutoff = dna_backbone_occ_cutoff
        self.dna_side_chain_occ_cutoff = dna_side_chain_occ_cutoff
        self.rna_backbone_occ_cutoff = rna_backbone_occ_cutoff
        self.rna_side_chain_occ_cutoff = rna_side_chain_occ_cutoff

        self.parse_protein = parse_protein
        self.parse_dna = parse_dna
        self.parse_rna = parse_rna
        self.parse_rna_as_dna = parse_rna_as_dna
        self.na_shared_tokens = na_shared_tokens

        self.crop_large_structures = crop_large_structures
        self.batch_tokens = batch_tokens
        self.na_ref_atom = na_ref_atom

        self.drop_protein_probability = drop_protein_probability

        self.atom_list_to_save = atom_list_to_save

        self.num_atoms_to_save = len(self.atom_list_to_save)

        self.atom_dict = dict(zip(self.atom_list_to_save, range(self.num_atoms_to_save)))

        self.cif_parser = cif_parser
        self.pdb_parser = pdb_parser

        self.polytypes = [
            'PP',
            'DNA',
            'RNA',
            'UNK',
            'MAS',
            'PAD'
        ]

        self.polytype_to_int = dict(zip(self.polytypes, range(len(self.polytypes))))

        if self.parse_rna_as_dna:
            self.polytype_to_int["RNA"] = self.polytype_to_int["DNA"]

        self.restypes = [
            'ALA',
            'ARG',
            'ASN',
            'ASP',
            'CYS',
            'GLN',
            'GLU',
            'GLY',
            'HIS',
            'ILE',
            'LEU',
            'LYS',
            'MET',
            'PHE',
            'PRO',
            'SER',
            'THR',
            'TRP',
            'TYR',
            'VAL',
            'UNK',
            'DA',
            'DC',
            'DG',
            'DT',
            'DX',
            'A',
            'C',
            'G',
            'U',
            'RX',
            'MAS',
            'PAD'
        ]

        self.protein_restypes = [
            'ALA',
            'ARG',
            'ASN',
            'ASP',
            'CYS',
            'GLN',
            'GLU',
            'GLY',
            'HIS',
            'ILE',
            'LEU',
            'LYS',
            'MET',
            'PHE',
            'PRO',
            'SER',
            'THR',
            'TRP',
            'TYR',
            'VAL',
            'UNK'
        ]

        self.dna_restypes = [
            'DA',
            'DC',
            'DG',
            'DT',
            'DX'
        ]

        self.rna_restypes = [
            'A',
            'C',
            'G',
            'U',
            'RX'
        ]

        self.unknown_restypes = [
            "UNK",
            "DX",
            "RX"
        ]

        self.num_protein_restypes = len(self.protein_restypes)
        self.num_dna_restypes = len(self.dna_restypes)
        self.num_rna_restypes = len(self.rna_restypes)

        self.restype_3_to_1 = {
            'ALA': 'A', 
            'ARG': 'R', 
            'ASN': 'N', 
            'ASP': 'D', 
            'CYS': 'C', 
            'GLN': 'Q', 
            'GLU': 'E', 
            'GLY': 'G', 
            'HIS': 'H', 
            'ILE': 'I', 
            'LEU': 'L', 
            'LYS': 'K', 
            'MET': 'M', 
            'PHE': 'F', 
            'PRO': 'P', 
            'SER': 'S', 
            'THR': 'T', 
            'TRP': 'W', 
            'TYR': 'Y', 
            'VAL': 'V',
            'UNK': 'X',
            'DA': 'a',
            'DC': 'c',
            'DG': 'g',
            'DT': 't',
            'DX': 'x',
            'A': 'b',
            'C': 'd',
            'G': 'h',
            'U': 'u',
            'RX': 'y',
            'MAS': '-',
            'PAD': '+'
        }

        self.restype_to_int = dict(zip(self.restypes, range(len(self.restypes))))
        self.int_to_restype = dict(zip(range(len(self.restypes)), self.restypes))

        if self.parse_rna_as_dna or self.na_shared_tokens:
            self.restype_to_int["A"] = self.restype_to_int["DA"]
            self.restype_to_int["C"] = self.restype_to_int["DC"]
            self.restype_to_int["G"] = self.restype_to_int["DG"]
            self.restype_to_int["U"] = self.restype_to_int["DT"]
            self.restype_to_int["RX"] = self.restype_to_int["DX"]
        
        self.protein_restype_ints = list(map(lambda x: self.restype_to_int[x], self.protein_restypes))
        self.dna_restype_ints = list(map(lambda x: self.restype_to_int[x], self.dna_restypes))
        self.rna_restype_ints = list(map(lambda x: self.restype_to_int[x], self.rna_restypes))
        self.unknown_restype_ints = list(map(lambda x: self.restype_to_int[x], self.unknown_restypes))

        self.na_canonical_base_pair_restypes = [
            ('DA', 'DT'), 
            ('DA', 'U'), 
            ('DC', 'DG'), 
            ('DC', 'G'), 
            ('DG', 'DC'), 
            ('DG', 'C'), 
            ('DT', 'DA'), 
            ('DT', 'A'), 
            ('A', 'DT'), 
            ('A', 'U'), 
            ('C', 'DG'), 
            ('C', 'G'), 
            ('G', 'DC'), 
            ('G', 'C'), 
            ('U', 'DA'), 
            ('U', 'A')
        ]

        self.na_canonical_base_pair_ints = []
        for (restype_i, restype_j) in self.na_canonical_base_pair_restypes:
            self.na_canonical_base_pair_ints.append((self.restype_to_int[restype_i], 
                                                     self.restype_to_int[restype_j]))

        self.protein_bb_idx_list = []
        self.dna_bb_idx_list = []
        self.rna_bb_idx_list = []

        self.protein_backbone_list = ["N", "CA", "C", "O"]
        self.dna_backbone_list = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"]
        self.rna_backbone_list = ['OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

        for atom in self.atom_list_to_save:
            if atom in self.protein_backbone_list:
                self.protein_bb_idx_list.append(self.atom_dict[atom])

        for atom in self.atom_list_to_save:
            if atom in self.dna_backbone_list:
                self.dna_bb_idx_list.append(self.atom_dict[atom])
        
        for atom in self.atom_list_to_save:
            if atom in self.rna_backbone_list:
                self.rna_bb_idx_list.append(self.atom_dict[atom])

    def __getitem__(self, index):
        """
        index = [[(example_dict, assembly_id), (example_dict, assembly_id)]]
        """
        x = [self.loader(example_dict, assembly_id) for (example_dict, assembly_id) in index[0]]
        return x

    def parse_structure(self, structure_path):
        if structure_path[-4:] == ".pdb" or structure_path[-7:] == ".pdb.gz":
            return self.pdb_parser.parse(structure_path)
        elif structure_path[-4:] == ".cif" or structure_path[-7:] == ".cif.gz":
            return self.cif_parser.parse(structure_path)
        else:
            raise Exception(f"{structure_path}: Unknown structure path extension.")

    def load_chains(self, chains):
        #------------------proteins vs not--------------------
        macromolecule_letter_list = []
        for chain_letter, chain in chains.items():
            if chain.type == "polypeptide(L)":
                macromolecule_letter_list.append(chain_letter)
            elif chain.type == "polydeoxyribonucleotide":
                macromolecule_letter_list.append(chain_letter)
            elif chain.type == "polyribonucleotide":
                macromolecule_letter_list.append(chain_letter)
            elif chain.type == "polydeoxyribonucleotide/polyribonucleotide hybrid":
                macromolecule_letter_list.append(chain_letter)
  
        macromolecule_chain_dict = {}

        for letter in macromolecule_letter_list:
            chain = chains[letter]

            macromolecule_chain_dict[letter] = {}

            macromolecule_chain_dict[letter]["type"] = chain.type
        
            L = len(list(set([a[1] for a in list(chain.atoms.keys())])))
            xyz = np.zeros([L, self.num_atoms_to_save, 3], dtype=np.float32)
            occ = np.zeros([L, self.num_atoms_to_save], dtype=np.float32)
            residue_idx = -100*np.ones([L], dtype=np.int32) 
            raw_sequence = L*["UNK"]
            for c, (res_id, res_atoms) in enumerate(itertools.groupby(list(chain.atoms.keys()), lambda x: x[1])):
                for atom_key in res_atoms:
                    _, res_idx_str, res_name, atom_name = atom_key
                    
                    if atom_name in self.atom_dict:
                        atom_idx = self.atom_dict[atom_name]
                        xyz[c,atom_idx,:] = np.array(chain.atoms[atom_key].xyz)
                        occ[c,atom_idx] = np.array(chain.atoms[atom_key].occ)
                        # Same for each atom in the residue.
                        raw_sequence[c] = res_name
                        residue_idx[c] = int(res_idx_str)
            
            macromolecule_chain_dict[letter]["xyz"] = xyz
            macromolecule_chain_dict[letter]["occ"] = occ
            macromolecule_chain_dict[letter]["seq"] = raw_sequence
            macromolecule_chain_dict[letter]["residue_idx"] = residue_idx
        
        return macromolecule_chain_dict

    def load_assembly(self, macromolecule_chain_dict, asmb, assembly_id):
        X_list = []
        protein_mask_list = []
        dna_mask_list = []
        rna_mask_list = []
        X_occ_list = []
        S_list = []
        R_idx_list = []
        chain_labels_list = []
        chain_multi = 0
        
        for i, (letter, transform_matrix) in enumerate(asmb[assembly_id]):
            if letter in macromolecule_chain_dict:
                xyz = macromolecule_chain_dict[letter]["xyz"]
                rotation_matrix = transform_matrix[:3,:3]
                translation = transform_matrix[:3,3]
                xyz = np.einsum('ij,raj->rai', rotation_matrix, xyz) + translation[None,None,:]
                X_list.append(xyz)

                X_occ_list.append(macromolecule_chain_dict[letter]["occ"])

                R_idx_list.append(macromolecule_chain_dict[letter]["residue_idx"])

                chain_labels_list.append(chain_multi*np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32))
                chain_multi += 1

                protein_mask = np.zeros_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                dna_mask = np.zeros_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                rna_mask = np.zeros_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                if macromolecule_chain_dict[letter]["type"] == "polypeptide(L)":
                    unknown_residue = "UNK"
                    protein_mask = np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                elif macromolecule_chain_dict[letter]["type"] == "polydeoxyribonucleotide":
                    unknown_residue = "DX"
                    dna_mask = np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32) 
                elif macromolecule_chain_dict[letter]["type"] == "polyribonucleotide":
                    unknown_residue = "RX"
                    rna_mask = np.ones_like(macromolecule_chain_dict[letter]["residue_idx"], dtype=np.int32)
                elif macromolecule_chain_dict[letter]["type"] == "polydeoxyribonucleotide/polyribonucleotide hybrid":
                    # Note, unknown residues in DNA/RNA hybrid chains are
                    # excluded from the DNA and RNA masks (it is not possible
                    # to know if a residue is one or the other due to the
                    # possibility of missing atoms). As such, the choice of
                    # unknown residue for the sequence does not matter.
                    unknown_residue = "DX"
                    for i, AA in enumerate(macromolecule_chain_dict[letter]["seq"]):
                        if AA in self.dna_restypes:
                            dna_mask[i] = 1
                        elif AA in self.rna_restypes:
                            rna_mask[i] = 1
                
                protein_mask_list.append(protein_mask)
                dna_mask_list.append(dna_mask)
                rna_mask_list.append(rna_mask)
                    
                seq_int = [self.restype_to_int.get(AA, self.restype_to_int[unknown_residue]) for AA in macromolecule_chain_dict[letter]["seq"]]
                S_list.append(np.array(seq_int, dtype=np.int32))

        X = np.concatenate(X_list, axis = 0) #[L, num_atoms, 3]
        X_occ = np.concatenate(X_occ_list, axis = 0) #[L, num_atoms]
        R_idx = np.concatenate(R_idx_list, axis = 0) #[L]
        chain_labels = np.concatenate(chain_labels_list, axis = 0) #[L]
        protein_mask = np.concatenate(protein_mask_list, axis = 0) #[L]
        dna_mask = np.concatenate(dna_mask_list, axis = 0) #[L]
        rna_mask = np.concatenate(rna_mask_list, axis = 0) #[L]
        S = np.concatenate(S_list, axis = 0) #[L]

        R_polymer_type = protein_mask * self.polytype_to_int["PP"] + \
                    dna_mask * self.polytype_to_int["DNA"] + \
                    rna_mask * self.polytype_to_int["RNA"] + \
                    (1 - protein_mask - dna_mask - rna_mask) * self.polytype_to_int["UNK"]

        side_chain_occ_cutoff = protein_mask * self.protein_side_chain_occ_cutoff + \
                                dna_mask * self.dna_side_chain_occ_cutoff + \
                                rna_mask * self.rna_side_chain_occ_cutoff

        X_m = (X_occ > side_chain_occ_cutoff[:, None]).astype(np.int32)

        backbone_occ_cutoff = protein_mask * self.protein_backbone_occ_cutoff + \
                              dna_mask * self.dna_backbone_occ_cutoff + \
                              rna_mask * self.rna_backbone_occ_cutoff
        
        # Protein, DNA, and RNA masks are updated to only include residues with
        # all backbone atoms.
        X_occ_mask = (X_occ > backbone_occ_cutoff[:, None]).astype(np.int32)
        protein_mask = protein_mask * (np.prod(X_occ_mask[:, self.protein_bb_idx_list], axis = -1))
        dna_mask = dna_mask * (np.prod(X_occ_mask[:, self.dna_bb_idx_list], axis = -1))
        rna_mask = rna_mask * (np.prod(X_occ_mask[:, self.rna_bb_idx_list], axis = -1))

        if self.parse_rna_as_dna:
            dna_mask = np.bitwise_or(dna_mask, rna_mask)
            rna_mask = np.zeros_like(dna_mask)

        mask_for_output = np.zeros_like(protein_mask)
        out_dict = {}

        if self.parse_protein:
            mask_for_output = np.bitwise_or(mask_for_output, protein_mask)
            out_dict["protein_L"] = np.count_nonzero(protein_mask)
        else:
            out_dict["protein_L"] = 0
        
        if self.parse_dna:
            mask_for_output = np.bitwise_or(mask_for_output, dna_mask)
            out_dict["dna_L"] = np.count_nonzero(dna_mask)
        else:
            out_dict["dna_L"] = 0
        
        if self.parse_rna:
            mask_for_output = np.bitwise_or(mask_for_output, rna_mask)
            out_dict["rna_L"] = np.count_nonzero(rna_mask)
        else:
            out_dict["rna_L"] = 0
        
        out_dict["macromolecule_L"] = np.count_nonzero(mask_for_output)

        mask_for_output = mask_for_output.astype(bool)

        out_dict["protein_mask"] = protein_mask[mask_for_output]
        out_dict["dna_mask"] = dna_mask[mask_for_output]
        out_dict["rna_mask"] = rna_mask[mask_for_output]

        out_dict["X"] = X[mask_for_output]
        out_dict["X_m"] = X_m[mask_for_output]

        out_dict["S"] = S[mask_for_output]

        out_dict["R_idx"] = R_idx[mask_for_output]

        out_dict["chain_labels"] = chain_labels[mask_for_output]
        out_dict["R_polymer_type"] = R_polymer_type[mask_for_output]

        return out_dict

    def load_preprocessed_data(self, out_dict, example_dict, assembly_id):
        """
        Load any preprocessed data for the given example, specified by the
        example_dict and assembly id.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
            example_dict (dict): containing a dictionary that represents the
                column to value mapping of an example (a row from a dataframe).
            assembly_id (str): the id that specifies the assembly; needed for
                indexing into the assembly dictionaries in example_dict.
        
        Side Effects:
            out_dict['interface_mask']: the precomputed protein-nucleic acid
                interface mask.
            out_dict['side_chain_interface_mask']: the precomputed protein side
                chain-nucleic acid side chain interface mask.
            out_dict['nearest_protein_side_chain_index']: the precomputed index
                of the nearest protein side chain for each nucleic acid residue.
            out_dict['base_pair_mask']: the precomputed base pairing mask for
                nucleic acid residues.
            out_dict['base_pair_index']: the precomputed index of the base
                pairing partner for residues that are marked in the
                base_pair_mask.
            out_dict['canonical_base_pair_mask']: the precomputed canonical base
                pairing mask for nucleic acid residues.
            out_dict['canonical_base_pair_index']: the precomputed index of the
                canonical base pairing partner for residues that are marked in
                the canonical_base_pair_mask.
        """
        out_dict["interface_mask"] = \
            np.load(example_dict["asmb_interface_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["side_chain_interface_mask"] = \
            np.load(example_dict["asmb_side_chain_interface_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["nearest_protein_side_chain_index"] = \
            np.load(example_dict["asmb_nearest_protein_side_chain_index_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int64)
        out_dict["base_pair_mask"] = \
            np.load(example_dict["asmb_base_pair_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["base_pair_index"] = \
            np.load(example_dict["asmb_base_pair_index_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int64)
        out_dict["canonical_base_pair_mask"] = \
            np.load(example_dict["asmb_canonical_base_pair_masks_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int32)
        out_dict["canonical_base_pair_index"] = \
            np.load(example_dict["asmb_canonical_base_pair_index_path"], 
                    allow_pickle = True).item()[assembly_id].astype(np.int64)

    def apply_crop_mask(self, out_dict, mask_to_keep):
        """
        Given a dictionary containing the loaded data for a biomolecule, and
        a mask of which residues to keep, crop all of the arrays of loaded
        data. For features that represent array indices, the indices need to
        be adjusted for the removed residues, and the associated masks need
        to be updated if the indices point to removed residues.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.
            mask_to_keep (bool np.ndarray): a mask indicating which residues
                to keep when cropping. True at positions to keep, and False
                otherwise.

        Side Effects:
            out_dict[k]: cropped to only include the residues indicated by
                mask_to_keep, if k is an np.ndarray. If k denotes one of the
                index features, adjust the index for the removed residues, and
                if the index points to a removed residue, update the associated
                mask. If k denotes a macromolecule length, recalculate.
        """
        # Crop the loaded data.
        for k in out_dict:
            if type(out_dict[k]) == np.ndarray:
                out_dict[k] = out_dict[k][mask_to_keep]

        # Update variables that represent indices and associated masks.
        mask_to_remove = np.logical_not(mask_to_keep)
        index_of_removed = np.where(mask_to_remove)[0]
        residues_removed_to_left = np.array([0] + list(np.add.accumulate(mask_to_remove.astype(np.int32))[:-1]), dtype = np.int64)

        index_and_mask_key_pairs = [
            ("base_pair_index", "base_pair_mask"),
            ("canonical_base_pair_index", "canonical_base_pair_mask"),
            ("nearest_protein_side_chain_index", "side_chain_interface_mask")
        ]
        for (index_key, mask_key) in index_and_mask_key_pairs:
            index_in_removed = np.isin(out_dict[index_key], index_of_removed)

            # If the index that is pointed to was removed, mark the
            # corresponding position in the mask as 0.
            out_dict[mask_key][index_in_removed] = 0

            # For the indices that remain, subtract the residues removed to the
            # left of the position indicated by the index.
            out_dict[index_key] = out_dict[index_key] - residues_removed_to_left[out_dict[index_key]]
            out_dict[index_key] = out_dict[index_key] * out_dict[mask_key]

        # Update length data.
        out_dict["protein_L"] = np.count_nonzero(out_dict["protein_mask"])
        out_dict["dna_L"] = np.count_nonzero(out_dict["dna_mask"])
        out_dict["rna_L"] = np.count_nonzero(out_dict["rna_mask"])
        out_dict["macromolecule_L"] = out_dict["protein_L"] + out_dict["dna_L"] + out_dict["rna_L"]

    def drop_protein(self, out_dict):
        """
        Given a dictionary containing the loaded data for a biomolecule,
        drop all protein residues with a certain probability, dictated by
        self.drop_protein_probability.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
            biomolecule.

        Side Effects:
            out_dict[k]: crop to remove any protein residues. Set the interface
                masks to zero.
        """
        if sample_bernoulli_rv(self.drop_protein_probability) == 1:
            # Crop out the protein.
            not_protein_mask = np.logical_not(out_dict["protein_mask"] == 1)
            self.apply_crop_mask(out_dict, not_protein_mask)
            
            # Zero out the interface masks.
            out_dict["interface_mask"] = np.zeros_like(out_dict["interface_mask"])
            out_dict["side_chain_interface_mask"] = np.zeros_like(out_dict["side_chain_interface_mask"])

    def random_crop_na(self, out_dict):
        """
        Given a dictionary containing the loaded information of a biomolecule,
        crop the structure spatially around a randomly selected nucleic acid
        residue to the number of tokens in a batch.

        Arguments:
            out_dict (dict): dictionary containing the loaded data for a
                biomolecule.

        Side Effects:
            out_dict[k]: crop the to the randomly selected, batch-sized spatial
                crop.
        """
        X = out_dict["X"]
        dna_mask = out_dict["dna_mask"]
        rna_mask = out_dict["rna_mask"]
        CA_idx = self.atom_dict["CA"]
        na_ref_atom_idx = self.atom_dict[self.na_ref_atom]

        ref_atom_X = X[:,CA_idx,:] + X[:,na_ref_atom_idx,:]

        # Choose a random nucleic acid to crop around.
        na_mask = dna_mask + rna_mask
        na_res_index = np.random.choice(np.where(na_mask == 1)[0])
        
        # Compute distance to all other residues.
        distance_to_na_res = np.sqrt(np.sum((ref_atom_X - ref_atom_X[na_res_index,:]) ** 2, axis = -1))
        argsorted_distance = np.argsort(distance_to_na_res)
        idx_to_keep = argsorted_distance[:self.batch_tokens]

        # Crop all array data.
        mask_to_keep = np.zeros_like(out_dict["S"], dtype=np.bool_)
        mask_to_keep[idx_to_keep] = True
        self.apply_crop_mask(out_dict, mask_to_keep)

    def loader(self, example_dict, assembly_id):
        try:
            chains, asmb, covalei, meta = self.parse_structure(example_dict["structure_path"])
        except:
            print('bad_structure: ', example_dict["structure_path"])
            return ("pass", "pass")
        
        if assembly_id not in list(asmb.keys()):
            print('bad_assembly_id: ', example_dict["structure_path"], assembly_id)
            return ("pass", "pass")

        macromolecule_chain_dict = self.load_chains(chains)

        out_dict = self.load_assembly(macromolecule_chain_dict, asmb, assembly_id)

        self.load_preprocessed_data(out_dict, example_dict, assembly_id)

        # Drop the protein with some probability.
        if self.drop_protein_probability > 0 and out_dict["macromolecule_L"] > out_dict["protein_L"]:
            self.drop_protein(out_dict)

        # Crop structures that are larger than the number of tokens in a batch.
        if self.crop_large_structures and out_dict["macromolecule_L"] > self.batch_tokens:
            self.random_crop_na(out_dict)
        
        out_dict["structure_path"] = example_dict["structure_path"]
        out_dict["assembly_id"] = assembly_id

        return (out_dict, out_dict["macromolecule_L"])
    
    def load_for_structure_preprocessing(self, example_dict):
        try:
            chains, asmb, covalei, meta = self.parse_structure(example_dict["structure_path"])
        except:
            print('bad_structure: ', example_dict["structure_path"])
            return ("pass", "pass")

        # Save the per-chain sequences, for clustering purposes.
        chain_sequences = []
        for chain_letter in chains:
            chain = chains[chain_letter]
            chain_sequences.append((chain.id, chain.type, chain.sequence))

        macromolecule_chain_dict = self.load_chains(chains)

        assemblies = []
        for assembly_id in list(asmb.keys()):
            out_dict = self.load_assembly(macromolecule_chain_dict, asmb, assembly_id)
            assemblies.append((assembly_id, out_dict))
        
        return assemblies, chain_sequences

class StructureLoader():
    def __init__(self, dataset, macromolecule_lengths, max_tokens_per_batch):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = macromolecule_lengths
        self.max_tokens_per_batch = max_tokens_per_batch
        sorted_ix = np.argsort(self.lengths)
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size > self.max_tokens_per_batch:
                continue

            if size * (len(batch) + 1) <= self.max_tokens_per_batch:
                batch.append(ix)
            else:
                if len(batch) > 0:
                    clusters.append(batch)
                batch = [ix]
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def make_batch_iter(df, batch_tokens, length_cutoff, date_cutoff, crop_large_structures, max_number_of_pdbs):
    """
    Creates an iterable batch for training or validation.

    Arguments:
        df (pd.DataFrame): a pandas DataFrame containing all information
            needed to load an example.
        batch_tokens (int): the maximum number of tokens (residues) in a batch.
        length_cutoff (int): the minimum macromolecule length for examples.
        date_cutoff (pd.DateTime): the date cutoff used for sampling; all
            samples past the date cutoff will be excluded.
        crop_large_structures (bool): if True, this indicates that large
            structures will be cropped to the batch size. If False, large
            structures (with more residues than the number of tokens in a batch)
            will be excluded.
        max_number_of_pdbs (int): the maximum number of PDBs to include in a
            batch.
    
    Returns:
        batch_iter (list_iterator): an iterable containing the row dictionaries
            for the samples in the batch.
    """
    samples=[]
    random_permutation = list(np.random.permutation(len(df)))
    for i in random_permutation:
        example_dict = df.iloc[i].to_dict()
        
        cluster_probability = example_dict["sampling_probability"]

        if (sample_bernoulli_rv(cluster_probability) == 1) and \
           (example_dict["date"] < date_cutoff): 
            samples.append(example_dict)

    L_list = []
    name_list = []
    for example_dict in samples:
        # Assembly ID to length dictionary.
        asmb_lengths_dict = np.load(example_dict["asmb_lengths_path"], allow_pickle = True).item()
        assembly_id_list = list(asmb_lengths_dict.keys())

        num_assemblies = len(assembly_id_list)
        if num_assemblies > 1:
            idx = np.random.randint(0, high = num_assemblies, dtype = int)
        else:
            idx = 0
        assembly_id = assembly_id_list[idx]

        (macromolecule_L, protein_L, dna_L, rna_L) = asmb_lengths_dict[assembly_id]

        if macromolecule_L >= length_cutoff and len(L_list) < max_number_of_pdbs:
            if macromolecule_L > batch_tokens and crop_large_structures and (dna_L + rna_L) > 0:
                macromolecule_L = batch_tokens
            L_list.append(macromolecule_L)
            name_list.append((example_dict, assembly_id))

    structure_loader = StructureLoader(name_list, L_list, max_tokens_per_batch=batch_tokens)

    batch_iter = []
    for _, batch in enumerate(structure_loader):
        batch_iter.append(batch)
    batch_iter = iter(batch_iter)
    return batch_iter
