#!/usr/bin/python3
import PDB_Helper as ph
import NW_Helper as nw
import Scores as scr
import sys
import numpy
import re

#pdbFile1 = ph.ReadPDBAsAtomsList("/home/jerry/Projects/NW-Struct/pdbs/1atz.pdb")
#pdbFile2 = ph.ReadPDBAsAtomsList("/home/jerry/Projects/NW-Struct/pdbs/1auo.pdb")
#chain1 = ph.GetChain(pdbFile1, 'A')
#chain2 = ph.GetChain(pdbFile2, 'A')
#aminos1 = ph.Get_Aminos(chain1)
#aminos2 = ph.Get_Aminos(chain2)
#revTracePath = ph.acc.Aminos_NWAlign(aminos1, aminos2, 10, 10)
#res_1, res_2 = nw.TracePath2OnlyAlignedList(revTracePath, chain1.residues, chain2.residues)
#scr.TMscoreAligned(res_1, res_2, len(res_1))
#
#print()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("\nStruct alignment. Usage: python3 main.py pdb_file_1 pdb_file_2\n")
        exit()

    pdbFile1 = ph.ReadPDBAsAtomsList(sys.argv[1])
    pdbFile2 = ph.ReadPDBAsAtomsList(sys.argv[2])
    #pdbFile1 = ph.ReadPDBAsAtomsList("test_pdbs/1/1whiA.pdb")
    #pdbFile2 = ph.ReadPDBAsAtomsList("test_pdbs/2/1czbA.pdb")
    chain1 = ph.GetChain(pdbFile1, 'A')
    chain2 = ph.GetChain(pdbFile2, 'A')
    aminos1 = ph.Get_Aminos(chain1)
    aminos2 = ph.Get_Aminos(chain2)
    revTracePath = ph.acc.Aminos_NWAlign(aminos1, aminos2, 10, 10)
    res_1, res_2 = nw.TracePath2OnlyAlignedList(revTracePath, chain1.residues, chain2.residues)
    rotMat, transVtr, a, b = scr.TMscoreAligned(res_1, res_2, len(res_1))

    theChain = ph.GetChain(pdbFile1, 'A')
    transAtomList = []
    for residue in theChain.residues:
        for atom in residue.atoms.values():
            transAtomList.append(atom.transform(rotMat.T, transVtr).infoList)

    theSaveFilename = re.findall(r"([^/]*).pdb$", sys.argv[1])[0] + "_2_" + re.findall(r"([^/]*).pdb$", sys.argv[2])[0]
    ph.SavePDBFile('out_pdbs/' + theSaveFilename + "_rotated.pdb", [ph.GetChain(transAtomList, 'A')])