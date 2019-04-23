#import numpy as np
#import TestMod
#a = np.array([[1,2,3],[4,5,6]])
#c = np.array([1,2,3,4,5])
#print(TestMod.test_array(a,a,c))

import PDB_Helper as ph
import NW_Helper as nw
import Scores as scr

pdbFile1 = ph.ReadPDBAsAtomsList("/home/jerry/Projects/NW-Struct/pdbs/1atz.pdb")
pdbFile2 = ph.ReadPDBAsAtomsList("/home/jerry/Projects/NW-Struct/pdbs/1auo.pdb")
chain1 = ph.GetChain(pdbFile1, 'A')
chain2 = ph.GetChain(pdbFile2, 'A')
aminos1 = ph.Get_Aminos(chain1)
aminos2 = ph.Get_Aminos(chain2)
revTracePath = ph.acc.Aminos_NWAlign(aminos1, aminos2, 10, 10)
res_1, res_2 = nw.TracePath2OnlyAlignedList(revTracePath, chain1.residues, chain2.residues)
scr.TMscoreAligned(res_1, res_2, len(res_1))

#start = time.time()
#ph.Accle_SurroundVectorSet(chain1, 4, 12)
#print("\nC implementation = " + str(time.time() - start))
#print()
#
#start = time.time()
#print(chain1.Get_SurroundVectorSet(4, 12)[0])
#print("\nPython implementation = " + str(time.time() - start))
print()
