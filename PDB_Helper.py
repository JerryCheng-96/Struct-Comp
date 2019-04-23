from Vector_Matrix import *
import matplotlib.pyplot as plt
import math
import CAccel as acc


class Atom:
    def __init__(self, infoList):
        self.infoList = infoList

    @property
    def vector(self):
        return np.array([float(val) for val in self.infoList[6:9]])

    @property
    def type(self):
        return self.infoList[2]

    @property
    def n(self):
        return int(self.infoList[1])

    def transform(self, rotMat, transVtr):
        vtr = np.dot(self.vector, rotMat) + transVtr
        self.infoList[6] = '%.3f' % vtr[0]
        self.infoList[7] = '%.3f' % vtr[1]
        self.infoList[8] = '%.3f' % vtr[2]
        return self


class Residue:
    def __init__(self, atomDict, aminoName):
        self.atoms = atomDict
        self.aminoAcid = aminoName


class Chain:
    def __init__(self, residueList):
        self.residues = residueList

    @property
    def AtomsList(self):
        atomsList = []
        for residue in self.residues:
            atomsList += residue.atoms.values()
        return atomsList

    @property
    def CAAtomsList(self):
        return [atom for atom in self.AtomsList if atom.type == 'CA']

    def __len__(self):
        return len(self.residues)


    # TODO: Implement the window function
    def Get_AminoNVtr(self, window = 1):
        assert window % 2 == 1
        offset = window
        listNVtr = [self.residues[1].atoms['N'].vector - self.residues[0].atoms['N'].vector]

        for i in range(1, len(self.residues) - 1):
            vtr_NC = self.residues[i].atoms['N'].vector - self.residues[i - 1].atoms['C'].vector
            vtr_NCa = self.residues[i].atoms['N'].vector - self.residues[i].atoms['CA'].vector
            norm_vtr = np.cross(vtr_NC, vtr_NCa)
            norm_vtr = norm_vtr / Len_Vtr(norm_vtr)
            axis, angle = SolveFor_rot(norm_vtr, np.array([0,0,1]))
            oriNVtr = self.residues[i].atoms['N'].vector - self.residues[i + 1].atoms['N'].vector
            listNVtr.append(Rodrigues_rot(oriNVtr, axis, angle))

        listNVtr.append(np.array([0,0,0]))
        return listNVtr

    def Get_CAContactMap(self):
        contactMap = np.empty((len(self), len(self)))
        for i in range(0, len(self)):
            for j in range(i, len(self)):
                contactMap[i][j] = Len_Vtr(self.residues[i].atoms['CA'].vector - self.residues[j].atoms['CA'].vector)
        return contactMap + contactMap.T


    #def Get_SurroundVectorSet(self, low_cutoff, high_cutoff, keep=10):
    #    neighborAminos = []

    #    for i in range(0, len(self)):
    #        residueLenAminos = {}

    #        vtr_CaN = self.residues[i].atoms['CA'].vector - self.residues[i].atoms['N'].vector
    #        vtr_CaC = self.residues[i].atoms['CA'].vector - self.residues[i].atoms['C'].vector
    #        norm_vtr = np.cross(vtr_CaN, vtr_CaC)
    #        norm_vtr = norm_vtr / Len_Vtr(norm_vtr)
    #        axis, angle = SolveFor_rot(norm_vtr, np.array([0,0,1]))

    #        for j in range(0, len(self)):
    #            theVtr = self.residues[i].atoms['CA'].vector - self.residues[j].atoms['CA'].vector
    #            theVtrLen = Len_Vtr(theVtr)
    #            if low_cutoff < theVtrLen < high_cutoff:
    #                residueLenAminos[Len_Vtr(theVtr)] = (theVtr, j)

    #        residueSurroundVtr = [(Rodrigues_rot(residueLenAminos[aaLen][0], axis, angle), residueLenAminos[aaLen][1], Len_Vtr(residueLenAminos[aaLen][0]))
    #                              for aaLen in sorted(residueLenAminos.keys())][:keep]
    #        neighborAminos.append(residueSurroundVtr)

    #    return neighborAminos


    def Get_SurroundVectorSet(self, low_cutoff, high_cutoff, keep=1000):
        neighborAminos = []

        for i in range(0, len(self)):
            residueLenAminos = {}

            vtr_CaN = self.residues[i].atoms['CA'].vector - self.residues[i].atoms['N'].vector
            vtr_CaC = self.residues[i].atoms['CA'].vector - self.residues[i].atoms['C'].vector

            for j in range(0, len(self)):
                theVtr = self.residues[j].atoms['CA'].vector - self.residues[i].atoms['CA'].vector
                theVtrLen = Len_Vtr(theVtr)
                if low_cutoff < theVtrLen < high_cutoff:
                    bondAngle = math.acos(np.dot(vtr_CaN, theVtr) / (Len_Vtr(vtr_CaN) * Len_Vtr(theVtr)))
                    #bondAngle = (np.dot(vtr_CaN, theVtr) / (Len_Vtr(vtr_CaN) * Len_Vtr(theVtr)))
                    plnNorm1 = np.cross(vtr_CaN, theVtr)
                    plnNorm2 = np.cross(vtr_CaN, vtr_CaC)
                    torsionAngle = math.acos(np.dot(plnNorm1, plnNorm2) / (Len_Vtr(plnNorm1) * Len_Vtr(plnNorm2)))
                    #torsionAngle = (np.dot(plnNorm1, plnNorm2) / (Len_Vtr(plnNorm1) * Len_Vtr(plnNorm2)))
                    residueLenAminos[theVtrLen] = (theVtrLen, bondAngle, torsionAngle)#, self.residues[j])
            residueSurroundVtr = [(residueLenAminos[aaLen])
                                  #for aaLen in sorted(residueLenAminos.keys())][:keep]
                                  for aaLen in (residueLenAminos.keys())][:keep]
            neighborAminos.append(residueSurroundVtr)

        return neighborAminos


def Accle_SurroundVectorSet(chain, low_cutoff, high_cutoff, keep=1000):
    aminosAtomsCoord = []
    for i in range(0, len(chain)):
        theAminoAtoms = []
        theAminoAtoms.append(chain.residues[i].atoms['N'].vector)
        theAminoAtoms.append(chain.residues[i].atoms['CA'].vector)
        theAminoAtoms.append(chain.residues[i].atoms['C'].vector)
        aminosAtomsCoord.append(theAminoAtoms)
    aminosAtomsCoord = np.array(aminosAtomsCoord)

    #print(aminosAtomsCoord[0])
    #print(aminosAtomsCoord[3])
    #print(acc.C_SurroundVectorSet(aminosAtomsCoord, low_cutoff, high_cutoff)[0])
    #acc.C_SurroundVectorSet(aminosAtomsCoord, low_cutoff, high_cutoff)
    #print(np.cross(aminosAtomsCoord[0][0], aminosAtomsCoord[0][2]))

    return acc.C_SurroundVectorSet(aminosAtomsCoord, low_cutoff, high_cutoff)
    #return aminosAtomsCoord


def Get_NeighborAminoNo(contactMap, low_cutoff, high_cutoff):
    neighborVector = []

    for i in range(0, len(contactMap)):
        theResidueNeighbor = []
        for j in range(0, len(contactMap)):
            if low_cutoff < contactMap[i][j] <= high_cutoff:
                theResidueNeighbor.append(j)
        neighborVector.append(theResidueNeighbor)

    return neighborVector


def ReadPDBAsAtomsList(filename):
    f = open(filename, 'r')
    content = f.readlines()
    atomLines = [theLine for theLine in content if theLine[:4] == "ATOM"]
    pdbAtomsList = []

    for line in atomLines:
        theAtom = []
        theAtom.append(line[:4])
        theAtom.append(line[4:11].strip())
        theAtom.append(line[11:17].strip())
        theAtom.append(line[17:20].strip())
        theAtom.append(line[20:22].strip())
        theAtom.append(line[22:26].strip())
        theAtom.append(line[26:38].strip())
        theAtom.append(line[38:46].strip())
        theAtom.append(line[46:54].strip())
        theAtom.append(line[54:60].strip())
        theAtom.append(line[60:66].strip())
        theAtom.append(line[66:80].strip())
        pdbAtomsList.append(theAtom)

    return pdbAtomsList


def FormatAtomLine(atomInfoList):
    return '%s%7s'%(atomInfoList[0], atomInfoList[1]) + 2 * ' ' + atomInfoList[2].ljust(4) + atomInfoList[3].ljust(4) + \
           atomInfoList[4] + atomInfoList[5].rjust(4) + atomInfoList[6].rjust(12) + atomInfoList[7].rjust(8) + \
           atomInfoList[8].rjust(8) + atomInfoList[9].rjust(6) + atomInfoList[10].rjust(6) + ' ' * 11 + \
           atomInfoList[11].ljust(3) + '\n'


def SavePDBFile(filename, chainList):
    f = open(filename, 'w')
    pdbFileLines = []
    atomsDict = {}

    for chain in chainList:
        for residue in chain.residues:
            for atom in residue.atoms.values():
                atomsDict[atom.n] = atom
        pdbFileLines += [FormatAtomLine(atomsDict[atomNo].infoList) + '\n' for atomNo in sorted(atomsDict.keys())]
        pdbFileLines += "TER\n"

    pdbFileLines += 'END\n'
    f.writelines(pdbFileLines)
    f.flush()
    f.close()


def GetChain(pdbAtomsList, chainId):
    atomsList = [atom for atom in pdbAtomsList if atom[4] == chainId]
    pdbAminoDict = {}

    for atom in atomsList:
        try:
            pdbAminoDict[int(atom[5])].atoms[atom[2]] = Atom(atom)
        except KeyError:
            pdbAminoDict[int(atom[5])] = Residue({}, atom[3])
            pdbAminoDict[int(atom[5])].atoms[atom[2]] = Atom(atom)
#    for i in range(1, 395):
#        try:
#            pdbAminoDict[i] == None
#        except KeyError:
#            print(i)

    return Chain([pdbAminoDict[aminoNo] for aminoNo in sorted(pdbAminoDict.keys())])


def Get_Aminos(chain, low_cutoff = 4, high_cutoff = 12):
    aminoNVtr = chain.Get_AminoNVtr()
    neighborVtr = Accle_SurroundVectorSet(chain, low_cutoff, high_cutoff)

    assert len(aminoNVtr) == len(neighborVtr)
    aminosList = []
    for i in range(0, len(aminoNVtr)):
        aminosList.append((aminoNVtr[i], neighborVtr[i]))

    return aminosList


def Dbg_GetVtrLen(chain, aaNo1, aaNo2):
    theVtr = chain.CAAtomsList[aaNo1].vector - chain.CAAtomsList[aaNo2].vector
    return theVtr, Len_Vtr(theVtr)


if __name__ == "__main__":
    pdbAtomsList1 = ReadPDBAsAtomsList("pdbs/Legacy/101M.pdb")
    pdbAtomsList2 = ReadPDBAsAtomsList("pdbs/Legacy/1MBA_New.pdb")
    chain1 = GetChain(pdbAtomsList1, 'A')
    chain2 = GetChain(pdbAtomsList2, 'A')
    ##cm = chainA.Get_CAContactMap()
    ##nv = Get_NeighborAminoNo(cm, 4.0, 12.0)
    neiVtr1 = chain1.Get_SurroundVectorSet(4, 12, 1000)
    neiVtr2 = chain2.Get_SurroundVectorSet(4, 12, 1000)
    #cm1 = chain1.Get_CAContactMap()
    #nv1 = Get_NeighborAminoNo(cm1, 4.0, 12.0)
    #cm2 = chain2.Get_CAContactMap()
    #nv2 = Get_NeighborAminoNo(cm2, 4.0, 12.0)

    simValArray = []
    for i in range(0, len(chain2)):
        print(str(i) + '\t\t' + str(Calc_SimVectorSet(neiVtr1[67], neiVtr2[i])))
        simValArray.append(Calc_SimVectorSet(neiVtr1[67], neiVtr2[i]))
    x = np.arange(0, len(chain2))
    plt.scatter(x, simValArray)
    plt.show()
    plt.figure()
    plt.hist(simValArray, bins=20)
    plt.show()

    #neiVtr1 = chain1.Get_SurroundVectorSet(12, 20, 1000)
    #neiVtr2 = chain2.Get_SurroundVectorSet(12, 20, 1000)
    #simValArray = []
    #for i in range(0, len(chain2)):
    #    #print(str(i) + '\t\t' + str(Calc_SimVectorSet(neiVtr1[100], neiVtr2[i])))
    #    simValArray.append(Calc_SimVectorSet(neiVtr1[80], neiVtr2[i]))
    #    print(i)
    #plt.scatter(x, simValArray)

    #neiVtr1 = chain1.Get_SurroundVectorSet(0, 100, 1000)
    #neiVtr2 = chain2.Get_SurroundVectorSet(0, 100, 1000)
    #simValArray = []
    #for i in range(0, len(chain2)):
    #    #print(str(i) + '\t\t' + str(Calc_SimVectorSet(neiVtr1[100], neiVtr2[i])))
    #    simValArray.append(Calc_SimVectorSet(neiVtr1[20], neiVtr2[i]))
    #plt.plot(x, simValArray)

    #SavePDBFile("pdbs/4xt3_new.pdb", [chainA])
    Calc_SimVectorSet(neiVtr1[27], neiVtr2[27], True)
    print()
