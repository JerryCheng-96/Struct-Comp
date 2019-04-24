from Vector_Matrix import *

# TODO:Change the Function for new .PDB file data structures.

def TMscoreAligned(resis_mod_aligned, resis_nat_aligned, len_nat):
    # First, get the N-CA-C chain atoms vector list
    listCAVtr_mod = np.array([[r.atoms["N"].vector, r.atoms["CA"].vector, r.atoms["C"].vector]
                              for r in resis_mod_aligned])
    listCAVtr_nat = np.array([[r.atoms["N"].vector, r.atoms["CA"].vector, r.atoms["C"].vector]
                              for r in resis_nat_aligned])
    listCAVtr_mod = np.array(list(listCAVtr_mod[:,0]) + list(listCAVtr_mod[:,1]) + list(listCAVtr_mod[:,2]))
    listCAVtr_nat = np.array(list(listCAVtr_nat[:,0]) + list(listCAVtr_nat[:,1]) + list(listCAVtr_nat[:,2]))
#    listCAVtr_mod = np.array([list(r["CA"].get_vector())
#                              for r in resis_mod_aligned])
#    listCAVtr_nat = np.array([list(r["CA"].get_vector())
#                              for r in resis_nat_aligned])

    # Second, Kabsching and rotate the model
    rotMat, transVtr, points_mod, points_nat = Kabsch_CenterRot(listCAVtr_mod, listCAVtr_nat)

    # Calculating TMscore
    ## Calculating d0
    d0 = 1.24 * pow(len_nat - 15, 1/3) - 1.8

    ## Calculating di's
    di = np.sqrt(np.sum(pow(points_mod - points_nat, 2), axis=1))
    di = di[int(len(di)/3) - 1: 2 * int(len(di)/3) - 1]

    ## Calculating TMscore
    tmScore = np.sum(1 / (pow(di / d0, 2) + 1)) / len_nat
    print("TM-score = " + str(tmScore))

    len_mod = len(resis_mod_aligned)

    # Calculating TMscore
    ## Calculating d0
    d0 = 1.24 * pow(len_mod - 15, 1/3) - 1.8

    ## Calculating di's
    di = np.sqrt(np.sum(pow(points_mod - points_nat, 2), axis=1))
    di = di[int(len(di)/3) - 1: 2 * int(len(di)/3) - 1]

    ## Calculating TMscore
    tmScore = np.sum(1 / (pow(di / d0, 2) + 1)) / len_mod
    print("TM-score = " + str(tmScore))

    return rotMat, transVtr, points_mod, points_nat
