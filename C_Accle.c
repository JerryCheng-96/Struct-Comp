#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include "Univ_NW.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#define MIN(A, B) (A) < (B) ? (A) : (B)
#define ZSCORE(VAL, AVGVAL, STDVAL) ((VAL) - (AVGVAL)) / (STDVAL)

// Definition: Accessing the data at a certain position...
#define PTR_DOUBLE_ELEM_3(PDATA, STRD, I, J, K)     (double*)((PDATA) + (I) * (STRD)[0] + (J) * (STRD)[1] + (K) * STRD[2])
#define PTR_DOUBLE_ELEM_2(PDATA, STRD, I, J)        (double*)((PDATA) + (I) * (STRD)[0] + (J) * (STRD)[1])
#define PTR_DOUBLE_VTR(PVTR, STRD, I)               (double*)((PVTR) + (I) * (STRD))


// Definition: Simple vector operations}
#define VTR_DOT_3D(PVTR1, STRD1, PVTR2, STRD2) (\
    (*(double*)((PVTR1) + (STRD1) * 0)) * (*(double*)((PVTR2) + (STRD2) * 0)) + \
    (*(double*)((PVTR1) + (STRD1) * 1)) * (*(double*)((PVTR2) + (STRD2) * 1)) + \
    (*(double*)((PVTR1) + (STRD1) * 2)) * (*(double*)((PVTR2) + (STRD2) * 2))   )
#define VTR_LEN_3D(PVTR, STRD) sqrt( \
    pow(*(double*)(PVTR), 2) + \
    pow(*(double*)((PVTR) + (STRD) * 1), 2) + \
    pow(*(double*)((PVTR) + (STRD) * 2), 2)   )
#define VTR_ANGLECOS(PVTR1, STRD1, PVTR2, STRD2) \
    VTR_DOT_3D((PVTR1), (STRD1), (PVTR2), (STRD2)) / ( \
        VTR_LEN_3D((PVTR1), (STRD1)) * \
        VTR_LEN_3D((PVTR2), (STRD2))   )

#define PRINT_VTR(PVTR, STRD) \
    printf("[%lf\t%lf\t%lf]\n", \
           *(double*)((PVTR) + 0 * (STRD)), \
           *(double*)((PVTR) + 1 * (STRD)), \
           *(double*)((PVTR) + 2 * (STRD))  )


PyObject* pOptimizeFunc = NULL;
typedef void (*pCLO)(double*, int, int, int*, int*);
pCLO pCLinearOptimize = NULL;


/*
    Vector operations that require returning new vector
*/
PyObject* VTR_CROSS_3D(char* pVtr1, int strd1, char* pVtr2, int strd2) {
    double vtr1[] = {*(double*)((pVtr1) + (strd1) * 0), *(double*)((pVtr1) + (strd1) * 1), *(double*)((pVtr1) + (strd1) * 2)};
    double vtr2[] = {*(double*)((pVtr2) + (strd2) * 0), *(double*)((pVtr2) + (strd2) * 1), *(double*)((pVtr2) + (strd2) * 2)};

    //
    //for (int i = 0; i < 3; i++) {
    //    printf("%lf\t", vtr1[i]);
    //}
    //printf("\n");
    //for (int i = 0; i < 3; i++) {
    //    printf("%lf\t", vtr2[i]);
    //}
    //printf("\n");
    //

    long int dims[] = {1, 3};
    PyObject* resVtr = PyArray_SimpleNew(2, (npy_intp*)dims, NPY_DOUBLE);
    npy_intp* np_strides = PyArray_STRIDES(resVtr);
    char* pResVtr = (char*)PyArray_DATA(resVtr);
    
    *(double*)(pResVtr + 0 * np_strides[1]) = vtr1[1] * vtr2[2] - vtr1[2] * vtr2[1];
    *(double*)(pResVtr + 1 * np_strides[1]) = vtr1[2] * vtr2[0] - vtr1[0] * vtr2[2];
    *(double*)(pResVtr + 2 * np_strides[1]) = vtr1[0] * vtr2[1] - vtr1[1] * vtr2[0];
    
    //for (int i = 0; i < 3; i++) {
    //    printf("%lf\t", *(double*)(pResVtr + i * np_strides[1]));
    //}
    //printf("\n");

    return resVtr;
}

PyObject* VTR_ADD_3D(char* pVtr1, int strd1, char* pVtr2, int strd2) {
    //double vtr1[] = {*(double*)((pVtr1) + (strd1) * 0), *(double*)((pVtr1) + (strd1) * 1), *(double*)((pVtr1) + (strd1) * 2)};
    //double vtr2[] = {*(double*)((pVtr2) + (strd2) * 0), *(double*)((pVtr2) + (strd2) * 1), *(double*)((pVtr2) + (strd2) * 2)};

    long int dims[] = {1, 3};
    PyObject* resVtr = PyArray_SimpleNew(2, (npy_intp*)dims, NPY_DOUBLE);
    npy_intp* np_strides = PyArray_STRIDES(resVtr);
    char* pResVtr = (char*)PyArray_DATA(resVtr);

    *(double*)(pResVtr + 0 * np_strides[1]) = *(double*)((pVtr1) + (strd1) * 0) + *(double*)((pVtr2) + (strd2) * 0);
    *(double*)(pResVtr + 1 * np_strides[1]) = *(double*)((pVtr1) + (strd1) * 1) + *(double*)((pVtr2) + (strd2) * 1);
    *(double*)(pResVtr + 2 * np_strides[1]) = *(double*)((pVtr1) + (strd1) * 2) + *(double*)((pVtr2) + (strd2) * 2);
    
    return resVtr;
}

PyObject* VTR_SUBTRACT_3D(char* pVtr1, int strd1, char* pVtr2, int strd2) {
    //double vtr1[] = {*(double*)((pVtr1) + (strd1) * 0), *(double*)((pVtr1) + (strd1) * 1), *(double*)((pVtr1) + (strd1) * 2)};
    //double vtr2[] = {*(double*)((pVtr2) + (strd2) * 0), *(double*)((pVtr2) + (strd2) * 1), *(double*)((pVtr2) + (strd2) * 2)};

    long int dims[] = {1, 3};
    PyObject* resVtr = PyArray_SimpleNew(2, (npy_intp*)dims, NPY_DOUBLE);
    npy_intp* np_strides = PyArray_STRIDES(resVtr);
    char* pResVtr = (char*)PyArray_DATA(resVtr);

    *(double*)(pResVtr + 0 * np_strides[1]) = *(double*)((pVtr1) + (strd1) * 0) - *(double*)((pVtr2) + (strd2) * 0);
    *(double*)(pResVtr + 1 * np_strides[1]) = *(double*)((pVtr1) + (strd1) * 1) - *(double*)((pVtr2) + (strd2) * 1);
    *(double*)(pResVtr + 2 * np_strides[1]) = *(double*)((pVtr1) + (strd1) * 2) - *(double*)((pVtr2) + (strd2) * 2);
    
    return resVtr;
}

double avg(double* theData, int len) {
    double sum = 0.0;
    for (int i = 0; i < len; i++) sum += theData[i];
    return sum / len;
}

double stddev(double* theData, double avgVal, int len) {
    double sum = 0.0;
    for (int i = 0; i < len; i++) sum += pow((theData[i] - avgVal), 2);
    return sqrt(sum / len);
}


/*
    Real accelerations here!
*/
static PyObject* C_SurroundVectorSet(PyObject* self, PyObject* args) {
    double low_cutoff, high_cutoff;
    PyObject* animoAtoms = NULL;
    PyObject* animoAtomsArray = NULL;

    if (!PyArg_ParseTuple(args, "Odd", &animoAtoms, &low_cutoff, &high_cutoff))
        return NULL;

    animoAtomsArray = PyArray_FROM_OTF(animoAtoms, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    npy_intp* dims = PyArray_DIMS(animoAtomsArray);
    npy_intp* strides = PyArray_STRIDES(animoAtomsArray);
    char* pAnimoAtomsArray = (char*)PyArray_DATA(animoAtomsArray);

    // DEBUGGING
    //printf("Cutoff = (%lf, %lf)\nDims:Strides = ", low_cutoff, high_cutoff); 
    //int i;
    //for (i = 0; i < PyArray_NDIM(animoAtomsArray); i++) {
    //    printf("%ld:%ld    ", dims[i], strides[i]);
    //} 
    //printf("\n");

    //i = 0;
    //int j, k;
    //for (j = 0; j < dims[1]; j++) {
    //    for (k = 0; k < dims[2]; k++) {
    //        //printf("%lf\t", *(double*)(pAnimoAtomsArray + i * strides[0] + j * strides[1] + k * strides[2]));
    //        printf("%lf\t", *PTR_DOUBLE_ELEM_3(pAnimoAtomsArray, strides, i, j, k));
    //    }
    //    printf("\n");
    //}

    //printf("%lf\n", VTR_DOT_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 0), strides[2], \
    //                           (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 1), strides[2]));

    //printf("%lf\n", VTR_LEN_3D((char*)pAnimoAtomsArray, strides[2]));
    //PyObject* crossRes = VTR_CROSS_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 0), strides[2], \
    //                                  (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 2), strides[2]);
    //for (i = 0; i < 3; i++) {
    //    printf("%lf\t", *PTR_DOUBLE_ELEM_2(PyArray_DATA(crossRes), PyArray_STRIDES(crossRes), 0, i));
    //}
    //printf("\n");

    //printf("Adding Vtrs: ");
    //PyObject* addRes = VTR_ADD_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 0), strides[2], \
    //                                  (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 2), strides[2]);
    //for (i = 0; i < 3; i++) {
    //    printf("%lf\t", *PTR_DOUBLE_ELEM_2(PyArray_DATA(addRes), PyArray_STRIDES(addRes), 0, i));
    //}
    //printf("\n");
    
    //printf("Subtracting Vtrs: ");
    //PyObject* subRes = VTR_SUBTRACT_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 0), strides[2], \
    //                                  (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, 0, 2), strides[2]);
    //for (i = 0; i < 3; i++) {
    //    printf("%lf\t", *PTR_DOUBLE_ELEM_2(PyArray_DATA(subRes), PyArray_STRIDES(subRes), 0, i));
    //}
    //printf("\n");

    // END OF DEBUGGING

    PyObject* aminosList = PyList_New(0);
    int aa1, aa2;
    for (aa1 = 0; aa1 < dims[0]; aa1++) {
        PyObject* theAminoContact = PyList_New(0);
        PyObject* vtr_CaN = VTR_SUBTRACT_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, aa1, 1), strides[2], \
                                            (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, aa1, 0), strides[2]);
        PyObject* vtr_CaC = VTR_SUBTRACT_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, aa1, 1), strides[2], \
                                            (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, aa1, 2), strides[2]);

        //if (aa1 == 0) {
        //    printf("CaN, CaC: \n");
        //    PRINT_VTR((char*)PyArray_DATA(vtr_CaN), PyArray_STRIDE(vtr_CaN, 1));
        //    PRINT_VTR((char*)PyArray_DATA(vtr_CaC), PyArray_STRIDE(vtr_CaC, 1));
        //    printf("\n");
        //}

        for (aa2 = 0; aa2 < dims[0]; aa2++) {
            PyObject* vtr_CaCa = VTR_SUBTRACT_3D((char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, aa2, 1), strides[2], \
                                                 (char*)PTR_DOUBLE_ELEM_2(pAnimoAtomsArray, strides, aa1, 1), strides[2]);
            double theVtrLen = VTR_LEN_3D((char*)PyArray_DATA(vtr_CaCa), PyArray_STRIDE(vtr_CaCa, 1));

            //if (aa1 == 0 && aa2 == 3) {
            //    printf("CaCa: \n");
            //    PRINT_VTR((char*)PyArray_DATA(vtr_CaCa), PyArray_STRIDE(vtr_CaCa, 1));
            //    printf("theVtrLen=%lf\n", theVtrLen);
            //}

            if (theVtrLen > low_cutoff && theVtrLen < high_cutoff) {
                double bondAngle = acos(VTR_ANGLECOS((char*)PyArray_DATA(vtr_CaN), PyArray_STRIDE(vtr_CaN, 1), \
                                                //(char*)PyArray_DATA(vtr_CaCa), PyArray_STRIDE(vtr_CaCa, 1), (aa1 == 0 && aa2 == 3));
                                                (char*)PyArray_DATA(vtr_CaCa), PyArray_STRIDE(vtr_CaCa, 1)));
                
                //if (aa1 == 0 && aa2 == 3) {
                //    printf("ANGLECOS = %lf\n", bondAngle);
                //    printf("STRIDE = %ld\n", PyArray_STRIDE(vtr_CaN, 1));
                //}

                PyObject* plnNorm1 = VTR_CROSS_3D( \
                    (char*)PyArray_DATA(vtr_CaN), PyArray_STRIDE(vtr_CaN, 1), \
                    (char*)PyArray_DATA(vtr_CaCa), PyArray_STRIDE(vtr_CaCa, 1)  \
                );
                PyObject* plnNorm2 = VTR_CROSS_3D( \
                    (char*)PyArray_DATA(vtr_CaN), PyArray_STRIDE(vtr_CaN, 1), \
                    (char*)PyArray_DATA(vtr_CaC), PyArray_STRIDE(vtr_CaC, 1)  \
                );
                double torsionAngle = acos(VTR_ANGLECOS((char*)PyArray_DATA(plnNorm1), PyArray_STRIDE(plnNorm1, 1), \
                                                   (char*)PyArray_DATA(plnNorm2), PyArray_STRIDE(plnNorm2, 1)));

                PyObject* contactInfo = PyList_New(3);
                PyList_SetItem(contactInfo, 0, PyFloat_FromDouble(theVtrLen));
                PyList_SetItem(contactInfo, 1, PyFloat_FromDouble(bondAngle));
                PyList_SetItem(contactInfo, 2, PyFloat_FromDouble(torsionAngle));
                PyList_Append(theAminoContact, PyArray_FROM_O(contactInfo));
            }
        }
        PyList_Append(aminosList, theAminoContact);
    }


    //Py_RETURN_NONE;
    return aminosList;
}

PyObject* CLinearOptimizeWrapper(PyObject* self, PyObject* args) {
    PyObject* pyMatrix = NULL;
    if (!PyArg_ParseTuple(args, "O", &pyMatrix)) printf("CLinearOptimizeWrapper args parse failed!!!\n");
    if (!PyArray_Check(pyMatrix)) printf("CLinearOptimizeWrapper arg not ndarray!!!\n");

    printf("PyArray ndims = %ld\n", PyArray_NDIM(pyMatrix));
    char* matData = PyArray_DATA(pyMatrix);
    npy_intp* matDims = PyArray_DIMS(pyMatrix);
    npy_intp* matStrides = PyArray_STRIDES(pyMatrix);

    double* cArrayMat = (double*)malloc(sizeof(double) * matDims[0] * matDims[1]);
    int i, j;
    for (i = 0; i < matDims[0]; i++) {
        for (j = 0; j < matDims[1]; j++)
            *(cArrayMat + i * matDims[1] + j) = *PTR_DOUBLE_ELEM_2(matData, matStrides, i, j);
    }
    
    int numPairs = MIN(matDims[0], matDims[1]);
    int* numRows = (int*)malloc(sizeof(int) * numPairs);
    int* numCols = (int*)malloc(sizeof(int) * numPairs);

    pCLinearOptimize(cArrayMat, matDims[0], matDims[1], numRows, numCols);
    for (i = 0; i < numPairs; i++) printf("(%d, %d)\t", numRows[i], numCols[i]);
    printf("\n");

    Py_RETURN_NONE;
}


double SimFunc_InternalCoord(PyObject* vtr1, PyObject* vtr2) {
    //printf("In SimFunc_Contact\n");
    if ((!PyArray_Check(vtr1)) || (!PyArray_Check(vtr2))) { printf("Not ndarray!\n"); return 0.0; }
    //printf("ndims = %ld, %ld\n", PyArray_NDIM(vtr1), PyArray_NDIM(vtr2));

    double vtrVals1[] = { *(double*)((char*)PyArray_DATA(vtr1) + 0 * PyArray_STRIDE(vtr1, 0)), 
                          *(double*)((char*)PyArray_DATA(vtr1) + 1 * PyArray_STRIDE(vtr1, 0)),
                          *(double*)((char*)PyArray_DATA(vtr1) + 2 * PyArray_STRIDE(vtr1, 0))};
    double vtrVals2[] = { *(double*)((char*)PyArray_DATA(vtr2) + 0 * PyArray_STRIDE(vtr2, 0)), 
                          *(double*)((char*)PyArray_DATA(vtr2) + 1 * PyArray_STRIDE(vtr2, 0)),
                          *(double*)((char*)PyArray_DATA(vtr2) + 2 * PyArray_STRIDE(vtr2, 0))};
    //if (showDbg) {
    //    printf("vtr1 = %lf, %lf, %lf\n", vtrVals1[0], vtrVals1[1], vtrVals1[2]);
    //    printf("vtr2 = %lf, %lf, %lf\n", vtrVals2[0], vtrVals2[1], vtrVals2[2]);
    //}

    double itemLength = vtrVals1[0] / vtrVals2[0];
    itemLength = itemLength < 1 ? itemLength : (1 / itemLength);

    double itemBond = cos(vtrVals1[1] - vtrVals2[1]);
    double itemTorsion = cos(vtrVals1[2] - vtrVals2[2]);

    return 3 - (itemLength + itemBond + itemTorsion) / 3;
}

PyObject* SimFunc_InternalCoord_Wrapper(PyObject* self, PyObject* args) {
    PyObject* vtr1 = NULL;
    PyObject* vtr2 = NULL;

    if (!PyArg_ParseTuple(args, "OO", &vtr1, &vtr2)) 
        { printf("SimFunc_InternalCoord args parse failed!!!\n"); Py_RETURN_NONE; }

    return PyFloat_FromDouble(SimFunc_InternalCoord(vtr1, vtr2));
}

double SimFunction(void* voidPtrAmino1, void* voidPtrAmino2) {
    PyObject* pAmino1 = (PyObject*)voidPtrAmino1; 
    PyObject* pAmino2 = (PyObject*)voidPtrAmino2; 

    if ((!PyTuple_Check(pAmino1)) || (!(PyTuple_Check(pAmino2)))) { printf("Not Tuple!!!\n"); return -1e9; }
    //printf("Tuple sizes = %ld, %ld\n", PyTuple_Size(pAmino1), PyTuple_Size(pAmino2));

    PyObject* amino1_vtr = PyTuple_GetItem(pAmino1, 0);
    PyObject* amino2_vtr = PyTuple_GetItem(pAmino2, 0);
    PyObject* amino1_ct = PyTuple_GetItem(pAmino1, 1);
    PyObject* amino2_ct = PyTuple_GetItem(pAmino2, 1);

    if ((!PyArray_Check(amino1_vtr)) || (!(PyArray_Check(amino2_vtr)))) { printf("Not ndarray!!!\n"); return -1e9; }
    if ((!PyList_Check(amino1_ct)) || (!(PyList_Check(amino2_ct)))) { printf("Not List!!!\n"); return -1e9; }
    
    Py_ssize_t len_amino1_ct = PyList_Size(amino1_ct);
    Py_ssize_t len_amino2_ct = PyList_Size(amino2_ct);

    //printf("%ld, %ld\n", PyArray_STRIDE(amino1_vtr, 0), PyArray_STRIDE(amino2_vtr, 0));

    double itemVtr = VTR_ANGLECOS((char*)PyArray_DATA(amino1_vtr), PyArray_STRIDE(amino1_vtr, 0), 
                                  (char*)PyArray_DATA(amino2_vtr), PyArray_STRIDE(amino2_vtr, 0));
    if (isnan(itemVtr)) itemVtr = 0;
    //printf("%lf\n", itemVtr);
    
    if (!pCLinearOptimize) { printf("pCLinearOptimize is NULL!!!\n"); return 0.0; }
    double* contactMatrix = (double*)malloc(sizeof(double) * len_amino1_ct * len_amino2_ct);

    int i, j;
    for (i = 0; i < len_amino1_ct; i++) {
        for (j = 0; j < len_amino2_ct; j++) 
            *(contactMatrix + i * len_amino2_ct + j) = \
                //SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j), (i == 1 && j == 1)); 
                //-SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j)); 
                SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j)); 
    }

    int numPairs = MIN(len_amino1_ct, len_amino2_ct);
    int* numRows = (int*)malloc(sizeof(int) * numPairs);
    int* numCols = (int*)malloc(sizeof(int) * numPairs);

    double sumValue = 0;
    pCLinearOptimize(contactMatrix, len_amino1_ct, len_amino2_ct, numRows, numCols);
    for (i = 0; i < numPairs; i++) sumValue += -(*(contactMatrix + numRows[i] * len_amino2_ct + numCols[i]) - 3);
    return sumValue / numPairs;
    //printf("\n");

    //if (!pOptimizeFunc) return 0.0;
    //long int contactMatNewDims[] = {len_amino1_ct, len_amino2_ct}; 
    //PyObject* contactMat = PyArray_SimpleNew(2, contactMatNewDims, NPY_DOUBLE);
    //npy_intp* contactMatDims = PyArray_DIMS(contactMat);
    //npy_intp* contactMatStrides = PyArray_STRIDES(contactMat);
    //char* pContactMatData = (char*)PyArray_DATA(contactMat);

    //int i, j;
    //for (i = 0; i < len_amino1_ct; i++) {
    //    for (j = 0; j < len_amino2_ct; j++) 
    //        *(double*)PTR_DOUBLE_ELEM_2(pContactMatData, contactMatStrides, i, j) = \
    //            //SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j), (i == 1 && j == 1)); 
    //            -SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j)); 
    //}

    //PyObject* argsTuple = PyTuple_New(1);
    //PyTuple_SetItem(argsTuple, 0, contactMat);
    //PyObject* optimizeRes = PyObject_CallObject(pOptimizeFunc, argsTuple);
    //if (!(PyTuple_Check(optimizeRes)) || (!PyTuple_Size(optimizeRes) == 2)) { printf("Optimize return value error.\n"); return 0.0; }

    //return 0.0;
}


PyObject* SimFunction_Dbg(void* voidPtrAmino1, void* voidPtrAmino2) {
    PyObject* pAmino1 = (PyObject*)voidPtrAmino1; 
    PyObject* pAmino2 = (PyObject*)voidPtrAmino2; 

    if ((!PyTuple_Check(pAmino1)) || (!(PyTuple_Check(pAmino2)))) { printf("Not Tuple!!!\n"); Py_RETURN_NONE; }
    //printf("Tuple sizes = %ld, %ld\n", PyTuple_Size(pAmino1), PyTuple_Size(pAmino2));

    PyObject* amino1_vtr = PyTuple_GetItem(pAmino1, 0);
    PyObject* amino2_vtr = PyTuple_GetItem(pAmino2, 0);
    PyObject* amino1_ct = PyTuple_GetItem(pAmino1, 1);
    PyObject* amino2_ct = PyTuple_GetItem(pAmino2, 1);

    if ((!PyArray_Check(amino1_vtr)) || (!(PyArray_Check(amino2_vtr)))) { printf("Not ndarray!!!\n"); Py_RETURN_NONE; }
    if ((!PyList_Check(amino1_ct)) || (!(PyList_Check(amino2_ct)))) { printf("Not List!!!\n"); Py_RETURN_NONE; }
    
    Py_ssize_t len_amino1_ct = PyList_Size(amino1_ct);
    Py_ssize_t len_amino2_ct = PyList_Size(amino2_ct);

    //printf("%ld, %ld\n", PyArray_STRIDE(amino1_vtr, 0), PyArray_STRIDE(amino2_vtr, 0));

    double itemVtr = VTR_ANGLECOS((char*)PyArray_DATA(amino1_vtr), PyArray_STRIDE(amino1_vtr, 0), 
                                  (char*)PyArray_DATA(amino2_vtr), PyArray_STRIDE(amino2_vtr, 0));
    if (isnan(itemVtr)) itemVtr = 0;
    //printf("%lf\n", itemVtr);
    
    //if (!pOptimizeFunc) return 0.0;
    if (!pOptimizeFunc) Py_RETURN_NONE;
    long int contactMatNewDims[] = {len_amino1_ct, len_amino2_ct}; 
    PyObject* contactMat = PyArray_SimpleNew(2, contactMatNewDims, NPY_DOUBLE);
    npy_intp* contactMatDims = PyArray_DIMS(contactMat);
    npy_intp* contactMatStrides = PyArray_STRIDES(contactMat);
    char* pContactMatData = (char*)PyArray_DATA(contactMat);

    int i, j;
    for (i = 0; i < len_amino1_ct; i++) {
        for (j = 0; j < len_amino2_ct; j++) 
            *(double*)PTR_DOUBLE_ELEM_2(pContactMatData, contactMatStrides, i, j) = \
                //SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j), (i == 1 && j == 1)); 
                -SimFunc_InternalCoord(PyList_GetItem(amino1_ct, i), PyList_GetItem(amino2_ct, j)); 
    }

    PyObject* argsTuple = PyTuple_New(1);
    PyTuple_SetItem(argsTuple, 0, contactMat);
    return PyObject_CallObject(pOptimizeFunc, argsTuple);

    //return 0.0;
    Py_RETURN_NONE;
}


PyObject* SimFunction_Wrapper(PyObject* self, PyObject* args) {
    PyObject* amino1 = NULL;
    PyObject* amino2 = NULL;

    if (!PyArg_ParseTuple(args, "OO", &amino1, &amino2)) 
        { printf("args parse failed in SimFunction_Wrapper!!!\n"); Py_RETURN_NONE; }

    return SimFunction_Dbg((void*)amino1, (void*)amino2);
}

static PyObject* Aminos_NWAlign(PyObject* self, PyObject* args) {
    printf("Now in Aminos_NWAlign\n");
    PyObject* aminosList1 = NULL;
    PyObject* aminosList2 = NULL;
    int gap_start = 0;
    int gap_ext = 0;

    if (!PyArg_ParseTuple(args, "OOii", &aminosList1, &aminosList2, &gap_start, &gap_ext)) Py_RETURN_NONE;
    //if ((!PyList_Check(aminosList1)) || (!PyList_Check(aminosList2))) Py_RETURN_NONE; 
    if ((!PyList_Check(aminosList1)) || (!PyList_Check(aminosList2))) printf("Not Receiving list.\n"); 

    Py_ssize_t lenAminos1 = PyList_Size(aminosList1);
    Py_ssize_t lenAminos2 = PyList_Size(aminosList2);
    PyObject** pAminos1 = (PyObject**)malloc(sizeof(PyObject*) * lenAminos1);
    PyObject** pAminos2 = (PyObject**)malloc(sizeof(PyObject*) * lenAminos2);

    printf("Len = %ld, %ld\n", lenAminos1, lenAminos2);

    int i;
    for (i = 0; i < lenAminos1; i++) pAminos1[i] = PyList_GetItem(aminosList1, i);
    for (i = 0; i < lenAminos2; i++) pAminos2[i] = PyList_GetItem(aminosList2, i);

    char* tracePath = NW_Align((void*)pAminos1, lenAminos1, 
                               (void*)pAminos2, lenAminos2,
                               SimFunction, 
                               gap_start, gap_ext);

    return PyUnicode_FromString(tracePath); 
    Py_RETURN_NONE;
    
}
    
static PyMethodDef CAccelMethods[] = {
    {"C_SurroundVectorSet", C_SurroundVectorSet, METH_VARARGS, ""},
    {"Aminos_NWAlign", Aminos_NWAlign, METH_VARARGS, ""},
    {"C_SimFunc_InternalCoord", SimFunc_InternalCoord_Wrapper, METH_VARARGS, ""},
    {"C_SimFunc_Amino", SimFunction_Wrapper, METH_VARARGS, ""},
    {"C_LinearOptimize", CLinearOptimizeWrapper, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cAccel_module = {
    PyModuleDef_HEAD_INIT,
    "CAccel", 
    NULL, 
    -1, 
    CAccelMethods
};

int init_numpy() {
    import_array();
    return 0;
}

PyMODINIT_FUNC
PyInit_CAccel(void) {
    Py_Initialize();
    init_numpy();

    //PyObject* pSciPyModule = PyImport_ImportModule("scipy.optimize._hungarian");
    //if (!pSciPyModule) printf("scipy.optimize import failed.\n");
    //pOptimizeFunc = PyObject_GetAttrString(pSciPyModule, "linear_sum_assignment");
    //if ((!pOptimizeFunc)) printf("lsa is null.\n");
    //if ((pOptimizeFunc) && (PyFunction_Check(pOptimizeFunc))) printf("lsa import succeeded.\n");

    void* handle = dlopen("./libhungarian.so", RTLD_NOW);
    char* err = dlerror();
    if (err || !handle) printf("libhungarian load failed!!!\n%s\n", err);
    //printf("libhungarian loaded at %lx\n", handle);
    pCLinearOptimize = (pCLO)dlsym(handle, "Linear_OptAssign");
    err = dlerror();
    if (err) printf("Load Linear_OptAssign function failed!!!\n%s\n", err);
    //printf("lib function loaded at %lx\n", pCLinearOptimize);

    //double testArray[] = { 10, 19, 8, 15, 0, 10, 18, 7, 17, 0, 13, 16, 9, 14, 0, 12, 19, 8, 18, 0 };
    //int numRows[4] = {0};
    //int numCols[4] = {0};
    //pCLinearOptimize(testArray, 4, 5, numRows, numCols);
    //for (int i = 0; i < 4; i++) printf("(%d, %d)\t", numRows[i], numCols[i]);
    //printf("\n");

    return PyModule_Create(&cAccel_module);
}
