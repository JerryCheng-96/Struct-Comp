
int arg_max(double* theArray, double* pMaxValue, int len) {
    double maxValue = theArray[0];
    int maxIndex = 0;

    for(int i = 0; i < len; i++) {
        if (theArray[i] > maxValue) {
            maxValue = theArray[i];
            maxIndex = i;
        }
    }

    *pMaxValue = maxValue;
    return maxIndex;
}


void ReverseString(char * theString) {
	char temp = '\0';
	int length = strlen(theString);

	for (int i = 0; i < length / 2 - 1; i++)
	{
		temp = theString[i];
		theString[i] = theString[length - 1 - i];
		theString[length - 1 - i] = temp;
	}
}


char* NW_Align(void** seq_1, int len_seq_1,
               void** seq_2, int len_seq_2,
               double (*sim_func)(void*, void*),
               int gap_start, int gap_ext) {

    // Initializing the matrices
    // The rows / first layer pointers
    double** scoresMat = (double**)malloc((len_seq_2 + 2) * sizeof(double*));
    int** dirMat = (int**)malloc((len_seq_2 + 2) * sizeof(int*));

    // The cols
    scoresMat[0] = (double*)malloc((len_seq_1 + 2) * (len_seq_2 + 2) * sizeof(double));
    dirMat[0] = (int*)malloc((len_seq_1 + 2) * (len_seq_2 + 2) * sizeof(int));
    for (int row = 1; row < len_seq_2 + 2; row++) {
        scoresMat[row] = scoresMat[0] + row * (len_seq_1 + 2);
        dirMat[row] = dirMat[0] + row * (len_seq_1 + 2);
    }

    /*
    *   The Rules for dirMat: (UPDATED to be the same with Dr. Cao's implementation.)
    *   Coming from the **LEFT**, dirMat[i][j] < 0;
    *   Coming from the **DIAG**, dirMat[i][j] = 0;
    *   Coming from the **TOP**, dirMat[i][j] > 0.
    */

    // Setting init values
    scoresMat[0][0] = 0;
    scoresMat[0][1] = -gap_start;
    scoresMat[1][0] = -gap_start;
    dirMat[0][0] = 0;
    dirMat[0][1] = -1;
    dirMat[1][0] = 1;

    for(int j = 2; j <= len_seq_1; j++) {
        scoresMat[0][j] = -(gap_ext * (j - 1)) - gap_start;
        dirMat[0][j] = -1;
    }
    
    for(int i = 2; i <= len_seq_2; i++) {
        scoresMat[i][0] = -(gap_ext * (i - 1)) - gap_start;
        dirMat[i][0] = 1;
    }


    // Calculating scoresMat and dirMat

    double scoreValues[] = {0, 0, 0};
    int dirValues[] = {1, -1, 0};

    // DEBUGGING
    //printf("Length of sequences: %d, %d\n", len_seq_1, len_seq_2);
    // END OF DEBUGGING

    for(int i = 1; i <= len_seq_2; i++) {
        printf("Now at (%d, %d)\n", i - 1, 0);
        for(int j = 1; j <= len_seq_1; j++) {

            //DEBUGGING
            //printf("SeqLen (%d, %d)\n", len_seq_1, len_seq_2)
            //printf("Now at (%d, %d)\n", i - 1, j - 1);

            // Considering "local" (2x2) elements
            scoreValues[1] = scoresMat[i][j - 1] - gap_start;
            dirValues[1] = -1;
            scoreValues[2] = scoresMat[i - 1][j - 1] + sim_func(seq_2[i - 1], seq_1[j - 1]);
            scoreValues[0] = scoresMat[i - 1][j] - gap_start;
            dirValues[0] = 1;

            // Considering "distance" (row/col) elements
            // A distant, previous **ROW** elem "travels" here?
            for (int k = 0; k < j; k++) {
                double theScore = scoresMat[i][k] - (j-k-1) * gap_ext - gap_start;
                if (theScore > scoreValues[1]) {
                    dirValues[1] = k - j;
                    scoreValues[1] = theScore;
                }
            }

            // A distant, previous **COL** elem "travels" here?
            for (int k = 0; k < i; k++) {
                double theScore = scoresMat[k][j] - (i-k-1) * gap_ext - gap_start;
                if (theScore > scoreValues[0]) {
                    dirValues[0] = i - k;
                    scoreValues[0] = theScore;
                }
            }

            // Finding the max value
            dirMat[i][j] = dirValues[arg_max(scoreValues, &scoresMat[i][j], 3)];
        }
    }


    // Finding start point

    int max_row = 0;
    int max_col = len_seq_1;
    int max_val = scoresMat[0][len_seq_1];
    
    for(int i = 1; i <= len_seq_2; i++) {
        if (scoresMat[i][len_seq_1] > max_val) {
            max_row = i;
            max_col = len_seq_1;
            max_val = scoresMat[i][len_seq_1];
        }
    }

    for(int j = 1; j <= len_seq_1; j++) {
        if (scoresMat[len_seq_2][j] > max_val) {
            max_row = len_seq_2;
            max_col = j;
            max_val = scoresMat[len_seq_2][j];
        }
    }


    // Finding the path
    int idx_row = len_seq_2;
    int idx_col = len_seq_1;

    char* tracePath = (char*)memset(malloc((len_seq_1 + len_seq_2 + 1) * sizeof(char)), 0, (len_seq_1 + len_seq_2 + 1) * sizeof(char));
    int tracePathIdx = 0;
    while (idx_row != 0 || idx_col != 0) {
        int dirMatVal = dirMat[idx_row][idx_col];

        if (dirMatVal > 0) {        // Greater than 0 means coming from the **TOP**
            idx_row -= dirMatVal;
            //tracePath[tracePathIdx++] = '0' + dirMatVal;
            for (int i = 0; i < dirMatVal; i++) tracePath[tracePathIdx++] = '+';
        }

        if (dirMatVal == 0) {       // Equals 0 means coming from the **DIAG**
            idx_row--;
            idx_col--;
            tracePath[tracePathIdx++] = '.';
        }

        if (dirMatVal < 0) {        // Less than 0 means coming from the **LEFT**
            idx_col += dirMatVal;
            //tracePath[tracePathIdx++] = 'A' - dirMatVal;
            for (int i = 0; i < -dirMatVal; i++) tracePath[tracePathIdx++] = '-';
        }
    }
    return tracePath;
}