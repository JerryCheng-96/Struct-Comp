
def TracePath2AlignedList(revTracePath, seq_1, seq_2):
    tracePath = revTracePath[::-1]
    res_1 = []
    res_2 = []
    idx_1 = 0
    idx_2 = 0

    for i in tracePath:
        if i == '.':
            res_1.append(seq_1[idx_1])
            res_2.append(seq_2[idx_2])
            idx_1 += 1
            idx_2 += 1
        elif i == '-':
            res_1.append(seq_1[idx_1])
            res_2.append(None)
            idx_1 += 1
        elif i == '+':
            res_2.append(seq_2[idx_2])
            res_1.append(None)
            idx_2 += 1

    return res_1, res_2

def TracePath2OnlyAlignedList(revTracePath, seq_1, seq_2):
    tracePath = revTracePath[::-1]
    res_1 = []
    res_2 = []
    idx_1 = 0
    idx_2 = 0

    for i in tracePath:
        if i == '.':
            res_1.append(seq_1[idx_1])
            res_2.append(seq_2[idx_2])
            idx_1 += 1
            idx_2 += 1
        elif i == '-':
            idx_1 += 1
        elif i == '+':
            idx_2 += 1

    return res_1, res_2

def TracePath2AlignedStr(revTracePath, seqStr_1, seqStr_2):
    tracePath = revTracePath[::-1]
    resStr_1 = ""
    resStr_2 = ""
    idx_1 = 0
    idx_2 = 0

    for i in tracePath:
        if i == '.':
            resStr_1 += seqStr_1[idx_1]
            resStr_2 += seqStr_2[idx_2]
            idx_1 += 1
            idx_2 += 1
        elif i == '-':
            resStr_1 += seqStr_1[idx_1]
            resStr_2 += '-'
            idx_1 += 1
        elif i == '+':
            resStr_2 += seqStr_2[idx_2]
            resStr_1 += '-'
            idx_2 += 1

    return resStr_1, resStr_2

