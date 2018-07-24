from collections import defaultdict

def get_counts(sequence):
    # 所有值被初始化为0
    counts = defaultdict(int)
    for x in sequence:
        counts[x] += 1
    return counts
