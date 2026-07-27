import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================================================
#  DNA PROMOTER CLASSIFIER
#  Predict whether a DNA sequence contains a promoter motif,
#  from the raw sequence alone. A full ML pipeline:
#  data -> features -> model -> honest evaluation.
# =============================================================

np.random.seed(42)
BASES = ['A', 'C', 'G', 'T']
SEQ_LEN = 50
MOTIF = "TATAAT"   # the Pribnow box: a real bacterial promoter signal


def random_sequence(length=SEQ_LEN):
    """A random stretch of DNA."""
    return ''.join(np.random.choice(BASES, size=length))


def make_dataset(n=1000):
    """Half the sequences hide a promoter motif (label 1), half don't (label 0)."""
    sequences, labels = [], []
    for _ in range(n):
        seq = random_sequence()
        if np.random.rand() < 0.5:
            pos = np.random.randint(0, SEQ_LEN - len(MOTIF))     # drop the motif in
            seq = seq[:pos] + MOTIF + seq[pos + len(MOTIF):]
            labels.append(1)
        else:
            labels.append(0)
        sequences.append(seq)
    return sequences, np.array(labels)


def kmer_features(sequences, k=3):
    """Turn each sequence into counts of every length-k subsequence (k-mer).
    This is how you feed text-like biological data to a numeric model."""
    kmers = [''.join(p) for p in product(BASES, repeat=k)]
    idx = {km: i for i, km in enumerate(kmers)}
    X = np.zeros((len(sequences), len(kmers)))
    for row, seq in enumerate(sequences):
        for i in range(len(seq) - k + 1):
            X[row, idx[seq[i:i + k]]] += 1
    return X


if __name__ == "__main__":
    # 1. Build the labeled dataset
    seqs, y = make_dataset(1000)

    # 2. Turn DNA strings into numeric features (3-mer counts)
    X = kmer_features(seqs, k=3)

    # 3. Split into train and test -- the model NEVER sees the test set while learning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 4. Train a logistic regression classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Evaluate HONESTLY on held-out data
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    baseline = max(y_test.mean(), 1 - y_test.mean())   # accuracy of dumb majority-guessing

    print("===== DNA PROMOTER CLASSIFIER =====")
    print(f"Test-set accuracy:        {acc:.1%}")
    print(f"Baseline (majority guess): {baseline:.1%}")
    print(f"Lift over baseline:        {acc - baseline:+.1%}")
    print("\nConfusion matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(y_test, pred))
    print("\nInterpretation: lift well above baseline means the model genuinely")
    print("learned the promoter signal from raw sequence -- not just guessing.")