import os

real_count = len(
    os.listdir("dataset_features/real")
)

fake_count = len(
    os.listdir("dataset_features/fake")
)

print("\nDATASET BALANCE")
print("===================")

print("REAL FEATURES:", real_count)
print("FAKE FEATURES:", fake_count)

if real_count > 0:

    ratio = fake_count / real_count

    print(f"\nFake/Real Ratio: {ratio:.2f}")

    if ratio > 1.5:

        print("\nDataset is biased toward FAKE")

    else:

        print("\nDataset is reasonably balanced")