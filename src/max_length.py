import os

from src.common.json_utils import JsonUtils


def analyze_text_lengths(folder):
    lengths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                text = JsonUtils.flatten(json_path)
                lengths.append(len(text.split()))

    return lengths

hired_lengths = analyze_text_lengths('resumes/results/hired/')
rejected_lengths = analyze_text_lengths('resumes/results/rejected/')
all_lengths = hired_lengths + rejected_lengths

# Plot histogram
import matplotlib.pyplot as plt

plt.hist(all_lengths, bins=50, alpha=0.75)
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Distribution of Resume Text Lengths')
plt.show()
