import pandas as pd

def calculate_frequency(input_string, char):
    count = 0
    for i in range(len(input_string) - len(char) + 1):
        if input_string[i:i + len(char)] == char:
            count += 1
    total_characters = len(input_string)
    frequency_char = count / total_characters
    return frequency_char

def calculate_positions(sequence):
    positions = {'A': [], 'C': [], 'G': [], 'T': []}
    for i, letter in enumerate(sequence):
        if letter in positions:
            positions[letter].append(i + 1)
    return positions

def calculate_mean_variance(positions):
    mean_variance = {}
    for letter, pos_list in positions.items():
        mean = sum(pos_list) / len(pos_list)
        variance = sum((x - mean) ** 2 for x in pos_list) / len(pos_list)
        mean_variance[f'Mean_{letter}'] = mean
        mean_variance[f'Variance_{letter}'] = variance
    return mean_variance

file_path = r"C:\Users\25832\Desktop"
data = pd.read_excel(file_path, header=None)

chars_to_count = ['A', 'C', 'G', 'T',
                  'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT',
                  'AAA', 'AAC', 'AAG', 'AAT', 'ACA', 'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG', 'ATT',
                  'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA', 'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT',
                  'GAA', 'GAC', 'GAG', 'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA', 'GTC', 'GTG', 'GTT',
                  'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG', 'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT']

frequency_df = pd.DataFrame()

for input_string in data.iloc[:, 0]:
    frequency_list = []
    for char in chars_to_count:
        frequency = calculate_frequency(input_string, char)
        frequency_list.append(frequency)
    positions = calculate_positions(input_string)
    mean_variance = calculate_mean_variance(positions)
    for letter, value in mean_variance.items():
        frequency_list.append(value)
    frequency_df = pd.concat([frequency_df, pd.DataFrame([frequency_list], columns=[f'Character_{c}' for c in chars_to_count] + list(mean_variance.keys()))], ignore_index=True)

frequency_df.to_excel(r"C:\Users\Desktop", index=False)
