import torch
from torch import nn
import torch.optim as optim
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

smiles_dataset = [
    "CCO",
    "CCN",
    "CCC",
    "CO",
    "CCOC",
]

class SMILESGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMILESGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.out(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))

input_size = len(Chem.MolToSmiles(Chem.MolFromSmiles("CCO")))
output_size = input_size
hidden_size = 128

generator = SMILESGenerator(input_size, hidden_size, output_size)

def generate_compound(generator, max_length, temperature=0.7):
    with torch.no_grad():
        hidden = generator.init_hidden(1)
        input = torch.zeros(1, 1, input_size)
        input[0, 0, input_size - 1] = 1  # Start token
        generated_compound = "C"  # Start with a carbon atom

        for _ in range(max_length):
            output, hidden = generator(input, hidden)
            output = output.view(-1)
            output = output / temperature
            output = torch.exp(output) / torch.sum(torch.exp(output))
            sampled_index = torch.multinomial(output, 1)[0]
            input = torch.zeros(1, 1, input_size)
            input[0, 0, sampled_index] = 1
            generated_compound += Chem.MolToSmiles(Chem.MolFromSmiles(smiles_dataset[sampled_index]))

            if sampled_index == input_size - 1:  # End token
                break

    return generated_compound

# Generate 5 compounds
generated_compounds = []
for _ in range(5):
    new_compound = generate_compound(generator, max_length=50, temperature=0.7)
    generated_compounds.append(new_compound)

print("Generated 5 compounds are:")
for compound in generated_compounds:
    print(compound)
