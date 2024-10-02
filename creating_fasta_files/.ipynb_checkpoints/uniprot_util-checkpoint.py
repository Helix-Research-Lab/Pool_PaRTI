import requests
import pandas as pd
from urllib.parse import quote_plus



def get_uniprot_data(accession, verbose=False):
    """
    Fetches UniProt data for a given accession number.

    Args:
    accession (str): The UniProt accession number.

    Returns:
    dict: A dictionary containing UniProt data.
    """
    url = f"https://www.uniprot.org/uniprot/{accession}.txt"
    response = requests.get(url)

    if response.status_code == 200:
        if verbose:
            print(response.text)
        data = parse_uniprot_data(response.text)
        return data
    else:
        return {"error": "Data not found or error in fetching data"}




def get_uniprot_accession_and_seq_from_gene_name(protein_name):
    """
    Retrieves the UniProt accession code, organism, and protein sequence for a given protein name in Homo sapiens.

    Args:
    protein_name (str): The name of the protein (e.g., "OPRM1").

    Returns:
    tuple: A tuple containing the UniProt accession code, organism, and protein sequence for the protein, 
           or None for each if not found.
    """
    # Construct the query URL for the UniProt search API
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f'gene_exact:"{protein_name}" AND reviewed:true',
        "fields": "accession,organism_name,sequence",
        "format": "json"
    }

    # Send the GET request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['results']:
            # Assuming the first result is the most relevant, extract the needed information
            result = data['results'][0]
            accession_code = result['primaryAccession']
            organism = result['organism']['scientificName']
            if organism == 'Homo sapiens':
                sequence = result.get('sequence', {}).get('value', None)  # Sequence might not always be available
                return accession_code, sequence
            else:
                print(f"Organism is not homo sapiens")
                return None, None
        else:
            print(f"No results found for protein name '{protein_name}' in Homo sapiens.")
            return None, None
    else:
        print(f"Error fetching data for protein name '{protein_name}' in Homo sapiens. Status code: {response.status_code}")
        return None, None



def parse_uniprot_data(data, verbose=False):
    """
    Parses the UniProt data from the raw text format into a structured dictionary.

    Args:
    data (str): Raw text data from UniProt.

    Returns:
    dict: Parsed UniProt data.
    """
    parsed_data = {}
    parsed_data["Domains"] = ""
    sequence_started = False
    sequence = []
    lines = data.split('\n')
    if verbose:
        print(lines)
    for line in lines:
        if line.startswith('ID'):
            parsed_data['Entry Name'] = line.split()[1]
        elif line.startswith('AC'):
            parsed_data['Accession'] = line.split()[1].rstrip(';')
        elif line.startswith('OS'):
            parsed_data['Organism'] = line[5:]
        elif line.startswith('OC'):
            parsed_data['Domains'] += line[5:]
        elif line.startswith('//'):
            break
        elif sequence_started:
            sequence.append(line.strip())
        elif line.startswith('SQ'):
            sequence_started = True

    parsed_data['Sequence'] = ''.join(sequence).replace(" ", "")
    return parsed_data

def dataframe_to_fasta(column_accession, column_sequence, output_file, notes=""):
    """
    Converts a DataFrame with UniProt accession numbers and sequences to a FASTA file.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    column_accession (str): Name of the column containing accession numbers.
    column_sequence (str): Name of the column containing protein sequences.
    output_file (str): Path to the output FASTA file.
    """
    with open(output_file, 'w') as fasta_file:
        fasta_file.write(f'>{column_accession}{notes}\n{column_sequence}\n')











