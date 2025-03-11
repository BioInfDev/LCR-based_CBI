import os, json, requests

class DATA_PARSER():
    @staticmethod
    def get_uniprot_accession(hpa_json: {}):
        formater = lambda line: str(line[0]).rstrip(']').lstrip('[').rstrip('"').strip("'")
        return list(set([formater(line['Uniprot']) for line in hpa_json if line['Uniprot'] != [] and line['UniProt evidence'] == "Evidence at protein level"]))
    @staticmethod
    def get_empty_proteins(hpa_json: {}):
        empty_search = lambda json: [line['Gene'] for line in json if line['Uniprot'] == []]
        return empty_search(hpa_json)
    @staticmethod
    def get_proteins_bad_evidence(hpa_json: {}):
        bad_evidence = lambda json: [line['Gene'] for line in json if line['UniProt evidence'] != "Evidence at protein level"]
        return bad_evidence(hpa_json)
    @staticmethod
    def get_urls_for_mapping_service(uniprot_accessions: [], package_size):
        split = lambda list: [list[i:i + package_size] for i in range(0, len(list), package_size)]
        packages = [','.join(package) for package in split(uniprot_accessions)]
        urls = ['https://rest.uniprot.org/uniprotkb/accessions?accessions={0}&format=fasta&proteinExistence="1: Evidence at protein level"'.format(accessions) for accessions in packages]
        return urls
    @staticmethod
    def get_exclusively_nucleoli_loc(all_uniprot_accession: [], uniprot_accession_multiloc: []):
        return [acc for acc in all_uniprot_accession if acc not in uniprot_accession_multiloc]
    @staticmethod
    def download_fasta_uniprotkb(dir, file_name, uniprot_accession: []):
        urls = DATA_PARSER.get_urls_for_mapping_service(uniprot_accession, 500)
        with open(f'{dir}/{file_name}.fasta', 'wt') as F:
            for url in urls:
                r = requests.get(url)
                F.writelines(r.text)

if __name__ == '__main__':
    # Setting up the output directory
    project_dir = os.getcwd()
    data_dir = f'{project_dir}/data/HPA/' # Create your own directory to store data files in project dir
    json_data = f'{data_dir}/nucleolus_proteins.json' # Place in the folder with data files json file from HPA

    # Downloading FASTA via Uniprot Accession
    log = {}

    F = open(json_data)
    J = json.load(F)
    A = DATA_PARSER.get_uniprot_accession(J)
    E = DATA_PARSER.get_empty_proteins(J)
    BE = DATA_PARSER.get_proteins_bad_evidence(J)
    print(f'Proteins with empty uniprot accession in data set:', E)
    print(f'Proteins with bad uniprot evidence in data set:', BE, '\n')
    DATA_PARSER.download_fasta_uniprotkb(f'{project_dir}/data/HPA/', 'nucleolus_proteins', A)
    log['Nucleolus proteins'] = {"HPA proteins" : len(J), "'Good' proteins" : len(A), "Percent" : [len(A)/len(J), f'{len(A)}/{len(J)}'], 'Uniprot accessions': A}
    print(log)