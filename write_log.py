import csv

def write_log(file_name, list_of_dict):
    key = list_of_dict[0].keys()
    with open(file_name,'w') as output_file:
        dict_writer = csv.DictWriter(output_file,key)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dict)