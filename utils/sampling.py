import csv

def naiveSampler(file_path, header, nb_rows):
    file_to_sample = open(file_path, 'r')
    reader = csv.reader(file_to_sample)
    next(reader)

    output_file = open("./outputs/naive_sampled_"+str(nb_rows)+".csv", 'w+')
    output_file.write(header)

    for row in reader:
        output_file.write(','.join(row))
        output_file.write('\n')
        if reader.line_num >= nb_rows:
            break

    file_to_sample.close()
    output_file.close()