import argparse
import random
import sys
from dataParse import construct_dataReader

def read_command():
    """
    main_file: the main data file
    main_format: format of the main data file
    --inputs: additional input data files
    --formats: formats of additional input data files
    --proportions: number of instances that should be used from each additional input file
    --output: output file name
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('main_file',
                        help='The main input file name.')
    parser.add_argument('main_format',
                        help='Format of the main input file.')
    parser.add_argument('-i', '--inputs',
                        nargs='+',
                        default=None,
                        help='Additional input files. Number of instances taken from each ')
    parser.add_argument('-f', '--formats',
                        nargs='+',
                        default=None,
                        help='Format of each input file data. Must be in one to one correspondence with input files',
                        choices=['SemEval', 'Naacl', 'standard'])
    parser.add_argument('-p', '--proportions',
                        nargs='+',
                        default=None,
                        type=int,
                        help='Number of instances that should be used from each additional input file. If not'
                             'specified, all instances of all additional input files will be used.')
    parser.add_argument('-o', '--output',
                        default=None,
                        help='Final output data file name.')

    opt = parser.parse_args()

    if opt.inputs:
        if not opt.formats:
            parser.error('with --inputs specified, --formats is required')
        elif not len(opt.inputs) == len(opt.formats):
            parser.error('--inputs should have same number of elements with --formats')
        elif opt.proportions and not len(opt.inputs) == len(opt.proportions):
            parser.error('--inputs should have same number of elements with --proportions')
    elif opt.formats or opt.proportions:
        parser.error('without --inputs, --formats and --proportions are not allowed')

    return opt


def cooperate_data(main_file, main_format, files=None, formats=None, proportions=None):
    """
    Cooperate data from different files together.
    :return: list of data instances
    """
    main_parser = construct_dataReader(main_file, main_format)
    final_data = main_parser.read_data()[0]

    if not files:
        return final_data

    parsers = []
    for i, file in enumerate(files):
        format = formats[i]
        parsers.append(construct_dataReader(file, format))

    if proportions:
        for i, reader in enumerate(parsers):
            data = reader.read_data()[0]
            # in case user requires more instances than a file contains
            try:
                sample = random.sample(data, proportions[i])
                final_data.extend(sample)
            except ValueError:
                print "%s does not contain enough data instances" % files[i]
                print "Total number of data instances: %s" % len(data)
                while True:
                    action = raw_input("please reenter the number of instances taken from this file, "
                                       "or 's' for skip, 'q' for quit")
                    try:
                        num = int(action)
                        if num > len(data):
                            continue
                        sample = random.sample(data, num)
                        final_data.extend(sample)
                        break
                    except ValueError:
                        if action == 's':
                            break
                        elif action == 'q':
                            sys.exit()
                        else:
                            print "Unknown command"

    else:
        for reader in parsers:
            final_data.extend(reader.read_data()[0])

    return final_data


def write_final_data(final_data, output_file="final.txt"):
    """
    e1  start_index1    end_index1  e2  start_index2    end_index2  relation    sentence    source
    """
    with open(output_file, 'w') as file:
        for i, instance in enumerate(final_data):
            file.write('%s\t%s\t%s\t' % (instance['e1'][0], instance['e1'][1], instance['e1'][2]))
            file.write('%s\t%s\t%s\t' % (instance['e2'][0], instance['e2'][1], instance['e2'][2]))
            file.write('%s\t' % instance['relation'])
            file.write('%s\t' % instance['sentence'])
            file.write('%s\n' % instance['source'])


if __name__ == '__main__':
    opt = read_command()

    final_data = cooperate_data(opt.main_file, opt.main_format, opt.inputs, opt.formats, opt.proportions)
    if opt.output:
        write_final_data(final_data, opt.output)
    else:
        write_final_data(final_data)