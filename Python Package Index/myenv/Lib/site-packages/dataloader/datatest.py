#!/usr/bin/env python
import dataloader
import argparse

# Handle arguments
def return_parser():
    parser = argparse.ArgumentParser(description='Tool for displaying data using :any:`loader.read_data_sets`.')
    parser.add_argument('datadirectory', type=str,
                        help='Path to folder where data to be loaded and displayed is stored.')

    parser.add_argument('-hashlist', nargs='+',
                        help='List of hashes to read. Files will be read of the form "features_<hash>.ext"  or'
                             '"labels_<hash>.ext" where <hash> is a string in hashlist. If a hashlist is not '
                             'specified all files of the form "features_<hash>.ext"  or "labels_<hash>.ext" regardless '
                             'what string <hash> is will be loaded.')
    subfolder_group = parser.add_mutually_exclusive_group()
    subfolder_group.add_argument('-cold', action="store_true",
                                 help="Extra loading and testing for cold datasets")
    subfolder_group.add_argument('-subfolders', default=('test', 'dev', 'train'), nargs='+',
                        help='List of subfolders to load and display.')
    return parser


if __name__ == '__main__':
    args = return_parser().parse_args()
    if args.cold:
        args.subfolders = ['train', 'dev', 'test', 'dev_cold_item', 'dev_cold_user',
                           'test_cold_item', 'test_cold_user', 'both_cold']
    dataloader.read_data_sets(args.datadirectory, args.subfolders, args.hashlist).show()