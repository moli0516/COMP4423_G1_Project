import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from tests.support.split_loader import SCRIPT, load_split_module

IMAGE_HEADERS = ['image_id', 'relative_path', 'label', 'group_id', 'session_id', 'scene_type', 'lighting', 'split', 'note']
LOCATION_HEADERS = ['group_id', 'class_name', 'location_description', 'boundary_verified']

def write_csv(path, headers, rows, encoding='utf-8', newline=''):
    with path.open('w', encoding=encoding, newline=newline) as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

def make_dataset(root, inside_sizes, outside_sizes, verified='yes', splits=None):
    dataset = Path(root) / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)
    images, locations = ([], [])
    for class_name, prefix, label, sizes in (('inside', 'in', '1', inside_sizes), ('outside', 'out', '0', outside_sizes)):
        for index, size in enumerate(sizes, start=1):
            group = f'{prefix}_{index:03d}'
            locations.append([group, class_name, f'place {group}', verified])
            for image_index in range(size):
                split_value = (splits or {}).get(group, '')
                images.append([f'{group}_{image_index}', f'images/{class_name}/{group}/{image_index}.jpg', label, group, 's1', 'tag', 'sunny', split_value, f'note {group}'])
    write_csv(dataset / 'images.csv', IMAGE_HEADERS, images)
    write_csv(dataset / 'locations.csv', LOCATION_HEADERS, locations)
    return dataset

def read_images(dataset):
    with (dataset / 'images.csv').open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))
