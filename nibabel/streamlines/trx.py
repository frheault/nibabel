import json
import mmap
import os
import shutil
import tempfile
import zipfile

from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np


def load(input_obj):
    # Check if 0 streamlines, if yes then 0 points is expected (vice-versa)
    # 4x4 affine matrices should contains values (no all-zeros)
    # 3x1 dimensions array should contains values at each position (int)
    # Catch the error if filename do not have a dtype extension (support bool?)
    #

    if os.path.isfile(input_obj):
        was_compressed = False
        with zipfile.ZipFile(input_obj, 'r') as zf:
            for info in zf.infolist():
                if info.compress_type != 0:
                    was_compressed = True
                    break
        if was_compressed:
            with zipfile.ZipFile(input_obj, 'r') as zf:
                tmpdir = tempfile.TemporaryDirectory()
                zf.extractall(tmpdir.name)
                trx = load_from_directory(tmpdir.name)
                trx._uncompressed_folder_handle = tmpdir
                print('File was compressed, call the close() function before quitting!')
        else:
            trx = load_from_zip(input_obj)
    elif os.path.isdir(input_obj):
        trx = load_from_directory(input_obj)

    # Example of robust check for metadata
    for dpg in trx.data_per_group.keys():
        if dpg not in trx.groups.keys():
            raise ValueError('An undeclared group ({}) has data_per_group.'.format(
                dpg))

    return trx


def load_from_zip(filename):
    trx = TrxFile()
    with zipfile.ZipFile(filename, mode='r') as zf:
        with zf.open('header.json') as zf_header:
            data = zf_header.read()
            trx.header = json.loads(data)
            trx.nbr_points = trx.header['nbr_points']
            trx.nbr_streamlines = trx.header['nbr_streamlines']

        files_pointer_size = {}
        for zip_info in zf.filelist:
            elem_filename = zip_info.filename
            if elem_filename == 'header.json':
                continue
            _, ext = os.path.splitext(elem_filename)

            mem_adress = zip_info.header_offset + len(zip_info.FileHeader())
            size = zip_info.file_size / np.dtype(ext[1:]).itemsize

            if size.is_integer():
                files_pointer_size[elem_filename] = mem_adress, int(size)
            else:
                raise ValueError('Wrong size or datatype')

    return create_trx_from_memmap(trx, files_pointer_size, root_zip=filename)


def load_from_directory(directory):
    directory = os.path.abspath(directory)
    trx = TrxFile()
    with open(os.path.join(directory, 'header.json')) as header:
        trx.header = json.load(header)
        trx.nbr_points = trx.header['nbr_points']
        trx.nbr_streamlines = trx.header['nbr_streamlines']

    files_pointer_size = {}
    for root, dirs, files in os.walk(directory):
        for name in files:
            elem_filename = os.path.join(root, name)
            if name == 'header.json':
                continue
            _, ext = os.path.splitext(elem_filename)

            size = os.path.getsize(elem_filename) / np.dtype(ext[1:]).itemsize
            if size.is_integer():
                files_pointer_size[elem_filename] = 0, int(size)
            else:
                raise ValueError('Wrong size or datatype')

    return create_trx_from_memmap(trx, files_pointer_size, root=directory)


def create_trx_from_memmap(trx, dict_pointer_size, root_zip=None, root=None):
    for elem_filename in dict_pointer_size.keys():
        if root_zip:
            filename = root_zip
        else:
            filename = elem_filename
        base, ext = os.path.splitext(elem_filename)
        folder = os.path.dirname(base)
        base = os.path.basename(base)
        mem_adress, size = dict_pointer_size[elem_filename]

        if root is not None and folder.startswith(root.rstrip('/')):
            folder = folder.replace(root, '').lstrip('/')

        if base in ['data', 'offsets', 'lengths'] and folder == '':
            if base == 'data':
                if size != trx.nbr_points*3:
                    raise ValueError('Wrong data size')
                data = np.memmap(filename, mode='r',
                                 offset=mem_adress,
                                 shape=(trx.nbr_points, 3),
                                 dtype=ext[1:])
            elif base == 'offsets':
                if size != trx.nbr_streamlines:
                    raise ValueError('Wrong offsets size')
                offsets = np.memmap(filename, mode='r',
                                    offset=mem_adress,
                                    shape=(trx.nbr_streamlines,),
                                    dtype=ext[1:])
            elif base == 'lengths':
                if size != trx.nbr_streamlines:
                    raise ValueError('Wrong lengths size')
                lengths = np.memmap(filename, mode='r',
                                    offset=mem_adress,
                                    shape=(trx.nbr_streamlines,),
                                    dtype=ext[1:])
        else:
            if folder == 'dps':
                if size != trx.nbr_streamlines:
                    raise ValueError('Wrong dps size')
                trx.data_per_streamline[base] = np.memmap(
                    filename, mode='r', offset=mem_adress,
                    shape=(trx.nbr_streamlines,), dtype=ext[1:])

            elif folder == 'dpp':
                if size != trx.nbr_points:
                    raise ValueError('Wrong dpp size')
                trx.data_per_point[base] = np.memmap(
                    filename, mode='r', offset=mem_adress,
                    shape=(trx.nbr_points, 1), dtype=ext[1:])
            elif folder.startswith('dpg'):
                if size != 1:
                    raise ValueError('Wrong lengths size')
                data_name = os.path.basename(base)
                sub_folder = os.path.basename(folder)
                if sub_folder not in trx.data_per_group:
                    trx.data_per_group[sub_folder] = {}
                trx.data_per_group[sub_folder][data_name] = np.memmap(
                    filename, mode='r', offset=mem_adress,
                    shape=(1,), dtype=ext[1:])
            elif folder == 'groups':
                trx.groups[base] = np.memmap(
                    filename, mode='r', offset=mem_adress,
                    shape=(size,), dtype=ext[1:])

    if data is not None \
        and offsets is not None \
            and lengths is not None:
        trx.streamlines = ArraySequence()
        trx.streamlines._data = data
        trx.streamlines._offsets = offsets
        trx.streamlines._lengths = lengths
    else:
        raise ValueError('Missing essential data')

    for dpp_key in trx.data_per_point:
        tmp = trx.data_per_point[dpp_key]
        trx.data_per_point[dpp_key] = ArraySequence()
        trx.data_per_point[dpp_key]._data = tmp
        trx.data_per_point[dpp_key]._offsets = offsets
        trx.data_per_point[dpp_key]._lengths = lengths

    return trx


def save(trx, filename):
    if os.path.isfile(filename):
        raise ValueError('file already exists, use Concatenate()')
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, 'header.json'), 'w') as out_json:
            json.dump(trx.header, out_json)

        trx.streamlines._data.tofile(os.path.join(tmp_dir, 'data.{}'.format(
            trx.streamlines._data.dtype.name)))
        trx.streamlines._offsets.tofile(os.path.join(tmp_dir, 'offsets.{}'.format(
            trx.streamlines._offsets.dtype.name)))
        trx.streamlines._lengths.tofile(os.path.join(tmp_dir, 'lengths.{}'.format(
            trx.streamlines._lengths.dtype.name)))

        if len(trx.data_per_point.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dpp/'))
        for dpp_key in trx.data_per_point.keys():
            to_dump = trx.data_per_point[dpp_key].get_data()
            to_dump.tofile(os.path.join(tmp_dir,
                                        'dpp/{}.{}'.format(dpp_key,
                                                           to_dump.dtype.name)))
        if len(trx.data_per_streamline.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dps/'))
        for dps_key in trx.data_per_streamline.keys():
            to_dump = trx.data_per_streamline[dps_key]
            to_dump.tofile(os.path.join(tmp_dir,
                                        'dps/{}.{}'.format(dps_key,
                                                           to_dump.dtype.name)))
        if len(trx.data_per_group.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'dpg/'))
        for group_key in trx.data_per_group.keys():
            for dpg_key in trx.data_per_group[group_key].keys():
                os.mkdir(os.path.join(tmp_dir, 'dpg/', group_key))
                to_dump = trx.data_per_group[group_key][dpg_key]
                to_dump.tofile(os.path.join(tmp_dir,
                                            'dpg/{}/{}.{}'.format(group_key,
                                                                  dpg_key,
                                                                  to_dump.dtype.name)))

        if len(trx.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, 'groups/'))
        for group_key in trx.groups.keys():
            to_dump = trx.groups[group_key]
            to_dump.tofile(os.path.join(tmp_dir,
                                        'groups/{}.{}'.format(group_key,
                                                              to_dump.dtype.name)))
        if os.path.splitext(filename)[1]:
            zip_from_folder(tmp_dir, filename)
        else:
            shutil.copytree(tmp_dir, filename)


def zip_from_folder(directory, filename):
    with zipfile.ZipFile(filename, mode='w') as zf:
        for root, dirs, files in os.walk(directory):
            for name in files:
                tmp_filename = os.path.join(root, name)
                zf.write(tmp_filename, tmp_filename.replace(directory+'/', ''))


class TrxFile():
    def __init__(self):
        self.header = None
        self.streamlines = []
        self.groups = {}
        self.data_per_streamline = {}
        self.data_per_point = {}
        self.data_per_group = {}

        self._uncompressed_folder_handle = None

    def close(self):
        if self._uncompressed_folder_handle is not None:
            self._uncompressed_folder_handle.cleanup()
